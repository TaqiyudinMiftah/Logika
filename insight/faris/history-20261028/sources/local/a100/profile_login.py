"""Sign in using an in-memory cookie jar and reuse the running server token.

The password is read with getpass and is never saved. No new API token is
created: the running Jupyter server's existing environment credential is reused.
"""
from __future__ import annotations

import getpass
from html.parser import HTMLParser
import http.cookiejar
import json
import os
from pathlib import Path
import tempfile
import time
from urllib.parse import urlencode, urlparse, quote
import urllib.error
import urllib.request
import uuid


class LoginFields(HTMLParser):
    def __init__(self):
        super().__init__()
        self.xsrf = None
        self.names = set()

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "input" and attrs.get("name"):
            self.names.add(attrs["name"])
        if tag == "input" and attrs.get("name") == "_xsrf":
            self.xsrf = attrs.get("value")


def write_private_token(path, token):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".credential-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            os.fchmod(stream.fileno(), 0o600)
            stream.write("JUPYTERHUB_API_TOKEN=" + token + "\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def login(profile):
    cookies = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookies))
    login_url = profile["hub_url"] + "login"
    with opener.open(login_url, timeout=30) as response:
        html = response.read().decode("utf-8", errors="replace")
    fields = LoginFields()
    fields.feed(html)
    if not {"username", "password"}.issubset(fields.names):
        raise RuntimeError("The Hub did not expose its username/password login form")
    print(f"Sign in to {urlparse(profile['base_url']).hostname} as {profile['username']}", flush=True)
    password = [REDACTED_CREDENTIAL].getpass("Password (not saved): ")
    form = {"username": profile["username"], "password": password}
    if fields.xsrf:
        form["_xsrf"] = fields.xsrf
    body = urlencode(form).encode()
    request = urllib.request.Request(login_url, data=body, method="POST")
    with opener.open(request, timeout=45) as response:
        result = response.read().decode("utf-8", errors="replace")
        final_url = response.geturl()
    del password, body, request, form
    if "Invalid username or password" in result or urlparse(final_url).path.rstrip("/").endswith("/login"):
        raise RuntimeError("The Hub rejected the supplied login")
    with opener.open(profile["base_url"] + "lab", timeout=60) as response:
        response.read()

    def api(path, method="GET", data=None):
        url = profile["base_url"] + path
        parsed = urlparse(url)
        candidates = [cookie for cookie in cookies
                      if cookie.name == "_xsrf" and parsed.path.startswith(cookie.path)
                      and parsed.hostname.endswith(cookie.domain.lstrip("."))]
        headers = {"Content-Type": "application/json"}
        if candidates:
            headers["X-XSRFToken"] = max(candidates, key=lambda item: len(item.path)).value
        request = urllib.request.Request(url, method=method, headers=headers,
                                         data=None if data is None else json.dumps(data).encode())
        with opener.open(request, timeout=45) as response:
            content = response.read()
        return json.loads(content) if content else None

    # Verify this is an authenticated user-server session before executing a
    # one-shot, isolated terminal. Existing terminals/notebook kernels are unused.
    api("api/kernels")
    filename = "a100_profile_bootstrap_" + uuid.uuid4().hex + ".json"
    code = (
        "import json,os; from pathlib import Path; "
        f"p=Path.home()/{filename!r}; "
        "token=os.environ.get('JUPYTERHUB_API_TOKEN'); "
        "assert token, 'Existing server credential is unavailable'; "
        "fd=os.open(str(p),os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600); "
        "f=os.fdopen(fd,'w'); json.dump({'token':token},f); f.close()"
    )
    server_client = None
    try:
        api("api/terminals", "POST", {"shell_command": ["/usr/bin/python3", "-c", code]})
        payload = None
        for attempt in range(12):
            try:
                content = api("api/contents/" + quote(filename) + "?format=text&type=file")
                payload = json.loads(content["content"])
                break
            except urllib.error.HTTPError as error:
                if error.code != 404 or attempt == 11:
                    raise
                time.sleep(1)
        token = payload.get("token") if payload else None
        if not isinstance(token, str) or not token or any(char.isspace() for char in token):
            raise RuntimeError("The existing server credential could not be retrieved")
        from jupyter_remote import JupyterClient
        server_client = JupyterClient(profile["base_url"], token)
        server_client.request("GET", "api/kernels")
        write_private_token(profile["sysinfo"], token)
    finally:
        try:
            if server_client is not None:
                server_client.request("DELETE", "api/contents/" + quote(filename))
            else:
                api("api/contents/" + quote(filename), "DELETE")
        except urllib.error.HTTPError as error:
            if error.code != 404:
                raise RuntimeError("Could not remove the temporary credential handoff file") from None
        except RuntimeError:
            raise RuntimeError("Could not remove the temporary credential handoff file") from None
    print(f"Connected. Existing server credential saved privately to {profile['sysinfo']}", flush=True)
