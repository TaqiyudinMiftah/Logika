# Faris (DEFAULT A100)

## Separate connection profiles

Use `profile.py` to select a destination explicitly. The original commands in
this folder still default to Faris/node480.

| Profile | Host | Account |
|---|---|---|
| `faris` | node480-ai-hub.ub.ac.id | fariselhakim |
| `node380` | node380-ai-hub.ub.ac.id | taqiyudinmiftah |

```bash
python profile.py list
python profile.py faris term 'whoami; hostname'
python profile.py node380 term 'whoami; hostname'
python profile.py node380 exec --session term -- 'pwd'
python profile.py faris sessions
python profile.py node380 sessions
```

Each profile has its own credential file, kernel/session cache, working
directories, transfer manifest cache, and shell history. Both may use the
session name `term` without sharing a kernel ID. Profile commands read only
their named token environment variable (`A100_FARIS_API_TOKEN` or
`A100_NODE380_API_TOKEN`); they never reuse a generic token from another host.

To refresh a connection, run `python profile.py node380 login` (or `faris`).
The password prompt disables terminal echo and does not save the password.
Login reuses the running Jupyter server's existing credential, stores it in a
mode-600 ignored file, and deletes its temporary remote handoff file. It does
not create a new API token.

Existing push/pull commands also accept profiles, for example:

```bash
python profile.py node380 pull projects/example/features.parquet ./features.parquet
python profile.py faris push ./features.parquet projects/example/features.parquet
```

Those two commands relay the file through this computer. They are examples;
profile setup itself does not move competition data or start training. The
existing transfer commands use helper kernels, as described below.

Configuration lives in `profiles/faris/profile.json` and
`profiles/node380/profile.json`. Check live cgroup limits when choosing a node;
host-level free RAM and container limits describe different resources.

- Host: `https://node480-ai-hub.ub.ac.id/`
- User: `fariselhakim`
- Client: `./jupyter_remote.py` (defaults to this host + user)
- Token file: `./sysinfo_fariselhakim.txt` (mode 600, never commit) or `JUPYTERHUB_API_TOKEN` env
- Local session state: `~/.cache/a100_faris_sessions.json` (separate from Nafis)

This is the DEFAULT. Use it unless the user explicitly says Nafis.

## Use one persistent control kernel

Use **`term` for all ordinary remote commands in this folder**. This is also
the session used by `a100-term`, including one-shot commands, interactive mode,
and tmux commands. The first command creates a kernel only if `term` has no live
kernel; later commands reuse its ID. Do not invent a new session for each task.

```bash
./a100-term 'whoami; pwd'
./a100-term 'nvidia-smi'
./jupyter_remote.py exec --session term -- 'free -h'
./jupyter_remote.py sessions  # lists existing kernels without creating one
```

Run commands sequentially on this kernel. An idle kernel with no browser
connections is normal for this workflow: retain `term` between commands.
`python train.py` still launches a separate training process; the shared kernel
is the command transport, not the training interpreter.

Avoid bare `jupyter_remote.py 'command'` and `exec` without `--session`: those
create temporary kernels. `run-many` and `:par` create worker kernels. The
`bg`, `poll`, `log`, `jobs`, transfer and sync commands use additional helper
kernels, even if given `--session term`; `doctor`, `vram`, and `health` also
use temporary kernels. For the single-kernel workflow, run their shell
equivalents through `a100-term` instead.

For a long job, start it in remote tmux through the same control kernel, then
inspect its log with another `a100-term` command:

```bash
./a100-term --tmux 'cd "$HOME/Faris/projects/joints-2026"'
./a100-term --tmux 'python train.py > train.log 2>&1'
./a100-term 'tail -n 30 "$HOME/Faris/projects/joints-2026/train.log"'
```

## First: create a project (trackable)

Do this before any run, so work is grouped per project:

```bash
# local
mkdir -p projects/joints-2026

# remote (via this client)
./jupyter_remote.py exec --session term -- 'mkdir -p "$HOME/Faris/projects/joints-2026"'
./jupyter_remote.py exec --session term -- 'cd ~/Faris/projects/joints-2026'
```

The second command records the remote working directory for `term`;
`a100-term` and subsequent `exec --session term` commands reuse it. The remote
JOINTS project observed during the memory inspection was under
`/home/fariselhakim/Faris/projects/joints-2026`.

## Get a token (do once per login)

1. Log in at `https://node480-ai-hub.ub.ac.id/hub/login` as `fariselhakim`.
2. Open a terminal in Jupyter, run `env | grep JUPYTERHUB_API_TOKEN`, save the value.
3. Save locally (never commit):
   `printf 'JUPYTERHUB_API_TOKEN=<token>\n' > sysinfo_fariselhakim.txt && chmod 600 sysinfo_fariselhakim.txt`
4. Verify: `./a100-term 'whoami; pwd'` should print `fariselhakim`.

## Examples

```bash
./a100-term 'nvidia-smi | head -20'
./jupyter_remote.py exec --session term -- 'python train.py'  # short runs only
./a100-term 'tail -n 30 train.log'
```

Credentials live only in the local 600 file or env. Do not paste tokens into chat/logs.

## a100-term (added 2026-10-01)

Local terminal for the A100, no SSH needed:

```bash
./a100-term 'nvidia-smi | head -20'   # one-shot command, reuses term kernel
./a100-term                            # interactive shell (sticky session+cd, :quit exits)
./a100-term --tmux 'cmd...'            # persistent tmux session 'a100' on the remote
```

Tmux mode survives disconnects; re-run to refresh pane output. Helper lib lives in
`.helpers/` (workspace copy of the kernel-channel runner; `/tmp` gets wiped).

## Inspect and clean up your processes

`a100-cleanup` uses an **existing, idle `term` kernel** and never creates one.
Its default mode only reports your largest processes, unused helper kernel
candidates, duplicate `nvtop` monitors, and actual zombies. Start/reuse `term`
with `a100-term` first if needed.

```bash
python3 ./a100-cleanup
python3 ./a100-cleanup --close-idle-helpers --close-duplicate-monitors  # preview
python3 ./a100-cleanup --close-idle-helpers --close-duplicate-monitors --apply
```

Helper cleanup requires all of these: your process, a locally tracked session,
idle for at least 24 hours, zero connections, no notebook attachment, and no
child processes. It always retains the control kernel and rechecks each
candidate immediately before requesting shutdown. Shutting down a kernel
discards its in-memory variables. Unknown kernels, notebook kernels, busy
kernels, and kernels supervising jobs stay running. Monitor cleanup sends
`SIGTERM` to your duplicate `nvtop` processes and retains the newest one.

To close a particular reviewed session manually, use JupyterLab's **Running**
panel or `./jupyter_remote.py session-kill SESSION_NAME`. Check that it has no
work you need before doing so. Do not use blanket `pkill python`, and keep other
users' jobs, JupyterHub, and your Jupyter server running.

A zombie (`Z` / `<defunct>`) has already exited and holds no application RAM.
`kill -9` cannot remove it; its parent must collect its exit status with
`wait()` / `waitpid()`. Idle Jupyter kernels are live processes, not zombies.
[Linux wait documentation](https://man7.org/linux/man-pages/man2/waitpid.2.html)

```bash
python3 ./a100-cleanup --reap-zombies          # preview
python3 ./a100-cleanup --reap-zombies --apply  # best-effort reaping
```

This reaps zombie children of the control kernel directly; for other parents
owned by your account it sends `SIGCHLD` to request that their existing handler
reap children. It reports how many zombies remain and never terminates parents
or signals another user's processes. A parent without a reaping handler may
need an application fix or a deliberate restart. Zombies adopted by root/PID 1
require administrator attention; the container should use an init/reaper.

## Remote installs (2026-10-01)

- Node 24: `~/tools/node24` (system node is v10; `~/.local` is root-owned, avoided)
- Codex 0.159.2: `~/.npm-global` via npm (`codex` on PATH through `.bashrc`)
- Muse bridge: `~/opt/lib/codex-muse-app-proxy.mjs` + `~/opt/bin/codex-muse` launcher,
  profile `~/.codex/muse-xhigh.config.toml`, provider `~/.codex/config.toml`,
  catalog `~/.codex/muse-spark-model-catalog.json`
- Auth: `~/.codex/auth.json` (ChatGPT login, mode 600) + `~/.codex/muse-opencode.env`
- tmux 3.7: `~/tools/tmux-env` via micromamba (`~/tools/bin/micromamba`, conda-forge)
- PATH additions are in remote `~/.bashrc`.

## Known limits

- `chatgpt.com` is blocked from the A100 network (403/timeouts), so stock `codex`
  with ChatGPT login cannot reach the model there. `codex-muse` works (proof:
  `MUSE_A100_OK`, 2026-10-01) because `opencode.ai` is reachable. `api.openai.com`
  is also reachable (JSON 401 on dummy key), so an `OPENAI_API_KEY` route would
  work if you add one later.
- The Jupyter kernel channel handles one command per kernel at a time. Reuse
  `term` sequentially, keep individual waits under ~150s, and put long-running
  work in tmux so the control kernel remains available.
