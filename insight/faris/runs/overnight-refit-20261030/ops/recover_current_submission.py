"""Late delivery of the already audited current winner; no training or clock edits."""
from pathlib import Path
import argparse
import csv
import datetime
import fcntl
import hashlib
import io
import json
import math
import os
import subprocess
import time

EXPECTED_SHA = "4dccde5fd16e635a3c06bf0b5d1464b268e2ab5e7dbf758800759726a76dd43a"
EXPECTED_KS = 0.5316414268134972
COMPETITION = "data-science-competition-logika-ui-2026"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    private = root / "private" / "recovery-20261030"
    private.mkdir(parents=True, exist_ok=True)
    lock = (root / "private" / "submit.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    token_path = root / "private" / "kaggle-token"
    assert token_path.stat().st_mode & 0o077 == 0
    token = token_path.read_text().strip()

    def dump(path, value):
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        temporary.replace(path)

    def call(arguments, label):
        try:
            result = subprocess.run(
                [str(root / "kaggle-env/bin/kaggle"), *arguments],
                env={**os.environ, "KAGGLE_API_TOKEN": token},
                text=True, capture_output=True, timeout=90,
            )
            value = dict(returncode=result.returncode, stdout=result.stdout,
                         stderr=result.stderr, epoch=time.time())
        except subprocess.TimeoutExpired as exc:
            def clean(value):
                return value.decode(errors="replace") if isinstance(value, bytes) else value or ""
            value = dict(returncode=None, stdout=clean(exc.stdout),
                         stderr=clean(exc.stderr), timeout=True, epoch=time.time())
        value["stdout"] = value["stdout"].replace(token, "[REDACTED]")
        value["stderr"] = value["stderr"].replace(token, "[REDACTED]")
        dump(private / (label + ".json"), value)
        return value

    csv_path = root / "submission_CURRENT_localglm-cann.csv"
    audit_path = root / "AUDIT_CURRENT_localglm-cann.json"
    audit = json.loads(audit_path.read_text())
    assert hashlib.sha256(csv_path.read_bytes()).hexdigest() == EXPECTED_SHA
    assert audit["status"] == "PASS" and audit["sha256"] == EXPECTED_SHA
    assert audit["fresh_training"] is True and audit["campaign"] == "overnight-refit-20261030"
    assert audit["selection"]["KS"] == EXPECTED_KS
    assert {(x["seed"], x["fold"]) for x in audit["members"]} == {
        (seed, fold) for seed in (2026, 7731) for fold in range(5)
    }
    reader = csv.DictReader(io.StringIO(csv_path.read_text()))
    assert reader.fieldnames == ["application_id", "default_probability"]
    rows = list(reader)
    ids = [x["application_id"] for x in rows]
    assert len(rows) == 30000 and len(set(ids)) == 30000
    assert ids == json.loads((root / "EXPECTED_TEST_IDS.json").read_text())
    assert all(math.isfinite(float(x["default_probability"])) and
               0 <= float(x["default_probability"]) <= 1 for x in rows)
    original_clock = root / "CLOCK.json"
    clock_sha = hashlib.sha256(original_clock.read_bytes()).hexdigest()
    assert clock_sha == "576f1da25b5636c93a8d93d36c3f993cdfdf300cdade61563799362e195507d0"
    message = "LogikaCurrentLocalGLM_KS05316414_" + EXPECTED_SHA[:12]
    summary = dict(
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        campaign="overnight-refit-20261030", local_development_KS=EXPECTED_KS,
        csv_sha256=EXPECTED_SHA, audit_sha256=hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        rows=30000, original_clock_sha256=clock_sha, original_clock_unchanged=True,
        original_deadline_missed=True, training_performed=False,
        historical_fallback_used=False, authorization="Earlier explicit submission request and latest human continue request",
        message=message,
    )

    def listing(label):
        result = call(["competitions", "submissions", "-c", COMPETITION, "--csv"], label)
        assert result["returncode"] == 0, "Authenticated listing failed; do not issue an upload"
        return list(csv.DictReader(io.StringIO(result["stdout"])))

    before = listing("before")
    found = next((x for x in before if x.get("description") == message or
                  x.get("description") == "Logika0630_" + EXPECTED_SHA[:12]), None)
    if found is None:
        assert not (root / "UPLOAD_INTENT.json").exists(), "Old upload intent requires reconciliation, not another upload"
        intent_path = private / "UPLOAD_INTENT.json"
        with intent_path.open("x") as stream:
            json.dump(summary, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        response = call(["competitions", "submit", "-c", COMPETITION,
                         "-f", str(csv_path), "-m", message], "upload")
        summary["upload_returncode"] = response["returncode"]
        for attempt in range(8):
            try:
                after = listing("after_" + str(attempt))
            except AssertionError:
                after = []
            found = next((x for x in after if x.get("description") == message), None)
            if found:
                break
            time.sleep(5)
    if found is None:
        summary["status"] = "UPLOAD_NOT_CONFIRMED_NO_BLIND_RETRY"
        dump(root / "LATE_SUBMISSION_RECOVERY.json", summary)
        raise RuntimeError("Inspect private upload response and server listings before any further upload")
    summary.update(status="SERVER_OBSERVED", receipt=found)
    dump(root / "LATE_SUBMISSION_RECOVERY.json", summary)
    dump(root / "SUBMITTED.json", summary)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
