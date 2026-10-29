#!/usr/bin/env python3
"""Initialize or resume a local competition workspace without resetting its deadline.

Usage: init_competition.py ROOT SLUG METRIC {lower,higher} HOURS
  --train PATH --test PATH --compute DESCRIPTION --rules FILE [--resume]
Relative input paths are resolved against ROOT. Uses only the standard library.
"""
import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parents[1]
STATE_NAME = "TRAINING_STATE.json"
BRIEF_NAME = "TRAINING_BRIEF.md"


def positive_hours(value: str) -> float:
    try:
        hours = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("hours must be a positive finite number") from exc
    if not math.isfinite(hours) or hours <= 0:
        raise argparse.ArgumentTypeError("hours must be a positive finite number")
    try:
        duration = timedelta(hours=hours)
        datetime.now(timezone.utc) + duration
    except (OverflowError, ValueError) as exc:
        raise argparse.ArgumentTypeError("hours exceed the supported deadline range") from exc
    if duration.total_seconds() <= 0:
        raise argparse.ArgumentTypeError("hours must represent at least one microsecond")
    return hours


def nonempty(value: str, label: str) -> str:
    value = value.strip()
    if not value or value.lower() in {"unspecified", "unknown", "tbd"}:
        raise ValueError(f"{label} must be supplied")
    return value


def input_path(root: Path, value: str, label: str, *, file_only: bool = False) -> Path:
    value_path = Path(value).expanduser()
    path = (root / value_path if not value_path.is_absolute() else value_path).resolve(strict=True)
    if path.is_file():
        if path.stat().st_size == 0:
            raise ValueError(f"{label} is empty: {path}")
        with path.open("rb") as handle:
            handle.read(1)
    elif path.is_dir() and not file_only:
        # Directory-based image/audio datasets are valid inputs too.
        if not any(item.is_file() and item.stat().st_size > 0 for item in path.rglob("*")):
            raise ValueError(f"{label} directory contains no nonempty files: {path}")
    else:
        raise ValueError(f"{label} must be a {'file' if file_only else 'file or data directory'}: {path}")
    return path


def parse_utc(value: str, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, AttributeError, TypeError) as exc:
        raise ValueError(f"invalid {label} timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{label} must include a UTC offset")
    return parsed.astimezone(timezone.utc)


def validate_deadline(info: dict) -> dict:
    start = parse_utc(info.get("start_utc"), "start_utc")
    end = parse_utc(info.get("end_utc"), "end_utc")
    try:
        hours = positive_hours(str(info.get("hours")))
    except argparse.ArgumentTypeError as exc:
        raise ValueError(f"invalid stored deadline: {exc}") from exc
    if end <= start or not math.isclose((end - start).total_seconds(), hours * 3600,
                                        rel_tol=1e-9, abs_tol=1e-6):
        raise ValueError("stored deadline start, end, and hours disagree")
    return {"start_utc": start.isoformat(), "end_utc": end.isoformat(), "hours": hours}


def read_deadline(root: Path) -> dict:
    text = (root / "DEADLINE.md").read_text(encoding="utf-8")
    info = {}
    for key in ("start_utc", "end_utc", "hours"):
        matches = re.findall(rf"^\s*(?:-\s*)?{key}:\s*(\S+)\s*$", text, flags=re.MULTILINE)
        if len(matches) != 1:
            raise ValueError(f"DEADLINE.md must contain exactly one {key}")
        info[key] = matches[0]
    return validate_deadline(info)


def render_template(name: str, values: dict) -> str:
    text = (SKILL_DIR / "templates" / name).read_text(encoding="utf-8")
    # One substitution pass keeps user-supplied rules verbatim, including angle brackets.
    return re.sub(r"<([a-z_]+)>", lambda match: str(values.get(match[1], match[0])), text)


def atomic_write(path: Path, text: str) -> None:
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_config(args: argparse.Namespace) -> dict:
    root = Path(args.root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError("competition root must be an existing directory")
    slug = nonempty(args.slug, "slug")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", slug):
        raise ValueError("slug must contain letters, digits, dots, underscores, or hyphens")
    train = input_path(root, args.train, "train data")
    test = input_path(root, args.test, "test data")
    if train == test or train.samefile(test):
        raise ValueError("train and test data must be distinct paths")
    rules_file = input_path(root, args.rules, "competition rules", file_only=True)
    rules = rules_file.read_text(encoding="utf-8")
    if not rules.strip():
        raise ValueError("competition rules must contain text")
    return {"schema_version": 1, "root": str(root), "slug": slug,
            "metric": nonempty(args.metric, "metric"), "direction": args.direction,
            "train": str(train), "test": str(test),
            "compute": nonempty(args.compute, "compute description"),
            "rules_file": str(rules_file),
            "rules_sha256": hashlib.sha256(rules.encode("utf-8")).hexdigest(),
            "rules_text": rules}


def initialize(args: argparse.Namespace) -> tuple[dict, bool]:
    config = prepare_config(args)
    root = Path(config["root"])
    for folder in ("experiments", "models", "submissions_held", "reports"):
        destination = root / folder
        if destination.is_symlink() or (destination.exists() and not destination.is_dir()):
            raise ValueError(f"campaign output must be a directory in the competition root: {destination}")
    for filename in (STATE_NAME, BRIEF_NAME, "DEADLINE.md", "AGENTS.md", "learnings.md",
                     "experiments/EXPERIMENTS.md"):
        destination = root / filename
        if filename in (STATE_NAME, BRIEF_NAME, "DEADLINE.md") and destination.is_symlink():
            raise ValueError(f"campaign state must be a file in the competition root: {destination}")
        if destination.exists() and not destination.is_file():
            raise ValueError(f"existing campaign file is not a regular file: {destination}")
    state_path = root / STATE_NAME
    resumed = state_path.exists()
    if args.resume and not resumed:
        raise ValueError(f"cannot resume without {STATE_NAME}; initialize once first")
    if resumed:
        old = json.loads(state_path.read_text(encoding="utf-8"))
        if not isinstance(old, dict) or old.get("schema_version") != 1:
            raise ValueError(f"unsupported {STATE_NAME} schema")
        changed = [key for key, value in config.items() if old.get(key) != value]
        if changed:
            raise ValueError("resume configuration changed: " + ", ".join(changed)
                             + "; existing campaign and deadline were preserved")
        deadline = validate_deadline(old)
        if (root / "DEADLINE.md").exists() and read_deadline(root) != deadline:
            raise ValueError(f"DEADLINE.md disagrees with {STATE_NAME}; deadline was preserved")
        state = old
    elif (root / "DEADLINE.md").exists():
        # Migrate an older skill workspace without granting it a new deadline.
        deadline = read_deadline(root)
        state = {**config, **deadline}
        resumed = True
    else:
        now = datetime.now(timezone.utc)
        deadline = {"start_utc": now.isoformat(),
                    "end_utc": (now + timedelta(hours=args.hours)).isoformat(), "hours": args.hours}
        state = {**config, **deadline}

    values = {**state, "competition_root": state["root"], "metric_name": state["metric"],
              "train_path": state["train"], "test_path": state["test"],
              "compute_description": state["compute"], "pasted_rules": state["rules_text"]}
    brief = render_template("TRAINING_BRIEF.template.md", values)
    brief_path = root / BRIEF_NAME
    if brief_path.exists() and brief_path.read_text(encoding="utf-8") != brief:
        raise ValueError(f"{BRIEF_NAME} differs from campaign inputs; existing file was preserved")
    # All validation occurs before creating any campaign output.
    for folder in ("experiments", "models", "submissions_held", "reports"):
        (root / folder).mkdir(exist_ok=True)
    if not (root / "AGENTS.md").exists():
        atomic_write(root / "AGENTS.md", render_template("AGENTS.template.md", values))
    if not (root / "DEADLINE.md").exists():
        atomic_write(root / "DEADLINE.md", render_template("DEADLINE.template.md", values))
    if not brief_path.exists():
        atomic_write(brief_path, brief)
    for destination, template in ((root / "experiments" / "EXPERIMENTS.md", "EXPERIMENTS.template.md"),
                                  (root / "learnings.md", "learnings.template.md")):
        if not destination.exists():
            atomic_write(destination, render_template(template, values))
    if not state_path.exists():
        atomic_write(state_path, json.dumps(state, indent=2, ensure_ascii=False) + "\n")
    return state, resumed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    parser.add_argument("slug")
    parser.add_argument("metric")
    parser.add_argument("direction", choices=("lower", "higher"))
    parser.add_argument("hours", type=positive_hours)
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--compute", required=True)
    parser.add_argument("--rules", required=True)
    parser.add_argument("--resume", action="store_true", help="require an existing initialized campaign")
    args = parser.parse_args()
    try:
        state, resumed = initialize(args)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"{'resumed' if resumed else 'initialized'}: {state['root']}")
    print(f"deadline preserved: {state['end_utc']}" if resumed else f"deadline: {state['end_utc']}")
    print(f"worker brief: {Path(state['root']) / BRIEF_NAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
