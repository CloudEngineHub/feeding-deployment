"""Relabel a deployment user's cross-day memory for the dining/movable table split.

The single "table" was split into two physical locations selected by the mealtime
setting -- `dining_table` (social settings) and `movable_table` (everything else) --
and the per-table preference dims were split with it:

    nav_offset_table   -> nav_offset_{dining,movable}_table
    plate_color_table  -> plate_color_{dining,movable}_table

Days recorded before a split still carry the old single-table field. Because the
setting deterministically implies which physical table a meal used, those records
can be relabeled rather than left to start the new dims cold: a day whose setting
is social was a dining-table meal, any other setting was a movable-table meal.
Relabeling lets the table the user has actually been eating at inherit its real
history instead of the new dims appearing with no entries (which, in per-dim
prediction, renders as a fabricated default that looks like a real observation).

What is rewritten, under
``<log>/<user>/preference_learning/<user>/``:

* ``full_history_memory/day_000N.json`` -- the records that reach a future day's
  LLM prompt. The old key is renamed in ``ground_truth_bundle`` and ``corrected``,
  and ``episode_text`` is REBUILT from the relabeled bundle (not string-patched)
  so field order and value formatting stay canonical.
* ``working_memory/day_000N.json`` -- same key rename in ``corrected``, for
  consistency (only the filenames of these are read, never their contents).

Safety: dry-run by default, an explicit ``--in-place`` to write, a full backup of
the memory directory before the first write, a provenance stamp in each rewritten
file, and a stop-and-list pass that refuses to write anything if any day's setting
is missing or unrecognized.

IMPORTANT: run this with the split already applied to the config (same commit).
``_format_pref_value`` only formats fields it knows are colors/nav offsets; under a
half-applied config a new field name would be rendered with ``str()`` as a Python
dict repr that matches nothing.

Example:

    # inspect
    python -m feeding_deployment.preference_learning.data_generation.migrate_table_split_memory \
        --user aimee
    # apply
    python -m feeding_deployment.preference_learning.data_generation.migrate_table_split_memory \
        --user aimee --in-place
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from feeding_deployment.preference_learning.config.mealtime_context import (
    SETTINGS,
    active_table_for_setting,
)
from feeding_deployment.preference_learning.methods.utils import _episode_text

# old single-table field -> new per-table field prefix
_SPLIT_FIELDS: Dict[str, str] = {
    "nav_offset_table": "nav_offset_",
    "plate_color_table": "plate_color_",
}

_SCHEMA_TAG = "table_split_dining_movable"


def _default_log_root() -> Path:
    """<repo>/src/feeding_deployment/integration/log (where run.py writes)."""
    return Path(__file__).resolve().parents[2] / "integration" / "log"


def _memory_dir(log_root: Path, user: str) -> Path:
    return log_root / user / "preference_learning" / user


def _new_field(old_field: str, table: str) -> str:
    """nav_offset_table + dining_table -> nav_offset_dining_table."""
    return f"{_SPLIT_FIELDS[old_field]}{table}"


def _rename_in_dict(d: Any, old_field: str, new_field: str) -> bool:
    """Rename one key in a dict, preserving its value. True if it was present."""
    if not isinstance(d, dict) or old_field not in d:
        return False
    d[new_field] = d.pop(old_field)
    return True


def _plan_record(record: Dict[str, Any], path: Path) -> Tuple[Optional[str], List[str]]:
    """Resolve which physical table a record's day used.

    Returns (table, problems). ``table`` is None when the record cannot be
    classified, in which case ``problems`` explains why.
    """
    problems: List[str] = []
    context = record.get("context") or {}
    if not isinstance(context, dict):
        problems.append(f"{path.name}: 'context' is not an object")
        return None, problems
    setting = context.get("setting")
    if not setting:
        problems.append(f"{path.name}: no context.setting -- cannot infer the table")
        return None, problems
    if setting not in SETTINGS:
        problems.append(
            f"{path.name}: setting {setting!r} is not a known setting -- cannot infer the table"
        )
        return None, problems
    return active_table_for_setting(setting), problems


def _migrate_record(
    record: Dict[str, Any], table: str, *, rebuild_episode_text: bool
) -> List[str]:
    """Relabel the split fields in one record in place. Returns change lines."""
    changes: List[str] = []
    touched_bundle = False

    for old_field in _SPLIT_FIELDS:
        new_field = _new_field(old_field, table)
        for section in ("ground_truth_bundle", "corrected"):
            if _rename_in_dict(record.get(section), old_field, new_field):
                changes.append(f"{section}: {old_field} -> {new_field}")
                if section == "ground_truth_bundle":
                    touched_bundle = True

    # Rebuild episode_text from the relabeled bundle rather than patching the
    # string: this also restores canonical PREF_FIELDS ordering and formatting.
    if rebuild_episode_text and touched_bundle and "episode_text" in record:
        bundle = record.get("ground_truth_bundle") or {}
        old_text = record["episode_text"]
        new_text = _episode_text(
            day=record.get("day"), context=record.get("context") or {}, prefs=bundle
        )
        if new_text != old_text:
            record["episode_text"] = new_text
            changes.append("episode_text: rebuilt from the relabeled bundle")

    if changes:
        stamp = record.setdefault("schema_migration", {})
        if isinstance(stamp, dict):
            stamp[_SCHEMA_TAG] = {
                "table": table,
                "script": Path(__file__).name,
            }
    return changes


def _load(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def migrate_user(
    user: str,
    log_root: Path,
    *,
    in_place: bool,
    backup: bool = True,
) -> int:
    mem_dir = _memory_dir(log_root, user)
    if not mem_dir.is_dir():
        print(f"[migrate] no memory directory for user {user!r} at {mem_dir}")
        return 1

    targets = [
        (mem_dir / "full_history_memory", True),   # rebuild episode_text
        (mem_dir / "working_memory", False),       # no episode_text stored
        (mem_dir / "episodic_memory", False),      # absent for full-history users
        (mem_dir / "long_term_memory", False),
    ]

    planned: List[Tuple[Path, Dict[str, Any], str, bool, List[str]]] = []
    problems: List[str] = []

    for sub_dir, rebuild in targets:
        if not sub_dir.is_dir():
            continue
        for path in sorted(sub_dir.glob("day_*.json")):
            record = _load(path)
            if record is None:
                problems.append(f"{sub_dir.name}/{path.name}: unreadable or not a JSON object")
                continue
            has_old = any(
                isinstance(record.get(sec), dict) and old in record[sec]
                for old in _SPLIT_FIELDS
                for sec in ("ground_truth_bundle", "corrected")
            )
            if not has_old:
                continue  # already migrated (or never had the field)
            table, record_problems = _plan_record(record, path)
            problems.extend(f"{sub_dir.name}/{p}" for p in record_problems)
            if table is None:
                continue
            changes = _migrate_record(record, table, rebuild_episode_text=rebuild)
            if changes:
                planned.append((path, record, table, rebuild, changes))

    # Stop and list: refuse to write a partial migration.
    if problems:
        print("[migrate] REFUSING to write -- unresolved records:")
        for p in problems:
            print(f"  - {p}")
        return 1

    if not planned:
        print(f"[migrate] nothing to do for {user!r} (no pre-split fields found).")
        return 0

    print(f"[migrate] user={user!r}  memory={mem_dir}")
    for path, _record, table, _rebuild, changes in planned:
        rel = path.relative_to(mem_dir)
        print(f"  {rel}  (setting -> {table})")
        for line in changes:
            print(f"      {line}")

    if not in_place:
        print(
            f"\n[migrate] DRY RUN -- {len(planned)} file(s) would change. "
            "Re-run with --in-place to apply."
        )
        return 0

    if backup:
        backup_dir = mem_dir.parent / f"{mem_dir.name}_pre_{_SCHEMA_TAG}"
        if backup_dir.exists():
            print(f"[migrate] backup already exists, not overwriting: {backup_dir}")
        else:
            shutil.copytree(mem_dir, backup_dir)
            print(f"[migrate] backed up {mem_dir} -> {backup_dir}")

    for path, record, _table, _rebuild, _changes in planned:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
            f.write("\n")
    print(f"[migrate] rewrote {len(planned)} file(s).")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--user", required=True, help="Deployment user (log/<user>).")
    parser.add_argument(
        "--log-root",
        type=str,
        default="",
        help="Override the log root (default: the repo's integration/log).",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Actually rewrite the files (default is a dry run).",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip the pre-write backup copy (not recommended).",
    )
    args = parser.parse_args()

    log_root = (
        Path(args.log_root).expanduser().resolve() if args.log_root else _default_log_root()
    )
    return migrate_user(
        args.user,
        log_root,
        in_place=args.in_place,
        backup=not args.no_backup,
    )


if __name__ == "__main__":
    sys.exit(main())
