"""Migrate a deployment dataset authored under the PRE-refactor preference
schema (before commit d748b4c5, "collapse autocontinue prefs into single
per-page confirm param") to the CURRENT schema in config/preference_bundle.py.

Why this exists
---------------
Datasets generated on/before 2026-07-16 store the old two-field model:
    confirm_*  in {"no", "yes (with auto-continue countdown)",
                   "yes (without any auto-continue)"}
    wait_before_autocontinue_{feeding_pickup,mealprep,task_selection}
               in {"15 sec", "30 sec", "60 sec", "no autocontinue"}

The current schema collapses "whether to confirm" and "how long the countdown
is" into a single per-page field:
    confirm_*  in {"skip", "countdown (15 sec)", "countdown (30 sec)",
                   "countdown (60 sec)", "wait for me"}
and splits the feeding-side waits into two standalone dims:
    wait_before_autocontinue_task_selection   (kept, values re-vocabularied)
    wait_before_autocontinue_bite_selection   (NEW)
while dropping wait_before_autocontinue_{feeding_pickup,mealprep} entirely.

Running the evaluator on an un-migrated dataset silently scores those dims 0.0
(the model predicts current-vocabulary options that can never match old truth
strings) and drops the renamed/removed dims via exclude_missing_dims -- see the
07-17 vs 07-21 per-dimension plots.

What this script does NOT touch: continuous dims (colors, nav offsets), and any
categorical dim whose stored values are already valid under the current schema
(e.g. microwave_time 1/2 min, detect_* open-mouth/button/perception).

Conflict policy (STOP & LIST)
-----------------------------
A day where confirm_* == "yes (with auto-continue countdown)" but its paired
wait dim == "no autocontinue" is contradictory under the single-field model.
This script prints every such day and ABORTS without writing, so the source
ground truth can be fixed by hand and the migration re-run.

Usage:
    PYTHONPATH=src python3 -m feeding_deployment.preference_learning.data_generation.migrate_dataset_to_current_schema \
        --input  out_manual_run_2026_07_16__15_41_53/manual_1__dep1__30d.json \
        [--output <path>]        # default: <input stem>__schema_migrated.json
        [--in-place]             # overwrite the input file instead
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

from feeding_deployment.preference_learning.config.preference_bundle import (
    PREFERENCE_BUNDLE,
)

# ---- current-schema option lookup (source of truth) ------------------------
CURRENT_OPTIONS: Dict[str, List[str]] = {
    dim.field: list(dim.options) for dim in PREFERENCE_BUNDLE
}
CURRENT_FIELDS = set(CURRENT_OPTIONS)

# ---- old -> new value maps -------------------------------------------------
# Old standalone wait vocabulary -> current countdown vocabulary.
WAIT_TO_CURRENT = {
    "15 sec": "countdown (15 sec)",
    "30 sec": "countdown (30 sec)",
    "60 sec": "countdown (60 sec)",
    "no autocontinue": "wait for me",
}
# Old detect_* readiness/completion "autocontinue" was renamed.
DETECT_RENAME = {"autocontinue": "proceed automatically after a pause"}
DETECT_FIELDS = {
    "detect_user_ready_for_initiating_transfer_feeding",
    "detect_user_ready_for_initiating_transfer_drinking",
    "detect_user_ready_for_initiating_transfer_wiping",
    "detect_user_completed_transfer_feeding",
    "detect_user_completed_transfer_drinking",
    "detect_user_completed_transfer_wiping",
}
# confirm field -> the old wait dim it absorbs its countdown length from.
CONFIRM_PAIR = {
    "confirm_feeding_pickup": "wait_before_autocontinue_feeding_pickup",
    "confirm_navigation_arrival": "wait_before_autocontinue_mealprep",
    "confirm_manipulation": "wait_before_autocontinue_mealprep",
}
# Old wait dims consumed by the migration and removed from the output.
DROP_FIELDS = {
    "wait_before_autocontinue_feeding_pickup",
    "wait_before_autocontinue_mealprep",
}


def _choice(prefs: Dict[str, Any], field: str) -> Any:
    rec = prefs.get(field)
    if isinstance(rec, dict):
        return rec.get("choice")
    return rec


def _set_choice(prefs: Dict[str, Any], field: str, value: str, note: str) -> None:
    """Write a migrated value, keeping the record's dict shape + a rationale
    breadcrumb so the provenance is visible in the dataset."""
    rec = prefs.get(field)
    if isinstance(rec, dict):
        rec = dict(rec)
    else:
        rec = {}
    rec["choice"] = value
    rec["rationale"] = note
    prefs[field] = rec


def _merge_confirm(confirm_val: Any, wait_val: Any) -> Tuple[str, bool]:
    """Return (new_value, is_conflict)."""
    if confirm_val == "no":
        return "skip", False
    if confirm_val == "yes (without any auto-continue)":
        return "wait for me", False
    if confirm_val == "yes (with auto-continue countdown)":
        if wait_val in {"15 sec", "30 sec", "60 sec"}:
            return f"countdown ({wait_val})", False
        return "", True  # countdown + "no autocontinue"/missing -> contradictory
    # Already-migrated or unexpected value: pass through, validate later.
    return confirm_val, False


def migrate_day(prefs: Dict[str, Any], day_no: int, conflicts: List[str]) -> Dict[str, Any]:
    out = copy.deepcopy(prefs)

    # 1) confirm_* merges (confirm choice + paired wait length -> one field).
    for confirm_f, wait_f in CONFIRM_PAIR.items():
        if confirm_f not in out:
            continue
        cval = _choice(out, confirm_f)
        wval = _choice(out, wait_f)
        new_val, is_conflict = _merge_confirm(cval, wval)
        if is_conflict:
            conflicts.append(
                f"  day {day_no:>2}: {confirm_f}={cval!r} + {wait_f}={wval!r}"
            )
            continue
        _set_choice(out, confirm_f, new_val, "migrated:confirm_merge")

    # 2) NEW wait_before_autocontinue_bite_selection derived from old
    #    feeding_pickup wait (before it is dropped).
    fp_wait = _choice(out, "wait_before_autocontinue_feeding_pickup")
    if fp_wait in WAIT_TO_CURRENT:
        _set_choice(
            out,
            "wait_before_autocontinue_bite_selection",
            WAIT_TO_CURRENT[fp_wait],
            "migrated:derived_from_feeding_pickup_wait",
        )

    # 3) task_selection wait: re-vocabulary in place.
    ts_wait = _choice(out, "wait_before_autocontinue_task_selection")
    if ts_wait in WAIT_TO_CURRENT:
        _set_choice(
            out,
            "wait_before_autocontinue_task_selection",
            WAIT_TO_CURRENT[ts_wait],
            "migrated:wait_revocab",
        )

    # 4) detect_* autocontinue rename (no-op if unused in this dataset).
    for f in DETECT_FIELDS:
        val = _choice(out, f)
        if val in DETECT_RENAME:
            _set_choice(out, f, DETECT_RENAME[val], "migrated:detect_rename")

    # 5) drop consumed wait dims.
    for f in DROP_FIELDS:
        out.pop(f, None)

    return out


def validate_day(prefs: Dict[str, Any], day_no: int, bad: List[str]) -> None:
    """Every categorical field present must hold a value valid under the current
    schema. Fields with empty options (text/color/nav/continuous) are skipped."""
    for field, val in prefs.items():
        opts = CURRENT_OPTIONS.get(field)
        if not opts:  # unknown-to-current, or free-form/continuous -> skip
            continue
        choice = val.get("choice") if isinstance(val, dict) else val
        if choice not in opts:
            bad.append(f"  day {day_no:>2}: {field}={choice!r} not in {opts}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="Path to the old-schema dataset JSON.")
    ap.add_argument("--output", default=None, help="Output path (default: <stem>__schema_migrated.json).")
    ap.add_argument("--in-place", action="store_true", help="Overwrite the input file.")
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"error: {in_path} not found", file=sys.stderr)
        return 2
    data = json.loads(in_path.read_text())

    days = data.get("days", [])
    conflicts: List[str] = []
    migrated_days = []
    for day_rec in days:
        day_no = int(day_rec.get("day", 0))
        prefs = day_rec.get("preferences", {}) or {}
        new_prefs = migrate_day(prefs, day_no, conflicts)
        new_rec = dict(day_rec)
        new_rec["preferences"] = new_prefs
        migrated_days.append(new_rec)

    if conflicts:
        print(
            "CONFLICT: confirm_* == 'yes (with auto-continue countdown)' paired with\n"
            "a 'no autocontinue' wait cannot be expressed as a single field.\n"
            "Fix these days in the SOURCE dataset (set confirm to\n"
            "'yes (without any auto-continue)' to mean wait-indefinitely, or give the\n"
            "paired wait a real countdown), then re-run.\n\n"
            f"{len(conflicts)} conflicting day(s):",
            file=sys.stderr,
        )
        print("\n".join(conflicts), file=sys.stderr)
        print("\nNo output written.", file=sys.stderr)
        return 1

    # Post-migration schema validation (catches e.g. microwave_time '3 min').
    bad: List[str] = []
    for rec in migrated_days:
        validate_day(rec["preferences"], int(rec.get("day", 0)), bad)
    if bad:
        print(
            "VALIDATION: migrated values still outside the current schema options\n"
            "(these need a manual decision -- no automatic mapping defined):\n",
            file=sys.stderr,
        )
        print("\n".join(bad), file=sys.stderr)
        print("\nNo output written.", file=sys.stderr)
        return 1

    out_data = dict(data)
    out_data["days"] = migrated_days
    out_data["schema_migration"] = {
        "from": "pre-d748b4c5 (two-field confirm/wait)",
        "to": "current preference_bundle.py",
        "script": "migrate_dataset_to_current_schema.py",
    }

    out_path = in_path if args.in_place else (
        Path(args.output) if args.output
        else in_path.with_name(in_path.stem + "__schema_migrated.json")
    )
    out_path.write_text(json.dumps(out_data, indent=2))
    print(f"OK: migrated {len(migrated_days)} day(s) -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
