"""Sanity checks for regenerated 27-dim deployment datasets (Phase 0.4).

For every dataset file (paired with its encoding file by basename) verify:
  1. every day record has exactly the 27 PREFERENCE_BUNDLE fields;
  2. color/nav choices are canonical dicts;
  3. hard rules hold in the ground truth (inside-mouth transfer =>
     outside_mouth_distance == "not applicable"; no dippables/sauces =>
     bite_dipping_preference == "do not dip");
  4. the continuous dims match continuous_truth(encoding tables, context)
     exactly -- i.e. they are stable across days for the same context.

Usage:
    PYTHONPATH=src python -m feeding_deployment.preference_learning.data_generation.sanity_check_dataset \
        --data-dir src/feeding_deployment/preference_learning/data/deployment_datasets \
        --encodings-dir src/feeding_deployment/preference_learning/data/user_encodings
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Any, Dict, List

from feeding_deployment.preference_learning.config.mealtime_context import MEAL_CONTENTS_BY_LABEL
from feeding_deployment.preference_learning.config.preference_bundle import PREFERENCE_BUNDLE
from feeding_deployment.preference_learning.data_generation.continuous_prefs import (
    CONTINUOUS_FIELDS,
    continuous_truth,
)

ALL_FIELDS = sorted(dim.field for dim in PREFERENCE_BUNDLE)
KIND_BY_FIELD = {dim.field: dim.kind for dim in PREFERENCE_BUNDLE}


def check_dataset(data_path: str, encoding_path: str) -> List[str]:
    """Returns a list of human-readable problems (empty = clean)."""
    problems: List[str] = []

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    with open(encoding_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    tables = payload.get("continuous_tables")
    if not isinstance(tables, dict) or not tables:
        return [f"{encoding_path}: missing continuous_tables (old-format encoding)"]

    for day_rec in data.get("days", []):
        day = day_rec.get("day")
        tag = f"{os.path.basename(data_path)} day {day}"
        prefs = day_rec.get("preferences", {}) or {}
        context = day_rec.get("context", {}) or {}

        got = sorted(prefs.keys())
        if got != ALL_FIELDS:
            missing = sorted(set(ALL_FIELDS) - set(got))
            extra = sorted(set(got) - set(ALL_FIELDS))
            problems.append(f"{tag}: field mismatch (missing={missing}, extra={extra})")
            continue

        choices: Dict[str, Any] = {f: (prefs[f].get("choice") if isinstance(prefs[f], dict) else None) for f in ALL_FIELDS}

        # 2. value shapes
        for f in ALL_FIELDS:
            kind = KIND_BY_FIELD[f]
            v = choices[f]
            if kind in ("color", "nav_offset"):
                if not isinstance(v, dict):
                    problems.append(f"{tag}: {f} should be a dict, got {type(v).__name__}: {v!r}")
            elif not isinstance(v, str) or not v.strip():
                problems.append(f"{tag}: {f} should be a non-empty string, got {v!r}")

        # 3. hard rules
        if choices.get("transfer_mode") == "inside mouth transfer" and choices.get("outside_mouth_distance") != "not applicable":
            problems.append(f"{tag}: hard rule violated (inside mouth but distance={choices.get('outside_mouth_distance')!r})")
        meal = MEAL_CONTENTS_BY_LABEL.get(str(context.get("meal", "")))
        if meal is not None:
            if ((not meal.dippable_items) or (not meal.sauces)) and choices.get("bite_dipping_preference") != "do not dip":
                problems.append(f"{tag}: hard rule violated (no dippables/sauces but dipping={choices.get('bite_dipping_preference')!r})")

        # 4. continuous dims replay exactly from the tables
        try:
            expected = continuous_truth(
                tables, str(context.get("time_of_day")), str(context.get("transient_affective_state"))
            )
        except KeyError as e:
            problems.append(f"{tag}: context not in tables: {e}")
            continue
        for f in CONTINUOUS_FIELDS:
            if choices.get(f) != expected[f]:
                problems.append(f"{tag}: {f} = {choices.get(f)!r}, expected {expected[f]!r}")

    return problems


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Sanity-check regenerated 27-dim deployment datasets.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--encodings-dir", required=True)
    args = parser.parse_args(argv)

    data_files = sorted(glob.glob(os.path.join(args.data_dir, "*.json")))
    if not data_files:
        print(f"No dataset files in {args.data_dir}", file=sys.stderr)
        return 1

    total_problems = 0
    for data_path in data_files:
        encoding_path = os.path.join(args.encodings_dir, os.path.basename(data_path))
        if not os.path.isfile(encoding_path):
            print(f"✗ {os.path.basename(data_path)}: no matching encoding file in {args.encodings_dir}")
            total_problems += 1
            continue
        problems = check_dataset(data_path, encoding_path)
        if problems:
            total_problems += len(problems)
            print(f"✗ {os.path.basename(data_path)}: {len(problems)} problem(s)")
            for p in problems:
                print(f"    - {p}")
        else:
            print(f"✓ {os.path.basename(data_path)}: clean")

    if total_problems:
        print(f"\nFAILED: {total_problems} problem(s).")
        return 1
    print("\nAll datasets clean.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
