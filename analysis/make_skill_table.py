#!/usr/bin/env python3
"""Emit tables/skill_success.tex: success rates for the perception-dependent skills.

Inclusion rule: a skill is listed if its implementation resolves its target pose
from sensing at run time, rather than replaying a fixed pose from the scene
description. Concretely, the skills below each call one of the
`perception_interface.perceive_*` routines (open-set detection with
GroundingDINO/SAM, or MolmoPoint for the microwave button, back-projected
through RealSense depth), or consume head-perception or FLAIR food detection.

Excluded by the same rule -- they move through recorded joint targets and so
cannot fail on perception: `place_plate_on_holder`, `pick_plate_from_holder`,
`pick_utensil`/`stow_utensil`, `pick_drink`/`stow_drink`, `pick_wipe`/`stow_wipe`,
and the `navigate_*` skills (which localize against the SLAM map, a separate
stack from manipulation perception).

    python analysis/make_skill_table.py [--user-dir DIR]
"""

import argparse
import collections
import json
import pathlib

# (hla, printed name, what it must perceive), grouped by station.
# Takeovers on these skills are not rescues. The CR uses the takeover control on
# the feeding loop to *skip* a bite or transfer she does not want -- the skill was
# working and she chose to move on. Counting them as takeovers would charge the
# perception stack for what is really a bite-level choice, so they are excluded
# from n the same way the SystemExit aborts are.
SKIP_TAKEOVERS = {"acquire_bite", "transfer_utensil", "transfer_drink",
                  "transfer_wipe"}

GROUPS = [
    ("Fridge", [
        ("open_fridge", "Open fridge", "door handle"),
        ("pick_plate_from_fridge", "Pick plate from fridge", "plate attachment"),
        ("close_fridge", "Close fridge", "door handle"),
    ]),
    ("Microwave", [
        ("open_microwave", "Open microwave", "door handle"),
        ("place_plate_in_microwave", "Place plate in microwave", "cavity pose"),
        ("press_microwave_button", "Press start button", "button keypoint"),
        ("close_microwave", "Close microwave", "door handle"),
        ("pick_plate_from_microwave", "Pick plate from microwave", "plate attachment"),
    ]),
    ("Table", [
        ("gaze_at_table", "Gaze at table", "placement surface"),
        ("place_plate_on_table", "Place plate on table", "placement surface"),
        ("pick_plate_from_dining_table", "Pick plate from dining table", "plate attachment"),
        ("pick_plate_from_movable_table", "Pick plate from movable table", "plate attachment"),
        ("pick_plate_from_table", "Pick plate from table", "plate attachment"),
    ]),
    ("Sink", [
        ("place_plate_in_sink", "Place plate in sink", "basin pose"),
    ]),
    ("Feeding", [
        ("acquire_bite", "Acquire bite", "food item"),
        ("transfer_utensil", "Transfer bite", "mouth pose"),
        ("transfer_drink", "Transfer drink", "mouth pose"),
        ("transfer_wipe", "Transfer wipe", "mouth pose"),
    ]),
]


def collect(user_dir):
    """{hla: Counter(outcome)} plus the SystemExit-aborted tally."""
    stats = collections.defaultdict(collections.Counter)
    aborted_reasons = collections.Counter()
    for day in sorted(user_dir.glob("day_*")):
        events = day / "events.jsonl"
        if not events.exists():
            continue
        for line in events.open():
            if not line.strip():
                continue
            e = json.loads(line)
            if e.get("category") != "skill_execute":
                continue
            stats[e["hla"]][e["outcome"]] += 1
            if e["outcome"] == "aborted":
                aborted_reasons[(e.get("failure_reason") or "").split(":")[0]] += 1
    return stats, aborted_reasons


def render(stats):
    rows, totals = [], collections.Counter()
    for group, skills in GROUPS:
        rows.append(("group", group))
        for hla, label, needs in skills:
            c = stats.get(hla)
            if not c:
                continue
            # `aborted` is always a SystemExit -- the operator terminating the
            # session -- so it is reported separately and kept out of the
            # denominator rather than counted against the skill. Feeding-loop
            # takeovers are skips, not rescues, and drop out the same way.
            takeover = 0 if hla in SKIP_TAKEOVERS else c["takeover"]
            n = c["success"] + takeover + c["failed"]
            rows.append(("skill", (label, needs, c, takeover, n)))
            totals["success"] += c["success"]
            totals["takeover"] += takeover
            totals["failed"] += c["failed"]
            totals["aborted"] += c["aborted"]
            totals["skipped"] += c["takeover"] - takeover
        totals["n"] = totals["success"] + totals["takeover"] + totals["failed"]

    # Spans both columns: the caption carries the inclusion/exclusion rules and is
# far too tall for a single column -- as `table` the float overflowed by ~44pt
# and LaTeX dropped it silently, taking \label{tab:skill_success} with it and
# leaving a "??" in Sec. Measures.
    L = [r"\begin{table*}[t]",
         r"\caption{Success rates of the perception-dependent skills, pooled over "
          r"all logged meals. A skill is listed if it resolves its target pose "
          r"from sensing at run time. \emph{TO} counts takeovers, where the CR "
          r"or a researcher teleoperated the robot through the step; \emph{F} "
          r"counts unrecovered failures.}",
         r"\label{tab:skill_success}",
         r"\vspace{-0.2cm}",
         r"{\centering\footnotesize",
         r"\setlength{\tabcolsep}{3.5pt}",
         r"\begin{tabular}{@{}llrrrr@{}}",
         r"\toprule",
         r"\textsc{Skill} & \textsc{Perceives} & $n$ & \textsc{TO} & "
         r"\textsc{F} & \textsc{Succ.} \\",
         r"\midrule"]

    for kind, payload in rows:
        if kind == "group":
            L.append(rf"\multicolumn{{6}}{{@{{}}l}}{{\emph{{{payload}}}}} \\")
            continue
        label, needs, c, takeover, n = payload
        rate = c["success"] / n * 100 if n else float("nan")
        cell = rf"{rate:.1f}\%"
        if n and rate < 90:
            cell = rf"\textbf{{{cell}}}"
        L.append(rf"\quad {label} & {needs} & {n} & {takeover} & "
                 rf"{c['failed']} & {cell} \\")

    overall = totals["success"] / totals["n"] * 100
    L += [r"\midrule",
          rf"\textbf{{All perception-dependent}} & & \textbf{{{totals['n']}}} & "
          rf"\textbf{{{totals['takeover']}}} & \textbf{{{totals['failed']}}} & "
          rf"\textbf{{{overall:.1f}\%}} \\",
          r"\bottomrule", r"\end{tabular}", r"}", r"\vspace{-0.3cm}",
          r"\end{table*}"]
    return "\n".join(L) + "\n", totals


def main():
    repo = pathlib.Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser()
    p.add_argument("--user-dir", default="/Volumes/Lexar/feeding/log/aimee",
                   type=pathlib.Path)
    p.add_argument("--out", type=pathlib.Path,
                   default=repo / "docs/feeding-deployment-docs/tables/skill_success.tex")
    args = p.parse_args()

    stats, aborted = collect(args.user_dir)
    tex, totals = render(stats)
    args.out.write_text(tex)
    print(f"wrote {args.out}")
    print(f"  perception-dependent: n={totals['n']}, "
          f"{totals['success']} success, {totals['takeover']} takeover, "
          f"{totals['failed']} failed "
          f"({totals['success'] / totals['n'] * 100:.1f}%)")
    print(f"  excluded aborted (session terminations): {totals['aborted']}")
    print(f"  abort reason prefixes: {dict(aborted)}")


if __name__ == "__main__":
    main()
