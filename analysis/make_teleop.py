#!/usr/bin/env python3
"""Emit figures/teleop_per_day.pdf: the CR's teleoperation per day, split into
navigation (she drove the base) and manipulation (she drove the arm).

Two contamination sources are excluded, both confirmed against the logs:

  * The researcher's Xbox controller publishes to `/cmd_vel_teleop` too
    (launch/shared_autonomy.launch:50), and is used to place the base during
    setup, teardown, and some mid-meal resets. The bag connection records carry
    the publisher callerid, so the two are separable: `/rosbridge_websocket` is
    the webapp on the CR's iPad, `/shared_autonomy_teleop` is the Xbox. Only the
    former is counted. Across the deployment the Xbox accounts for more base
    driving than the CR does, so merging them roughly doubles the count.
  * 48 of the 62 logged arm sessions contain zero `command_received` events. The
    CR used the arm-teleop screen as an escape hatch -- open it, press Done, then
    Next or Redo -- to skip past a skill without ever jogging the arm. Only
    sessions that actually commanded motion count as teleoperation here.
  * A further 8 sessions did jog the arm, but were not rescues of a stuck arm:
    the CR presses arm-control to STOP the robot so she can reach the "Move
    base" button that appears after Done (the base series already records the
    drive that follows), a researcher does the same to adjust hardware, and once
    she nudged an idle arm between bites. `failure_context` separates these
    exactly -- see COUNTED_CONTEXTS below -- leaving 6 counted manipulation
    sessions, all of them joint-limit rescues.

The base series comes from the rosbags, not the jsonl logs. No jsonl file
records base driving: `teleop_intervention_log.jsonl` holds arm sessions only,
`nav_offset_log.jsonl` sees a drive only when it ends at the nav-adjust prompt
with a measured offset (2 of 34 episodes), and the Done -> "Move base" detour out
of arm teleop writes nothing at all. The bags do record `/cmd_vel_teleop` as its
own topic, so every human base command is recoverable with exact timestamps --
see analysis/bag_teleop.py. Bag coverage of the meal windows is 92-100% per day
and is reported per day so any gap is visible rather than silently read as zero.

    python analysis/make_teleop.py [--user-dir DIR]
"""

import argparse
import collections
import datetime
import json
import pathlib
import subprocess

import bag_teleop
from figure_geometry import calibrate

# Publisher on /cmd_vel_teleop that corresponds to the CR's iPad. The other
# publisher, /shared_autonomy_teleop, is the researcher's Xbox controller.
WEBAPP_CALLERID = "/rosbridge_websocket"

# A session counts as manipulation teleoperation only if the arm was actually
# stuck. The log's own `failure_context` separates the cases exactly, and the
# other two values are not manipulation assistance:
#
#   joint_limit_failure  -- the arm hit a joint limit and could not proceed; the
#                           CR drove it out. This is the real thing. COUNTED.
#   mid_skill_takeover   -- the CR pressed arm-control to STOP the robot, either
#                           to reach the "Move base" button that appears after
#                           Done (so the intent was to drive the base, and the
#                           base series already records it) or so a researcher
#                           could adjust hardware -- e.g. all three day-4
#                           open_fridge sessions were a researcher fixing the
#                           gripper. Any arm motion here is incidental.
#   user_initiated_idle  -- she opened the screen with the robot idle between
#                           bites and nudged it; not a response to a failure.
#
# Verified against every motion-commanding session in the deployment: the six
# joint-limit sessions are exactly the six the CR and researcher confirmed as
# genuine arm rescues, and the eight others are exactly the ones they identified
# as red herrings. This also subsumes the feeding-loop exclusion -- no
# joint-limit session ever occurred inside acquire_bite or a transfer.
COUNTED_CONTEXTS = {"joint_limit_failure"}

from make_survey_table import CALENDAR

def collect_arm(user_dir):
    """Arm-teleop sessions per day, plus each meal's wall-clock window."""
    day_of_date, windows = {}, collections.defaultdict(list)
    ran, meal_window = set(), {}

    for i in range(1, len(CALENDAR) + 1):
        d = user_dir / f"day_{i:02d}"
        events = d / "events.jsonl"
        if not events.exists():
            continue
        ran.add(i)
        meta = json.loads((d / "metadata.json").read_text())
        started = datetime.datetime.fromisoformat(meta["started"]["iso"])
        day_of_date[started.date().isoformat()] = i
        epochs = []
        for line in events.open():
            if not line.strip():
                continue
            e = json.loads(line)
            epochs.append(e["epoch"])
            if e.get("category") != "skill_execute":
                continue
            windows[i].append((e["start_epoch"],
                               e["start_epoch"] + e.get("duration_s", 0),
                               e["hla"]))
        meal_window[i] = (min(epochs), max(epochs))

    # Walk the arm log by session: a session counts only if she commanded motion.
    sessions = collections.defaultdict(
        lambda: {"day": None, "commands": 0, "start": None, "ctx": None})
    for line in (user_dir / "teleop_intervention_log.jsonl").open():
        if not line.strip():
            continue
        r = json.loads(line)
        s = sessions[r["session_id"]]
        if r.get("event") == "session_start":
            s["day"] = day_of_date.get(r["t"][:10])
            s["start"] = datetime.datetime.fromisoformat(r["t"]).timestamp()
            s["ctx"] = r.get("failure_context")
        elif r.get("event") == "command_received":
            s["commands"] += 1

    counts = collections.defaultdict(collections.Counter)
    for s in sessions.values():
        if s["day"] is None:
            continue
        if not s["commands"]:
            counts[s["day"]]["no_op"] += 1
        elif s["ctx"] not in COUNTED_CONTEXTS:
            counts[s["day"]]["not_rescue"] += 1
        else:
            counts[s["day"]]["manip"] += 1

    return counts, day_of_date, ran, meal_window


def base_episodes(bag_dir, cache, meal_window, verbose=True):
    """{day: (episode count, seconds driven, bag coverage fraction)}.

    Every `/cmd_vel_teleop` message is the CR driving the base, whatever route
    she took to get there, so this captures the arm-teleop detour that leaves no
    trace in any jsonl log.
    """
    scanned = bag_teleop.scan(bag_dir, cache, callerid=WEBAPP_CALLERID,
                              verbose=verbose)
    out = {}
    for day, (lo, hi) in meal_window.items():
        stamps, spans = [], []
        for info in scanned.values():
            span = info["span"]
            if not span or span[1] < lo or span[0] > hi:
                continue
            spans.append((max(span[0], lo), min(span[1], hi)))
            stamps += [t for t in info["stamps"] if lo <= t <= hi]
        covered, edge = 0.0, None
        for a, b in sorted(spans):
            if edge is None or a > edge:
                covered += b - a
                edge = b
            elif b > edge:
                covered += b - edge
                edge = b
        eps = bag_teleop.episodes(sorted(set(stamps)))
        out[day] = (len(eps), sum(b - a for a, b in eps), covered / (hi - lo))
    return out


def collect(user_dir, bag_dir, cache, verbose=True):
    counts, _, ran, meal_window = collect_arm(user_dir)
    base = base_episodes(bag_dir, cache, meal_window, verbose)
    return [{
        "n": i,
        "ran": i in ran,
        "nav": base.get(i, (0, 0, 0))[0],
        "nav_seconds": base.get(i, (0, 0, 0))[1],
        "coverage": base.get(i, (0, 0, 0))[2],
        "manip": counts[i]["manip"],
        "no_op": counts[i]["no_op"],
        "not_rescue": counts[i]["not_rescue"],
    } for i in range(1, len(CALENDAR) + 1)]


FIGURE_SRC = r'''
import json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

days, cal = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
out = sys.argv[3]

logged = [d for d in days if d["ran"]]
xs     = [d["n"] for d in logged]
nav    = [d["nav"] for d in logged]
manip  = [d["manip"] for d in logged]
n      = len(days)

plt.rcParams.update({"font.size": 8, "font.family": "serif"})

W_IN = cal["textwidth_pt"] / 72.27
fig = plt.figure(figsize=(W_IN, 1.30))
ax = fig.add_axes([cal["axes_left"], 0.09, cal["axes_width"], 0.87])

NAVC, MANC = "#2E5E8C", "#B0A8A0"
w = cal["bar_width"]
ax.bar(xs, nav, color=NAVC, width=w, label="Navigation (base)")
ax.bar(xs, manip, bottom=nav, color=MANC, width=w, label="Manipulation (arm)")

ax.set_ylabel("Teleoperation", fontsize=7.5, labelpad=2)
ax.set_xlim(0.5, n + 0.5)
# Headroom for the legend, which sits over the tallest bars otherwise.
ax.set_ylim(0, max([a + b for a, b in zip(nav, manip)] + [1]) + 2.6)
ax.set_xticks(range(1, n + 1))
ax.set_xticklabels([])          # the table's Day/Date rows below label this axis
ax.tick_params(axis="x", length=2, width=0.5)
ax.tick_params(axis="y", labelsize=6.5, length=2, width=0.5)
ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True, nbins=4))

for d in days:
    if not d["ran"]:
        ax.axvspan(d["n"] - 0.5, d["n"] + 0.5, color="black", alpha=0.055, lw=0)

for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
for spine in ("left", "bottom"):
    ax.spines[spine].set_linewidth(0.5)
ax.grid(axis="y", color="0.9", lw=0.5)
ax.set_axisbelow(True)
ax.legend(fontsize=6.5, frameon=False, ncol=2, loc="upper right",
          handlelength=1.1, columnspacing=1.0, borderpad=0.1, handletextpad=0.4)
fig.savefig(out)
print("wrote", out)
'''


def main():
    here = pathlib.Path(__file__).resolve().parent
    repo = here.parent
    p = argparse.ArgumentParser()
    p.add_argument("--user-dir", default="/Volumes/Lexar/feeding/log/aimee",
                   type=pathlib.Path)
    p.add_argument("--python", default="/Users/rkjenamani/miniconda3/envs/feed/bin/python")
    p.add_argument("--fitz-python",
                   default="/Users/rkjenamani/miniconda3/envs/report/bin/python")
    p.add_argument("--bag-dir", type=pathlib.Path,
                   default=pathlib.Path("/Volumes/Lexar/feeding/log/system_logs"),
                   help="directory of deployment rosbags")
    p.add_argument("--scratch", type=pathlib.Path, default=here / ".build")
    args = p.parse_args()

    days = collect(args.user_dir, args.bag_dir, args.scratch / "bagcache.json")
    cal = calibrate(repo / "docs/feeding-deployment-docs/main.pdf", len(days),
                    args.fitz_python, args.scratch)
    print(f"  [calib] axes_left={cal['axes_left']:.4f} width={cal['axes_width']:.4f}")

    args.scratch.mkdir(parents=True, exist_ok=True)
    (args.scratch / "teleop.json").write_text(json.dumps(days))
    (args.scratch / "calibration.json").write_text(json.dumps(cal))
    (args.scratch / "_plot_teleop.py").write_text(FIGURE_SRC)
    out = repo / "docs/feeding-deployment-docs/figures/teleop_per_day.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([args.python, str(args.scratch / "_plot_teleop.py"),
                    str(args.scratch / "teleop.json"),
                    str(args.scratch / "calibration.json"), str(out)], check=True)

    tot = collections.Counter()
    for d in days:
        for k in ("nav", "manip", "no_op", "not_rescue", "nav_seconds"):
            tot[k] += d[k]
    print(f"  {tot['nav'] + tot['manip']} teleoperation events: "
          f"{tot['nav']} navigation (base, webapp only), "
          f"{tot['manip']} manipulation (arm, motion commanded)")
    print(f"  excluded: {tot['not_rescue']} arm sessions that jogged the arm but were "
          f"not joint-limit rescues (stop-to-move-base, researcher hardware fixes, idle nudges)")
    print(f"  excluded: {tot['no_op']} arm sessions with no motion "
          f"(escape-hatch skill skips)")
    worst = min((d["coverage"] for d in days if d["ran"]), default=1.0)
    print(f"  base driving: {tot['nav_seconds']:.0f}s total; "
          f"bag coverage of meal windows {worst*100:.0f}%-100%")
    print("  per day (nav, manip):",
          {d["n"]: (d["nav"], d["manip"]) for d in days if d["ran"]})


if __name__ == "__main__":
    main()
