#!/usr/bin/env python3
"""Emit the per-day correction ledger (appendix table) and the corrections-per-day
figure.

A "correction" is a preference dimension the CR overrode. Two channels carry
them and both are reported:

  ask-flow  -- a dim listed in `preference_asked.corrected`, i.e. she rejected
               the prediction on the pre-meal preferences page.
  settings  -- a `preference_settings_edit` with changed=true, i.e. she changed
               a dim mid-meal from the settings page.

Predicted values are reconstructed by replaying the day's events in epoch order:
start from `preference_predicted.predicted_bundle`, apply each
`preference_repredicted.changed`, and read the dim just before the override. A
corrected dim never appears in its own reprediction's `changed` map (that map
holds the propagated dims), so the value read at the ask is the pre-correction
prediction.

    python analysis/make_corrections.py [--user-dir DIR]
"""

import argparse
import collections
import json
import pathlib
import subprocess

from figure_geometry import calibrate
from make_survey_table import CALENDAR  # single source of truth for the calendar

# Rendered names for the preference dimensions, so the appendix reads as English
# rather than as log keys.
DIM_LABELS = {
    "robot_speed": "Robot speed",
    "microwave_time": "Microwave time",
    "skewering_axis": "Skewering axis",
    "confirm_feeding_pickup": "Confirm feeding pickup",
    "confirm_navigation_arrival": "Confirm navigation arrival",
    "confirm_manipulation": "Confirm manipulation",
    "transfer_mode": "Transfer mode",
    "outside_mouth_distance": "Outside-mouth distance",
    "convey_robot_ready_for_initiating_transfer": "Convey robot ready (initiate)",
    "convey_robot_ready_for_completing_transfer": "Convey robot ready (complete)",
    "detect_user_ready_for_initiating_transfer_feeding": "Detect user ready (feeding)",
    "detect_user_ready_for_initiating_transfer_drinking": "Detect user ready (drinking)",
    "detect_user_ready_for_initiating_transfer_wiping": "Detect user ready (wiping)",
    "detect_user_completed_transfer_feeding": "Detect transfer done (feeding)",
    "detect_user_completed_transfer_drinking": "Detect transfer done (drinking)",
    "detect_user_completed_transfer_wiping": "Detect transfer done (wiping)",
    "retract_between_bites": "Retract between bites",
    "bite_dipping_preference": "Bite dipping",
    "bite_ordering": "Bite ordering",
    "wait_before_autocontinue_bite_selection": "Autocontinue wait (bite sel.)",
    "wait_before_autocontinue_task_selection": "Autocontinue wait (task sel.)",
}


def jl(path):
    if not path.exists():
        return []
    return [json.loads(l) for l in path.open() if l.strip()]


# A few dimensions (bite_ordering above all) accept free text, and one day-12
# entry runs to two sentences -- long enough to stretch its row over a third of
# the column. Clip for the ledger; the full strings are in the logs.
VALUE_CLIP = 88


def tex_escape(s, clip=None):
    """Escape for LaTeX. Clips before escaping so the appended ellipsis, which
    is already LaTeX, does not get escaped along with the payload."""
    s = str(s)
    truncated = clip is not None and len(s) > clip
    if truncated:
        s = s[:clip].rstrip()
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                 ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"),
                 ("}", r"\}"), ("~", r"\textasciitilde{}"),
                 ("^", r"\textasciicircum{}")]:
        s = s.replace(a, b)
    return s + r"\,\ldots" if truncated else s


def collect(user_dir):
    """Per-day list of corrections, each {channel, dim, predicted, corrected}."""
    days = []
    for i, date in enumerate(CALENDAR, start=1):
        d = user_dir / f"day_{i:02d}"
        events = jl(d / "events.jsonl")
        if not events:
            days.append({"n": i, "date": date, "context": None, "corrections": None})
            continue

        events.sort(key=lambda e: e["epoch"])
        bundle, context, corrections = {}, None, []

        for e in events:
            cat = e["category"]
            if cat == "preference_predicted" and not bundle:
                bundle = dict(e.get("predicted_bundle") or {})
                context = e.get("context")
            elif cat == "preference_repredicted":
                for dim, mv in (e.get("changed") or {}).items():
                    bundle[dim] = mv.get("to")
            elif cat == "preference_asked":
                truth = e.get("ground_truth") or {}
                for dim in e.get("corrected") or []:
                    corrections.append({
                        "channel": "ask",
                        "dim": dim,
                        "predicted": bundle.get(dim),
                        "corrected": truth.get(dim),
                    })
                    bundle[dim] = truth.get(dim)
            elif cat == "preference_settings_edit" and e.get("changed"):
                dim = e.get("field")
                corrections.append({
                    "channel": "settings",
                    "dim": dim,
                    "predicted": bundle.get(dim),
                    "corrected": e.get("value"),
                })
                bundle[dim] = e.get("value")
            elif cat == "preference_color_recorded" and e.get("changed"):
                # The plate-handle colour is re-observed at each pickup, and the
                # CR can correct it on the detection page. She supplies it by
                # picking off the image rather than from a list of options, so it
                # is a different kind of input from the other two channels and is
                # drawn as its own series rather than folded into them.
                dim = e.get("field")
                corrections.append({
                    "channel": "colour",
                    "dim": dim,
                    "predicted": bundle.get(dim),
                    "corrected": e.get("offset") or e.get("color") or e.get("value"),
                })

        days.append({"n": i, "date": date, "context": context,
                     "corrections": corrections})
    return days


# --------------------------------------------------------------------------- #
# Appendix table
# --------------------------------------------------------------------------- #

def render_appendix(days):
    # Spans both columns: at single-column width nearly every row wrapped to two
    # lines and the ledger ran off the bottom of the page.
    L = [r"\begin{table*}[t]",
         r"\caption{Every preference correction the CR made, by day. \emph{Ask} "
          r"corrections are overrides on the pre-meal preferences page; "
          r"\emph{Set} corrections are mid-meal changes from the settings "
          r"page; \emph{Col} corrections are plate-handle colours she "
          r"re-picked on a detection page. \emph{Predicted} is the value the "
          r"system held for that dimension immediately before the override.}",
         r"\label{tab:correction_ledger}",
         r"\vspace{-0.2cm}",
         r"{\centering\footnotesize",
         r"\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{@{}ll p{0.25\textwidth} p{0.31\textwidth} p{0.31\textwidth}@{}}",
         r"\toprule",
         r"\textsc{Day} & & \textsc{Dimension} & \textsc{Predicted} & "
         r"\textsc{Corrected to} \\",
         r"\midrule"]

    for d in days:
        if d["corrections"] is None:
            continue  # meal not run yet; the caption says which days those are
        if not d["corrections"]:
            # Ran, drew zero corrections. Stated explicitly so it cannot be
            # confused with a day that simply has no data.
            L.append(rf"{d['n']} & \multicolumn{{4}}{{l}}{{\emph{{no corrections}}}} \\")
            L.append(r"\addlinespace[2pt]")
            continue
        for k, c in enumerate(d["corrections"]):
            day_cell = rf"\multirow{{{len(d['corrections'])}}}{{*}}{{{d['n']}}}" if k == 0 else ""
            L.append(" & ".join([
                day_cell,
                {"ask": "Ask", "settings": "Set", "colour": "Col"}[c["channel"]],
                tex_escape(DIM_LABELS.get(c["dim"], c["dim"])),
                tex_escape(c["predicted"] if c["predicted"] is not None else "--", VALUE_CLIP),
                tex_escape(c["corrected"] if c["corrected"] is not None else "--", VALUE_CLIP),
            ]) + r" \\")
        L.append(r"\addlinespace[2pt]")

    L += [r"\bottomrule", r"\end{tabular}", r"}", r"\vspace{-0.3cm}",
          r"\end{table*}"]
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------- #
# Corrections-per-day figure
# --------------------------------------------------------------------------- #

FIGURE_SRC = r'''
import json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

days, cal = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
out = sys.argv[3]

logged = [d for d in days if d["corrections"] is not None]
xs   = [d["n"] for d in logged]
ask  = [sum(1 for c in d["corrections"] if c["channel"] == "ask") for d in logged]
sett = [sum(1 for c in d["corrections"] if c["channel"] == "settings") for d in logged]
col  = [sum(1 for c in d["corrections"] if c["channel"] == "colour") for d in logged]
n    = len(days)

plt.rcParams.update({"font.size": 8, "font.family": "serif"})

# Authored at exactly \textwidth so \includegraphics[width=\textwidth] applies no
# scaling; the axes rect is then placed so bar centers land on the table's day
# column centers directly below. See calibrate() for where these come from.
W_IN = cal["textwidth_pt"] / 72.27
# Short bottom margin: this axis carries no tick labels of its own (the table's
# Day/Date rows serve as them), so leaving room for them would just open a gap
# between the two panels and weaken the visual pairing.
fig = plt.figure(figsize=(W_IN, 1.30))
ax = fig.add_axes([cal["axes_left"], 0.09, cal["axes_width"], 0.87])

ax.bar(xs, ask, color="#C0504D", width=cal["bar_width"], label="Pre-meal (ask flow)")
ax.bar(xs, sett, bottom=ask, color="#E8A33D", width=cal["bar_width"],
       label="Mid-meal (settings)")
ax.bar(xs, col, bottom=[a + s for a, s in zip(ask, sett)], color="#4E79A7",
       width=cal["bar_width"], label="Plate colour")

ax.set_ylabel("Corrections", fontsize=7.5, labelpad=2)
ax.set_xlim(0.5, n + 0.5)
ax.set_ylim(0, max([a + s + c for a, s, c in zip(ask, sett, col)] + [1]) + 0.5)
ax.set_xticks(range(1, n + 1))
ax.set_xticklabels([])          # the table's Day/Date rows below label this axis
ax.tick_params(axis="x", length=2, width=0.5)
ax.tick_params(axis="y", labelsize=6.5, length=2, width=0.5)
ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True, nbins=4))

# Days that have not run yet: an empty column here must not read as "zero
# corrections", because days 10 and 11 genuinely drew none.
for d in days:
    if d["corrections"] is None:
        ax.axvspan(d["n"] - 0.5, d["n"] + 0.5, color="black", alpha=0.055, lw=0)
first_pending = next((d["n"] for d in days if d["corrections"] is None), None)
if first_pending is not None:
    ax.text((first_pending + n) / 2, ax.get_ylim()[1] * 0.5, "not yet run",
            ha="center", va="center", fontsize=6, color="0.5", style="italic")

for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
for spine in ("left", "bottom"):
    ax.spines[spine].set_linewidth(0.5)
ax.grid(axis="y", color="0.9", lw=0.5)
ax.set_axisbelow(True)
ax.legend(fontsize=6.5, frameon=False, ncol=3, loc="upper right",
          handlelength=1.1, columnspacing=1.0, borderpad=0.1, handletextpad=0.4)
fig.savefig(out)
print("wrote", out)
'''


def build_figure(days, cal, out_pdf, python_bin, scratch):
    scratch.mkdir(parents=True, exist_ok=True)
    data = scratch / "corrections.json"
    data.write_text(json.dumps(days))
    calib = scratch / "calibration.json"
    calib.write_text(json.dumps(cal))
    script = scratch / "_plot_corrections.py"
    script.write_text(FIGURE_SRC)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([python_bin, str(script), str(data), str(calib), str(out_pdf)],
                   check=True)


def main():
    here = pathlib.Path(__file__).resolve().parent
    repo = here.parent
    p = argparse.ArgumentParser()
    p.add_argument("--user-dir", default="/Volumes/Lexar/feeding/log/aimee",
                   type=pathlib.Path)
    p.add_argument("--python", default="/Users/rkjenamani/miniconda3/envs/feed/bin/python",
                   help="interpreter with matplotlib")
    p.add_argument("--fitz-python",
                   default="/Users/rkjenamani/miniconda3/envs/report/bin/python",
                   help="interpreter with PyMuPDF, for measuring the table geometry")
    p.add_argument("--scratch", type=pathlib.Path, default=here / ".build")
    args = p.parse_args()

    days = collect(args.user_dir)

    tex = repo / "docs/feeding-deployment-docs/tables/correction_ledger.tex"
    tex.write_text(render_appendix(days))
    print(f"wrote {tex}")

    cal = calibrate(repo / "docs/feeding-deployment-docs/main.pdf", len(days),
                    args.fitz_python, args.scratch)
    print(f"  [calib] axes_left={cal['axes_left']:.4f} "
          f"width={cal['axes_width']:.4f} textwidth={cal['textwidth_pt']:.1f}pt")
    pdf = repo / "docs/feeding-deployment-docs/figures/corrections_per_day.pdf"
    build_figure(days, cal, pdf, args.python, args.scratch)

    per_day = collections.Counter()
    chan = collections.Counter()
    for d in days:
        for c in d["corrections"] or []:
            per_day[d["n"]] += 1
            chan[c["channel"]] += 1
    total = sum(chan.values())
    print(f"  {total} corrections "
          f"({chan['ask']} ask-flow, {chan['settings']} settings, "
          f"{chan['colour']} plate colour) "
          f"over {sum(1 for d in days if d['corrections'] is not None)} meals")
    print("  per day:", dict(sorted(per_day.items())))


if __name__ == "__main__":
    main()
