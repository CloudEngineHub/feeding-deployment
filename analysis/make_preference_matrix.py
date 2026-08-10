#!/usr/bin/env python3
"""Emit tables/preference_matrix.tex from the deployment memory files.

Columns are days, rows are preference dimensions; the header carries each day's
setting and plate. Values are abbreviated against a legend so 16 columns fit
across a \\textwidth table*; cells the CR corrected that day are shaded + bold.
"""
import json, os, collections

MEM = ("/home/isacc/deployment_ws/src/feeding-deployment/src/feeding_deployment/"
       "integration/log/aimee/preference_learning/aimee/full_history_memory")
OUT = ("/home/isacc/deployment_ws/src/feeding-deployment/docs/feeding-deployment-docs/"
       "tables/preference_matrix.tex")
NDAYS = 16

days = [json.load(open(os.path.join(MEM, f"day_{d:04d}.json"))) for d in range(1, NDAYS + 1)]

def setting(d):
    s = d["context"]["setting"]
    return "Social" if "Social" in s else ("TV" if "Watching TV" in s else "Personal")

# Plate contents, matching the labels used in Fig. 2 (figures/meal_grid.tex).
PLATE = {
    1: "chicken nuggets", 2: "orange chicken", 3: "teriyaki chicken",
    4: "chicken kebab", 5: "strawberry + w.cream", 6: "hash brown, sausage",
    7: "chicken nuggets + ketchup", 8: "orange chicken", 9: "teriyaki chicken",
    10: "hash brown", 11: "hash brown, sausage", 12: "hash brown + ranch",
    13: "orange chicken", 14: "chicken kebab", 15: "pancake, sausage + syrup",
    16: "steak, mozzarella sticks",
}

# value -> (short code, legend gloss). Codes are unique across the whole table.
ABB = {
    "fast": ("F", "fast"),
    "medium": ("M", "medium"),
    "slow": ("S", "slow"),
    "no microwave": ("off", "no microwave"),
    "30 secs": ("30s", "30 secs"),
    "1 min": ("1m", "1 min"),
    "2 min": ("2m", "2 min"),
    "skip": ("sk", "skip"),
    "countdown (15 sec)": ("c15", "countdown 15\\,s"),
    "countdown (30 sec)": ("c30", "countdown 30\\,s"),
    "countdown (60 sec)": ("c60", "countdown 60\\,s"),
    "wait for me": ("W", "wait for me"),
    "perpendicular to major axis": ("$\\perp$", "perpendicular to major axis"),
    "along major axis": ("$\\parallel$", "along major axis"),
    "do not dip": ("--", "do not dip"),
    "more": ("+", "dip more"),
    "less": ("$-$", "dip less"),
    "outside mouth transfer": ("out", "outside-mouth transfer"),
    "inside mouth transfer": ("in", "inside-mouth transfer"),
    "near": ("nr", "near"),
    "far": ("fr", "far"),
    "not applicable": ("n/a", "not applicable"),
    "yes": ("Y", "yes"),
    "no": ("N", "no"),
    "LED": ("led", "LED"),
    "speech": ("sp", "speech"),
    "both": ("b", "both"),
    "open mouth": ("OM", "open mouth"),
    "button": ("BT", "button"),
    "proceed automatically after a pause": ("AP", "proceed automatically after a pause"),
    "perception": ("PC", "perception"),
}

SECTIONS = [
    ("Pace and confirmation", [
        ("robot_speed", "Robot speed"),
        ("microwave_time", "Microwave time"),
        ("confirm_navigation_arrival", "Confirm navigation arrival"),
        ("confirm_manipulation", "Confirm manipulation"),
        ("confirm_feeding_pickup", "Confirm feeding pickup"),
    ]),
    ("Bite handling", [
        ("skewering_axis", "Skewering axis"),
        ("bite_dipping_preference", "Dipping amount"),
        ("transfer_mode", "Transfer mode"),
        ("outside_mouth_distance", "Outside-mouth distance"),
        ("retract_between_bites", "Retract between bites"),
    ]),
    ("Signalling: robot to CR", [
        ("convey_robot_ready_for_initiating_transfer", "Convey robot ready (initiate)"),
        ("convey_robot_ready_for_completing_transfer", "Convey robot ready (complete)"),
    ]),
    ("Signalling: CR to robot", [
        ("detect_user_ready_for_initiating_transfer_feeding", "Detect user ready (feeding)"),
        ("detect_user_ready_for_initiating_transfer_drinking", "Detect user ready (drinking)"),
        ("detect_user_ready_for_initiating_transfer_wiping", "Detect user ready (wiping)"),
        ("detect_user_completed_transfer_feeding", "Detect transfer done (feeding)"),
        ("detect_user_completed_transfer_drinking", "Detect transfer done (drinking)"),
        ("detect_user_completed_transfer_wiping", "Detect transfer done (wiping)"),
    ]),
    ("Autocontinue", [
        ("wait_before_autocontinue_bite_selection", "Autocontinue wait (bite sel.)"),
        ("wait_before_autocontinue_task_selection", "Autocontinue wait (task sel.)"),
    ]),
]

# A code that means two different things would silently mislead the reader.
_bycode = collections.defaultdict(set)
for _v, (_c, _g) in ABB.items():
    _bycode[_c].add(_g)
_dupe = {c: sorted(g) for c, g in _bycode.items() if len(g) > 1}
assert not _dupe, f"abbreviation collision: {_dupe}"

used = {}          # code -> gloss, for the legend
def cell(d, k):
    v = d["ground_truth_bundle"].get(k)
    if v is None:
        return "\\pmna"
    code, gloss = ABB[str(v)]
    used[code] = gloss
    return ("\\pmk{%s}" % code) if k in (d.get("corrected") or {}) else ("\\pmc{%s}" % code)

def row(k, label):
    return ("\\quad %s & " % label) + " & ".join(cell(d, k) for d in days) + " \\\\"

ncol = NDAYS
body = []
for title, rows in SECTIONS:
    body.append("\\addlinespace[2pt]")
    body.append("\\multicolumn{%d}{@{}l}{\\emph{%s}} \\\\[1pt]" % (ncol + 1, title))
    for k, lab in rows:
        body.append(row(k, lab))

daynums = " & ".join("\\pmday{%d}" % d["day"] for d in days)
setrow  = " & ".join("\\pmhead{%s}" % setting(d) for d in days)
foodrow = " & ".join("\\pmhead{%s}" % PLATE[d["day"]] for d in days)

# Legend, ordered by the sections so related codes sit together.
seen, legend_items = set(), []
for _t, rows in SECTIONS:
    for k, _l in rows:
        for d in days:
            v = d["ground_truth_bundle"].get(k)
            if v is None: continue
            code, gloss = ABB[str(v)]
            if code + gloss not in seen:
                seen.add(code + gloss)
                legend_items.append("\\texttt{%s}~=~%s" % (code, gloss))
legend = ";\\; ".join(legend_items)

corr_total = sum(1 for d in days for _t, rs in SECTIONS for k, _l in rs
                 if k in (d.get("corrected") or {}))

# The deployment-wide count reported in the body table (ask-flow overrides +
# settings-page edits, read from events.jsonl) uses a different unit: it counts
# every override event, so a dimension corrected twice in one meal counts twice,
# and it includes bite_ordering, which has no row in this matrix. Compute it here
# so the caption can reconcile the two numbers instead of appearing to contradict.
def _paper_correction_count():
    log_root = os.path.dirname(os.path.dirname(os.path.dirname(MEM)))
    n = 0
    for d in range(1, NDAYS + 1):
        p = os.path.join(log_root, f"day_{d:02d}", "events.jsonl")
        if not os.path.exists(p):
            continue
        for line in open(p):
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            if e.get("category") == "preference_asked":
                n += len(e.get("corrected") or [])
            elif e.get("category") == "preference_settings_edit" and e.get("changed"):
                n += 1
    return n

paper_total = _paper_correction_count()

TEMPLATE = r"""% Generated by analysis/make_preference_matrix.py from the per-day finalized
% preference bundles (log/<CR>/preference_learning/<CR>/full_history_memory/
% day_NNNN.json). Re-run it after each meal rather than editing this file.
%
% Columns are days; the two rotated header rows give the setting and the plate so
% each column is self-describing. Values are abbreviated against the legend in
% the caption -- the alternative, spelling them out, does not fit 16 columns
% across a table*. Shaded bold cells are dimensions the CR corrected that day;
% everything else she accepted as predicted.

\newlength{\pmcw}\setlength{\pmcw}{11pt}
\newcommand{\pmc}[1]{\makebox[\pmcw]{\tiny #1}}
\newcommand{\pmk}[1]{\cellcolor{black!12}\makebox[\pmcw]{\tiny\textbf{#1}}}
\newcommand{\pmna}{\cellcolor{black!4}\makebox[\pmcw]{\tiny\textcolor{black!45}{--}}}
\newcommand{\pmhead}[1]{\makebox[\pmcw]{\rotatebox{90}{\tiny #1}}}
\newcommand{\pmday}[1]{\makebox[\pmcw]{\scriptsize #1}}

\begin{table*}[t]
\caption{\textbf{The preference bundle the system held at the end of every
meal.} One column per day, headed by that meal's setting and the plate as
served; one row per preference dimension. \textbf{Shaded bold} cells are
dimensions the CR corrected during that meal; every other cell is a value she
accepted as predicted. The continuous dimensions (plate-handle colour at each
pickup location, and the per-location parking offsets) and the free-text
bite-ordering dimension are omitted. Days absent are meals not yet run.
\textsc{Key:} @@LEGEND@@.}
\label{tab:preference_matrix}
\vspace{-0.2cm}
{\centering
\setlength\extrarowheight{0.3mm}
\setlength{\tabcolsep}{1.5pt}
\resizebox{\textwidth}{!}{%
\begin{tabular}{@{}l@@COLSPEC@@@{}}
\toprule
\textsc{Day} & @@DAYS@@ \\
\textsc{Setting} & @@SETTINGS@@ \\
\textsc{Plate} & @@PLATES@@ \\
\midrule
@@BODY@@
\bottomrule
\end{tabular}}
}
\vspace{-0.3cm}
\end{table*}
"""

TEX = (TEMPLATE
       .replace("@@NCORR@@", str(corr_total))
       .replace("@@NPAPER@@", str(paper_total))
       .replace("@@LEGEND@@", legend)
       .replace("@@COLSPEC@@", "c" * ncol)
       .replace("@@DAYS@@", daynums)
       .replace("@@SETTINGS@@", setrow)
       .replace("@@PLATES@@", foodrow)
       .replace("@@BODY@@", "\n".join(body)))

open(OUT, "w").write(TEX)
print("wrote", OUT)
print("corrected cells in matrix:", corr_total)
print("legend entries:", len(legend_items))
print("distinct codes:", sorted(used))
