#!/usr/bin/env python3
"""Emit tables/survey_timeline.tex: daily survey responses, one column per
deployment day, cells shaded on the 1-7 scale.

Reads the post-meal `survey_response` events straight out of each day's
events.jsonl, so the table is regenerated -- never hand-edited -- as the
remaining deployment days land. Question wording and item order come from
feeding_deployment.integration.survey.SURVEY_QUESTIONS.

    python analysis/make_survey_table.py [--user-dir DIR] [--out FILE]
"""

import argparse
import datetime
import json
import pathlib

# Full deployment calendar. Days 1-15 are the dates actually logged; the
# remainder is the planned schedule through the end of the study. Days with no
# events.jsonl yet render as empty "planned" cells.
CALENDAR = [
    "2026-07-17", "2026-07-21", "2026-07-22", "2026-07-23", "2026-07-24",
    "2026-07-25", "2026-07-27", "2026-07-28", "2026-07-29", "2026-07-30",
    "2026-08-02", "2026-08-03", "2026-08-04", "2026-08-05", "2026-08-06",
    "2026-08-09", "2026-08-10", "2026-08-11", "2026-08-12", "2026-08-13",
    "2026-08-14", "2026-08-17",
]

# (key, printed label). Split by valence: the NASA-TLX workload items are
# reverse-coded when shading so that green always means "good".
LOWER_IS_BETTER = [
    ("mental_demand", "Mental demand"),
    ("physical_demand", "Physical demand"),
    ("temporal_demand", "Temporal demand"),
    ("effort", "Effort"),
    ("frustration", "Frustration"),
]
HIGHER_IS_BETTER = [
    ("performance", "Performance"),
    ("trust", "Trust"),
    ("safety", "Safety"),
    ("predictability", "Predictability"),
    ("robot_adaptation", "Robot adaptation"),
    ("satisfaction", "Satisfaction"),
]

# Red -> green ramp, kept light enough that black digits stay legible in print.
#
# Deliberately not a linear 1-7 sweep. Once the workload items are reverse-coded,
# the responses only ever occupy shading levels 3-7, and level 6 alone accounts
# for ~half the cells -- a linear ramp puts two colors on empty range and renders
# most of the table one flat green. So the ramp is anchored to the range actually
# observed: yellow sits at 5 rather than 4, and 6/7 are pulled apart so the
# most common level is visibly distinct from the best one. Levels 1-2 keep red
# shades in case later meals reach them. The legend prints the mapping, so the
# shift is visible to the reader rather than implied.
RAMP = ["D65A4E", "E8846A", "F2A85F", "F8C97A", "FBE79A", "B5DC7E", "5FB85F"]


# The logged setting names carry the companion's or the TV's position, which is
# more detail than the table needs; only the social context is shown. The
# vocabulary changed mid-deployment (commit f6b80a69 collapsed the positional
# variants into bare "Personal"/"Social"/"Watching TV"), so both generations are
# mapped. Unknown values raise rather than falling through verbatim: a raw label
# reaches the table as a long rotated string in a 9pt column, which is how day
# 16 first rendered as "Social with person on Left".
SETTING_LABELS = {
    "Personal": "Personal",
    "Social": "Social",
    "Watching TV": "TV",
    "Social with person on Left": "Social",
    "Social with person on Right": "Social",
    "Watching TV with TV in Front": "TV",
    "Watching TV with TV on Left": "TV",
    "Watching TV with TV on Right": "TV",
}


# Survey responses corrected after the fact by checking back with the CR. The
# logged value stays untouched in events.jsonl -- that is the record of what was
# tapped -- and the correction is applied here, where the table is built, so the
# override is visible and reviewable rather than buried in the raw log.
#
#   (day, item): corrected value  # provenance
SURVEY_CORRECTIONS = {
    (8, "safety"): 6,         # logged 3; mis-tap the researcher noted at the time
                              # ("she wanted to put 6, but put 3 instead by mistake")
    (16, "satisfaction"): 7,  # logged 1; mis-tap, confirmed with the CR 2026-08-10
}


def read_day(day_dir):
    """({item key: 1-7 value}, setting) for one day; ({}, None) if it has not run."""
    events = day_dir / "events.jsonl"
    if not events.exists():
        return None, None
    responses, setting = {}, None
    for line in events.open():
        if not line.strip():
            continue
        e = json.loads(line)
        if e.get("category") == "survey_response" and isinstance(e.get("value"), int):
            responses[e["field"]] = e["value"]
        elif e.get("category") == "survey":
            for k, v in (e.get("responses") or {}).items():
                if isinstance(v, int):
                    responses.setdefault(k, v)
        elif e.get("category") == "preference_predicted":
            # LAST prediction wins, not the first. A meal can be re-entered
            # before it starts -- day 16 was logged once as "Social with person
            # on Left" and then corrected to "Social" three minutes later -- and
            # the governing context is the one the meal actually ran under, which
            # is also what lands in full_history_memory.
            raw = (e.get("context") or {}).get("setting")
            if raw:
                if raw not in SETTING_LABELS:
                    raise KeyError(
                        f"{day_dir.name}: unmapped setting {raw!r}; add it to "
                        "SETTING_LABELS so it does not reach the table verbatim"
                    )
                setting = SETTING_LABELS[raw]
    return (responses or None), setting


def build(user_dir):
    days = []
    for i, date in enumerate(CALENDAR, start=1):
        responses, setting = read_day(user_dir / f"day_{i:02d}")
        if responses:
            for (day, item), corrected_value in SURVEY_CORRECTIONS.items():
                if day == i:
                    if item not in responses:
                        raise KeyError(
                            f"day {i}: SURVEY_CORRECTIONS names {item!r}, which "
                            "this day has no logged response for -- the override "
                            "would invent a value rather than correct one"
                        )
                    print(f"  [correction] day {i} {item}: "
                          f"{responses[item]} -> {corrected_value}")
                    responses[item] = corrected_value
        days.append({
            "n": i,
            "date": datetime.date.fromisoformat(date),
            "responses": responses,
            "setting": setting,
        })
    return days


def cell(value, invert):
    if value is None:
        return r"\nodata"
    level = (8 - value) if invert else value
    return rf"\sv{{{level}}}{{{value}}}"


def render(days):
    n_logged = sum(1 for d in days if d["responses"])
    n_planned = len(days) - n_logged
    colspec = "l" + "c" * len(days)

    L = []
    for i, hexcode in enumerate(RAMP, start=1):
        L.append(rf"\definecolor{{sv{i}}}{{HTML}}{{{hexcode}}}")
    # Every day column is forced to one fixed width. Left to natural sizing the
    # two-digit headers (10-22) make their columns ~2pt wider than the
    # single-digit ones, which would break the column-to-bar alignment with the
    # corrections panel above. The length is absolute, not em-based, so it does
    # not drift with the font size of the row it appears in.
    L.append(r"\newlength{\svcw}\setlength{\svcw}{9pt}")
    L.append(r"\newcommand{\sv}[2]{\cellcolor{sv#1}\makebox[\svcw]{#2}}")
    L.append(r"\newcommand{\nodata}{\cellcolor{black!7}\makebox[\svcw]{}}")
    L.append(r"\newcommand{\svhead}[1]{\makebox[\svcw]{\rotatebox{90}{\scriptsize #1}}}")
    L.append(r"\newcommand{\svday}[1]{\makebox[\svcw]{\scriptsize #1}}")
    L.append("")
    L.append(r"\begin{table*}[t]")
    # The paper's other table* floats end on \vspace{-0.6cm}, which eats into the
    # separation when two of them share a page; restore it on our side.
    L.append(r"\vspace{0.35cm}")
    # Kept to what a reader needs to interpret the panels. The counting rules,
    # exclusions and provenance live in the module docstrings of the three
    # generators and belong in an appendix, not in the caption.
    L.append(r"\caption{The deployment day by day, on a shared horizontal axis. "
             r"\textbf{Top:} preference corrections per meal, split into "
             r"pre-meal overrides on the preferences page, mid-meal changes from "
             r"the settings page, and plate-handle colours re-picked on a "
             r"detection page; the individual corrections are "
             r"enumerated in \cref{sec:appendix_corrections}, and the bundle "
             r"the system finally held each meal is given dimension by "
             r"dimension in \cref{tab:preference_matrix}. "
             r"\textbf{Middle:} the CR's teleoperation, split into episodes "
             r"driving the base and sessions driving the arm. "
             r"\textbf{Bottom:} daily survey responses, all 7-point "
             r"(1 = very low, 7 = very high), shaded so that \textbf{green is "
             r"always the favorable end}: the five NASA-TLX workload items "
             r"($\downarrow$) are reverse-shaded, the remaining six "
             r"($\uparrow$) shaded directly; the free-text item is omitted. "
             + (r"All " + str(n_logged) + r" meals of the deployment are shown.}"
                if not n_planned else
                rf"Days 1--{n_logged} are logged; the final {n_planned} "
                + ("column is a scheduled meal" if n_planned == 1
                   else "columns are scheduled meals")
                + r" not yet run.}"))
    L.append(r"\label{tab:deployment_timeline}")
    L.append(r"\vspace{-0.2cm}")
    L.append(r"{\centering")
    # In the same float as the tabular, not a separate figure*: LaTeX keeps
    # figure and table floats in independent queues, so adjacency -- the whole
    # point of a shared axis -- could not otherwise be guaranteed.
    L.append(r"\includegraphics[width=\textwidth]{figures/corrections_per_day.pdf}\\[1pt]")
    L.append(r"\includegraphics[width=\textwidth]{figures/teleop_per_day.pdf}\\[1pt]")
    L.append(r"\setlength\extrarowheight{0.4mm}")
    L.append(rf"\resizebox{{\textwidth}}{{!}}{{%")
    L.append(rf"\begin{{tabular}}{{{colspec}}}")
    L.append(r"\toprule")

    L.append(r"\textsc{Day} & " +
             " & ".join(rf"\svday{{{d['n']}}}" for d in days) + r" \\")
    L.append(r"\textsc{Date} & " +
             " & ".join(rf"\svhead{{{d['date']:%b}~{d['date'].day}}}" for d in days) +
             r" \\")
    # Placed under Date rather than between Day and Date: Day and Date are the
    # same fact twice, and sitting directly above the data block is where the
    # setting is easiest to read against the values. The gap matters -- rotated
    # "Personal" is tall enough to butt against the date above it and read as one
    # string ("Personal Jul 22") without it.
    L.append(r"\addlinespace[3pt]")
    L.append(r"\textsc{Setting} & " +
             " & ".join(rf"\svhead{{{d['setting']}}}" if d["setting"] else r"\svhead{}"
                        for d in days) + r" \\")
    L.append(r"\midrule")

    for title, items, invert in [
        (r"\emph{Workload} ($\downarrow$ lower is better)", LOWER_IS_BETTER, True),
        (r"\emph{Experience} ($\uparrow$ higher is better)", HIGHER_IS_BETTER, False),
    ]:
        # Each block carries its own key, because a printed digit maps to
        # opposite ends of the ramp in the two blocks. Both strips run red to
        # green; only the numbering reverses.
        order = range(7, 0, -1) if invert else range(1, 8)
        # Emitted inline rather than through a macro argument: the "&" separators
        # must be read by the nested tabular's own alignment, not the outer one.
        key = " & ".join(rf"\sv{{{(8 - v) if invert else v}}}{{{v}}}" for v in order)
        L.append(rf"\multicolumn{{10}}{{l}}{{{title}}} & "
                 rf"\multicolumn{{13}}{{r}}{{\scriptsize worse~"
                 rf"\setlength{{\tabcolsep}}{{2pt}}"
                 rf"\begin{{tabular}}{{|*{{7}}{{c|}}}}\hline {key} \\\hline\end{{tabular}}"
                 rf"~better}} \\")
        for key, label in items:
            cells = [cell((d["responses"] or {}).get(key), invert) for d in days]
            L.append(rf"\quad {label} & " + " & ".join(cells) + r" \\")
        if invert:
            L.append(r"\midrule")

    L.append(r"\bottomrule")
    L.append(r"\end{tabular}}")
    L.append(r"}")
    L.append(r"\vspace{-0.4cm}")
    L.append(r"\end{table*}")
    return "\n".join(L) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--user-dir", default="/Volumes/Lexar/feeding/log/aimee",
                   type=pathlib.Path)
    p.add_argument("--out", type=pathlib.Path,
                   default=pathlib.Path(__file__).resolve().parent.parent /
                   "docs/feeding-deployment-docs/tables/survey_timeline.tex")
    args = p.parse_args()

    days = build(args.user_dir)
    args.out.write_text(render(days))
    logged = [d["n"] for d in days if d["responses"]]
    print(f"wrote {args.out}")
    print(f"  {len(logged)} days with survey data (day {logged[0]}-{logged[-1]}), "
          f"{len(days) - len(logged)} planned")


if __name__ == "__main__":
    main()
