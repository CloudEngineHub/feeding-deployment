"""Shared axis calibration for the panels stacked above the deployment table.

Every panel in that float has to put deployment day k directly over column k of
the table below it. The column centers depend on how wide the row-label column
came out, which is a LaTeX layout result no script can predict -- so measure it
off the compiled PDF and hand the resulting axes rect to matplotlib.

Needs one prior compile of the paper. Until then callers get FALLBACK_CAL and a
second run tightens it.
"""

import json
import subprocess

# Used before main.pdf exists. Close enough to compile against, not exact.
FALLBACK_CAL = {
    "textwidth_pt": 514.08,
    "axes_left": 0.1455,
    "axes_width": 0.8545,
    "bar_width": 0.68,
}

# Runs under an interpreter with PyMuPDF, kept separate from the plotting step
# because no single env here has both PyMuPDF and matplotlib.
CALIB_SRC = r'''
import json, sys
# PyMuPDF >= 1.24 renamed the module; the `fitz` compatibility shim prints its
# deprecation notice to STDOUT, which lands in front of our JSON and breaks the
# caller's parse. Import the modern name when it exists.
try:
    import pymupdf as fitz
except ImportError:
    import fitz

pdf_path, n_days = sys.argv[1], int(sys.argv[2])

for page in fitz.open(pdf_path):
    if "deployment day by day" not in page.get_text():
        continue
    # The tabular sits in a \resizebox spanning \textwidth, so its booktabs
    # rules give the text block's left and right edges on this page.
    rules = [d["rect"] for d in page.get_drawings()
             if d["rect"].width > 400 and d["rect"].height < 3]
    if not rules:
        break
    left, right = min(r.x0 for r in rules), max(r.x1 for r in rules)
    # The day-number header row is the baseline carrying the most integers in
    # 1..n_days -- survey values are 1-7 so they never outnumber it.
    baselines = {}
    for w in page.get_text("words"):
        if w[4].isdigit() and 1 <= int(w[4]) <= n_days:
            baselines.setdefault(round(w[1], 1), []).append(
                (int(w[4]), (w[0] + w[2]) / 2))
    best = max(baselines.values(), key=len, default=[])
    if len(best) < n_days:
        break
    centers = dict(best)
    step = (centers[n_days] - centers[1]) / (n_days - 1)
    span = right - left
    print(json.dumps({
        "textwidth_pt": span,
        "axes_left": (centers[1] - step / 2 - left) / span,
        "axes_width": (n_days * step) / span,
        "bar_width": 0.68,
    }))
    sys.exit(0)
sys.exit(3)
'''


def calibrate(pdf_path, n_days, python_bin, scratch):
    """Measure the deployment table's day-column geometry from the compiled paper."""
    if not pdf_path.exists():
        print("  [calib] no compiled PDF yet; using fallback geometry")
        return dict(FALLBACK_CAL)
    scratch.mkdir(parents=True, exist_ok=True)
    script = scratch / "_calibrate.py"
    script.write_text(CALIB_SRC)
    try:
        r = subprocess.run([python_bin, str(script), str(pdf_path), str(n_days)],
                           capture_output=True, text=True)
    except OSError as e:
        # No such interpreter (the PyMuPDF env lives on whichever machine last
        # ran this). Degrade to the fallback like every other failure here --
        # crashing would take the whole figure down over a calibration nicety.
        print(f"  [calib] cannot run {python_bin} ({e.strerror}); "
              f"using fallback geometry")
        return dict(FALLBACK_CAL)
    if r.returncode != 0:
        print(f"  [calib] measurement failed ({r.stderr.strip()[:80]}); "
              f"using fallback geometry")
        return dict(FALLBACK_CAL)
    # Take the last JSON object on stdout, so any library chatter ahead of it is
    # harmless rather than fatal.
    for line in reversed([l for l in r.stdout.splitlines() if l.strip()]):
        if line.lstrip().startswith("{"):
            return json.loads(line)
    print("  [calib] no geometry on stdout; using fallback geometry")
    return dict(FALLBACK_CAL)
