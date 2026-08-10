
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
