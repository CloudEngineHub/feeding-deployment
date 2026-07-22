"""Live progress monitor for an evaluate_prediction_model.py run.

The evaluator checkpoints each finished day to
    <run_dir>/logs/<user>/day_metrics/day_<NNNN>.json
with {day, affective_state, context, steps, per_dim_m0}. This script polls those
checkpoints every --interval seconds, rebuilds a partial user_report using the
SAME reduction the evaluator applies at the end (a faithful copy of
apply_day_result), and hands it to the real metrics._generate_metrics -- so the
live images are genuine plot_a/plot_b/plot_c/plot_d/plot_e (plus
summary_metrics.json), identical in style to the final report, just computed on
however many days have finished so far.

Output goes to <run_dir>/live/ by default so it never collides with the
evaluator's own final plots (written to <run_dir>/ at the very end). The watcher
exits once the run writes its report.json (i.e. the eval finished).

Usage:
    PYTHONPATH=src python3 -m feeding_deployment.preference_learning.methods.watch_progress \
        [--run-dir <reports/run_*>]   # default: newest reports/run_* (by name)
        [--interval 60]               # poll seconds (default 60)
        [--out-dir <dir>]             # default: <run_dir>/live
        [--once]                      # render a single frame and exit
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from feeding_deployment.preference_learning.methods.metrics import _generate_metrics

REPORTS_DIR = Path(__file__).parent / "reports"


def _newest_run_dir() -> Optional[Path]:
    # Sort by the timestamp in the dir NAME (run_YYYY_MM_DD__HH_MM_SS sorts
    # chronologically as text). mtime is unreliable -- editing files in an old
    # run dir (e.g. cleanup) bumps its mtime and would mask a newer run.
    runs = sorted((p for p in REPORTS_DIR.glob("run_*") if p.is_dir()), reverse=True)
    return runs[0] if runs else None


def _load_checkpoints(run_dir: Path) -> List[Dict[str, Any]]:
    """Every day checkpoint under any user, sorted by day. Skips partial (.tmp)
    and unreadable files so a mid-write frame never crashes the watcher."""
    recs: List[Dict[str, Any]] = []
    for ckpt in run_dir.glob("logs/*/day_metrics/day_*.json"):
        try:
            recs.append(json.loads(ckpt.read_text()))
        except (json.JSONDecodeError, OSError):
            continue
    recs.sort(key=lambda r: int(r.get("day", 0)))
    return recs


def _build_user_report(recs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Reduce day checkpoints into the user_report shape _generate_metrics reads.
    Mirrors evaluate_prediction_model.apply_day_result + the per_day / by_state /
    per_dim assembly so the live plots match the eventual final report exactly."""
    acc_m0_sum = defaultdict(float); acc_m0_n = defaultdict(int)
    acc_m1_sum = defaultdict(float); acc_m1_n = defaultdict(int)
    mism_m0_sum = defaultdict(float); mism_m0_n = defaultdict(int)
    mism_m1_sum = defaultdict(float); mism_m1_n = defaultdict(int)
    mstar_sum = defaultdict(float); mstar_n = defaultdict(int)
    acc_m0_state_sum = defaultdict(float); acc_m0_state_n = defaultdict(int)
    mstar_state_sum = defaultdict(float); mstar_state_n = defaultdict(int)
    per_dim_total = defaultdict(int); per_dim_correct = defaultdict(int)

    for r in recs:
        day = int(r.get("day", 0))
        state = str(r.get("affective_state", "unknown"))
        steps = r.get("steps") or []
        if not steps:
            continue
        for st in steps:
            m_i = int(st["m"]); acc_i = float(st["acc"]); nm_i = float(st["mismatches"])
            if m_i == 0:
                acc_m0_sum[day] += acc_i; acc_m0_n[day] += 1
                mism_m0_sum[day] += nm_i; mism_m0_n[day] += 1
                acc_m0_state_sum[state] += acc_i; acc_m0_state_n[state] += 1
            elif m_i == 1:
                acc_m1_sum[day] += acc_i; acc_m1_n[day] += 1
                mism_m1_sum[day] += nm_i; mism_m1_n[day] += 1
        for f, ok in (r.get("per_dim_m0") or {}).items():
            per_dim_total[f] += 1
            if ok:
                per_dim_correct[f] += 1
        last = steps[-1]
        m_star = int(last["m"])
        mstar_sum[day] += m_star; mstar_n[day] += 1
        mstar_state_sum[state] += m_star; mstar_state_n[state] += 1
        # Zero-correction meal: perfect at m=0 -> counts as a clean m=1 too, so
        # Plot C has an m=1 point on those days (matches the evaluator).
        if m_star == 0 and float(last["mismatches"]) == 0:
            acc_m1_sum[day] += 1.0; acc_m1_n[day] += 1
            mism_m1_sum[day] += 0.0; mism_m1_n[day] += 1

    all_days = sorted(
        set(acc_m0_n) | set(acc_m1_n) | set(mism_m0_n) | set(mism_m1_n) | set(mstar_n)
    )
    per_day_metrics: List[Dict[str, Any]] = []
    for d in all_days:
        rec: Dict[str, Any] = {"day": d}
        if acc_m0_n[d]:  rec["acc_m0"] = acc_m0_sum[d] / acc_m0_n[d]
        if acc_m1_n[d]:  rec["acc_m1"] = acc_m1_sum[d] / acc_m1_n[d]
        if mism_m0_n[d]: rec["mismatches_m0"] = mism_m0_sum[d] / mism_m0_n[d]
        if mism_m1_n[d]: rec["mismatches_m1"] = mism_m1_sum[d] / mism_m1_n[d]
        if mstar_n[d]:   rec["m_star"] = mstar_sum[d] / mstar_n[d]
        per_day_metrics.append(rec)

    by_affective_state: Dict[str, Dict[str, float]] = {}
    for state in sorted(set(acc_m0_state_n) | set(mstar_state_n)):
        by_affective_state[state] = {}
        if acc_m0_state_n[state]:
            by_affective_state[state]["acc_m0"] = acc_m0_state_sum[state] / acc_m0_state_n[state]
        if mstar_state_n[state]:
            by_affective_state[state]["mean_m_star"] = mstar_state_sum[state] / mstar_state_n[state]

    per_dimension_m0_accuracy = {
        f: per_dim_correct[f] / float(per_dim_total[f])
        for f in per_dim_total if per_dim_total[f]
    }

    return {
        "per_day_metrics": per_day_metrics,
        "by_affective_state": by_affective_state,
        "per_dimension_m0_accuracy": per_dimension_m0_accuracy,
    }


def render(run_dir: Path, out_dir: Path) -> int:
    recs = _load_checkpoints(run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if recs:
        _generate_metrics([_build_user_report(recs)], out_dir)
    return len(recs)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", default=None, help="reports/run_* dir (default: newest by name).")
    ap.add_argument("--interval", type=float, default=60.0, help="Poll seconds (default 60).")
    ap.add_argument("--out-dir", default=None, help="Output dir (default: <run_dir>/live).")
    ap.add_argument("--once", action="store_true", help="Render one frame and exit.")
    args = ap.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else _newest_run_dir()
    if not run_dir or not run_dir.is_dir():
        print("error: no run directory found", flush=True)
        return 2
    out_dir = Path(args.out_dir) if args.out_dir else (run_dir / "live")
    print(f"watching {run_dir}\n -> {out_dir}/plot_[a-e]_*.png", flush=True)

    while True:
        finished = (run_dir / "report.json").exists()
        n = render(run_dir, out_dir)
        print(f"[{datetime.now():%H:%M:%S}] {n} day(s) done"
              f"{'  — run FINISHED, final live frame written' if finished else ''}", flush=True)
        if args.once or finished:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
