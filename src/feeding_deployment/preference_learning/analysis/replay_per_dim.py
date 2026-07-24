#!/usr/bin/env python3
"""
Faithful offline per-dim replay of a real deployment, compared to live joint.

The deployment ran the JOINT predictor (one LLM call predicts the whole bundle).
Its per-day initial prediction is already in each day's ``events.jsonl`` (the
``preference_predicted`` / ``stage:"start"`` record). PER-DIM (one isolated LLM
call per open dimension, each seeing only that dimension's own history) was never
run live, so its numbers must be COMPUTED -- but from the exact same inputs the
deployment logged, so the two are directly comparable.

This is NOT the synthetic evaluator (``methods/evaluate_prediction_model.py``).
That harness needs a ``--profile-label`` from a fixed 5-way set, re-runs the
method from scratch, and reveals corrections in a random (seeded) order -- none
of which matches a real deployment. Here we reuse the deployment's OWN
``PredictionModel`` construction (``integration/run.py:693`` /
``integration/emulate_preference_pipeline.py:312``), flipping only
``prediction_mode="joint" -> "per_dim"``:

  * physical profile   : aimee's real free-text capability description, recovered
                         verbatim from a logged prediction prompt (the deployment
                         passed it via ``physical_profile_description=``, NOT one
                         of the labels). Override with --physical_profile_file.
  * memory_mode        : single_full_history (what the deployment ran). The live
                         ``full_history_memory/day_*.json`` already carry each
                         day's finalized ``ground_truth_bundle``; per-dim slices
                         them per field. We load them AS-IS and never write back,
                         so day t's memory is days 1..t-1's GROUND TRUTH (never
                         per-dim's own guesses -- errors must not compound).
  * ground truth       : each day's ``ground_truth_bundle`` (the value aimee
                         converged to), the target both methods are scored on.

For day t: load prior memory (days 1..t-1) -> per-dim predicts each open dim at
m=0 -> score the finalized bundle against day t's ground truth. Mismatches =
the corrections per-dim would need at m=0 (per-dim has no cross-dim propagation,
so one correction per wrong dim). Joint's m=0 bundle is read straight from the
logs and scored the same way. We compare each method's INITIAL prediction against
the same fixed ground truth -- simulating a full interactive per-dim correction
session is counterfactual (aimee reacted to joint, not to per-dim), so it is out
of scope.

Scoring covers the 21 CATEGORICAL + TEXT dimensions (color / nav-offset dims are
not scored). By default those 8 unscored dims are pinned so per-dim spends no LLM
calls on them: per-dim predicts each dim in isolation, so pinning a dim cannot
change any other dim's prediction -- the 21 scored predictions are byte-identical
whether or not color/nav are pinned, at 8 fewer calls/day. Pass
--predict-all-dims for a fully literal replay that also predicts (and discards)
color/nav.

Hard-rule dims (``bite_dipping_preference`` forced to "do not dip" on a known meal
with no dip; ``outside_mouth_distance`` forced to "not applicable" under
inside-mouth transfer) are deterministic, not learned -- they are flagged per day
and reported both included and excluded from accuracy.

Runs against an ISOLATED copy of the memory (the real aimee logs are never
touched) and makes real Anthropic calls (~21 per open day). Use --dry-run to
validate wiring and see joint's baseline with zero API cost.

Usage:
    # dry run: parse logs, score joint, recover profile, count planned calls
    python -m feeding_deployment.preference_learning.analysis.replay_per_dim --dry-run

    # full per-dim replay of aimee (real Anthropic calls)
    python -m feeding_deployment.preference_learning.analysis.replay_per_dim

    # a different deployment log / subset of days
    python -m feeding_deployment.preference_learning.analysis.replay_per_dim \
        --log-dir /path/to/log/<user> --user <user> --days 3
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from feeding_deployment.preference_learning.config.preference_bundle import (
    COLOR_FIELDS,
    NAV_OFFSET_FIELDS,
)
from feeding_deployment.preference_learning.methods.prediction_model import (
    PredictionModel,
    _get_meal_info,
)
from feeding_deployment.preference_learning.methods.utils import (
    PREF_FIELDS,
    _pred_matches_truth,
)

# The 21 scored dims: everything except color / nav-offset (20 categorical + the
# one text dim, bite_ordering), in bundle order.
_UNSCORED = set(COLOR_FIELDS) | set(NAV_OFFSET_FIELDS)
SCORED_FIELDS: List[str] = [f for f in PREF_FIELDS if f not in _UNSCORED]

# Anchors bracketing the free-text physical profile in every prediction prompt
# (see methods/prompts/{bundle,dim}_prediction.txt). Used to recover the exact
# profile string the deployment ran, so the replay needs no manual profile input.
_PROFILE_ANCHOR_START = "Consider a care recipient with the following physical functioning abilities:"
_PROFILE_ANCHOR_END = "You are a personalized preference predictor"

# Default deployment log to replay.
_DEFAULT_LOG_DIR = (
    Path(__file__).resolve().parents[2]
    / "integration" / "log" / "aimee"
)
_DEFAULT_USER = "aimee"


# --------------------------------------------------------------------------- #
# Log parsing (read-only against the real deployment logs)
# --------------------------------------------------------------------------- #
def _pref_root(log_dir: Path, user: str) -> Path:
    return log_dir / "preference_learning" / user


def recover_physical_profile(log_dir: Path, user: str) -> str:
    """Recover the deployment's exact free-text physical profile from the
    earliest logged prediction prompt (bracketed by the two fixed anchors)."""
    calls_dir = _pref_root(log_dir, user) / "prediction_model_llm_calls"
    txts = sorted(calls_dir.glob("*.txt"))
    if not txts:
        raise FileNotFoundError(
            f"No logged prediction prompts under {calls_dir}; cannot recover the "
            f"physical profile. Pass --physical_profile_file instead."
        )
    for path in txts:
        text = path.read_text(encoding="utf-8")
        i = text.find(_PROFILE_ANCHOR_START)
        j = text.find(_PROFILE_ANCHOR_END)
        if i >= 0 and j > i:
            profile = text[i + len(_PROFILE_ANCHOR_START):j].strip()
            if profile:
                return profile
    raise ValueError(
        f"Could not locate the physical-profile block in any prompt under "
        f"{calls_dir}. Pass --physical_profile_file instead."
    )


def discover_days(log_dir: Path, user: str) -> List[int]:
    """Finalized day numbers, from the full-history memory day files."""
    fh_dir = _pref_root(log_dir, user) / "full_history_memory"
    days = []
    for p in sorted(fh_dir.glob("day_*.json")):
        try:
            days.append(int(p.stem.split("_", 1)[1]))
        except (ValueError, IndexError):
            continue
    return sorted(days)


def load_day_memory(log_dir: Path, user: str, day: int) -> Dict[str, Any]:
    """The finalized full-history record for a day: context + ground_truth_bundle."""
    p = _pref_root(log_dir, user) / "full_history_memory" / f"day_{day:04d}.json"
    return json.loads(p.read_text(encoding="utf-8"))


def load_joint_baseline(log_dir: Path, day: int) -> Optional[Dict[str, Any]]:
    """Joint's m=0 initial bundle from the day's events.jsonl
    (category=preference_predicted, stage=start). None if the log is absent."""
    events = log_dir / f"day_{day:02d}" / "events.jsonl"
    if not events.exists():
        return None
    for line in events.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("category") == "preference_predicted" and rec.get("stage") == "start":
            return rec.get("predicted_bundle") or {}
    return None


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def hard_rule_forced(bundle: Dict[str, Any], meal: str) -> List[str]:
    """Which SCORED dims in ``bundle`` were set by a deterministic hard rule
    (not learned) -- mirrors prediction_model._apply_hard_rules with no active
    corrections. These are freebie matches (ground truth gets the same rule)."""
    forced: List[str] = []
    info = _get_meal_info(meal)
    if (
        info.get("known_meal", False)
        and ((not info["has_dippable"]) or (not info["has_sauce"]))
        and str(bundle.get("bite_dipping_preference")) == "do not dip"
    ):
        forced.append("bite_dipping_preference")
    if (
        str(bundle.get("transfer_mode")) == "inside mouth transfer"
        and str(bundle.get("outside_mouth_distance")) == "not applicable"
    ):
        forced.append("outside_mouth_distance")
    return forced


def score_bundle(
    bundle: Dict[str, Any], truth: Dict[str, Any], forced: List[str]
) -> Dict[str, Any]:
    """Match ``bundle`` against ``truth`` over the scored dims present in truth.
    Returns per-dim matches, mismatch list (= corrections needed), and accuracy
    both over all scored dims and over predicted-only (excluding hard-rule dims)."""
    fields = [f for f in SCORED_FIELDS if f in truth]
    per_dim = {f: _pred_matches_truth(f, bundle.get(f), truth.get(f)) for f in fields}
    mismatches = [f for f in fields if not per_dim[f]]
    predicted = [f for f in fields if f not in forced]
    n_pred_correct = sum(1 for f in predicted if per_dim[f])
    return {
        "scored_dims": len(fields),
        "matches": len(fields) - len(mismatches),
        "mismatches": mismatches,
        "num_mismatches": len(mismatches),
        "accuracy": (len(fields) - len(mismatches)) / len(fields) if fields else 1.0,
        "accuracy_predicted_only": (n_pred_correct / len(predicted)) if predicted else 1.0,
        "per_dim": per_dim,
    }


# --------------------------------------------------------------------------- #
# Per-dim prediction (isolated model + copied memory, real logs untouched)
# --------------------------------------------------------------------------- #
def _seed_isolated_logs(log_dir: Path, user: str, out_logs: Path) -> None:
    """Copy the real cross-day memory (full-history + working-memory markers)
    into a fresh logs dir so PredictionModel.load_prior_memory finds it, without
    the replay writing anything back to the deployment logs."""
    src = _pref_root(log_dir, user)
    dst = out_logs / user
    for sub in ("full_history_memory", "working_memory"):
        s = src / sub
        if s.exists():
            shutil.copytree(s, dst / sub, dirs_exist_ok=True)


def predict_per_dim_day(
    profile: str,
    out_logs: Path,
    user: str,
    day: int,
    context: Dict[str, Any],
    truth: Dict[str, Any],
    predict_all_dims: bool,
) -> Dict[str, Any]:
    """One day's per-dim initial (m=0) prediction, reusing the deployment's
    PredictionModel with prediction_mode flipped to per_dim. A fresh model per
    day (load_prior_memory reads days 1..day-1) mirrors the deployment's
    one-process-per-day lifecycle."""
    model = PredictionModel(
        user=user,
        physical_profile_label="deployment_physical_profile",
        logs_dir=out_logs,
        physical_profile_description=profile,
        memory_mode="single_full_history",
        prediction_mode="per_dim",
    )
    model.load_prior_memory(day)

    # Unless a literal replay is requested, pin color/nav (as confirmed) so
    # per-dim spends no LLM calls on the 8 unscored dims. Per-dim isolation means
    # this does not change any scored dim's prediction. Seed pins from truth when
    # present (canonicalized downstream); missing keys pin to defaults.
    confirmed: Dict[str, Any] = {}
    if not predict_all_dims:
        confirmed = {f: truth.get(f) for f in (list(COLOR_FIELDS) + list(NAV_OFFSET_FIELDS))}

    bundle = model.predict_bundle(context=dict(context), corrected={}, confirmed=confirmed)
    return bundle


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _fmt_pct(x: float) -> str:
    return f"{100.0 * x:5.1f}%"


def print_day_report(day: int, meal: str, res: Dict[str, Any]) -> None:
    truth = res["truth"]
    pd_score = res["per_dim_score"]
    jt_score = res.get("joint_score")
    forced = res["forced"]
    print(f"\n{'=' * 78}")
    print(f"DAY {day}  |  meal: {meal}")
    print(f"{'=' * 78}")
    header = f"  {'dimension':<48} {'ground truth':<22} PD  JT"
    print(header)
    print(f"  {'-' * 74}")
    for f in SCORED_FIELDS:
        if f not in truth:
            continue
        pd_ok = "OK " if pd_score["per_dim"].get(f) else "XX "
        if jt_score is not None:
            jt_ok = "OK" if jt_score["per_dim"].get(f) else "XX"
        else:
            jt_ok = "--"
        tag = " (hard-rule)" if f in forced else ""
        gt = str(truth.get(f))[:20]
        print(f"  {f:<48} {gt:<22} {pd_ok} {jt_ok}{tag}")
    print(f"  {'-' * 74}")
    pd_m = pd_score["num_mismatches"]
    print(
        f"  PER-DIM : {pd_score['matches']}/{pd_score['scored_dims']} correct "
        f"| {pd_m} correction(s) needed | acc {_fmt_pct(pd_score['accuracy'])} "
        f"(predicted-only {_fmt_pct(pd_score['accuracy_predicted_only'])})"
    )
    if jt_score is not None:
        print(
            f"  JOINT   : {jt_score['matches']}/{jt_score['scored_dims']} correct "
            f"| {jt_score['num_mismatches']} correction(s) needed "
            f"| acc {_fmt_pct(jt_score['accuracy'])} (live baseline)"
        )
    else:
        print("  JOINT   : (no events.jsonl baseline for this day)")
    if forced:
        print(f"  hard-rule-forced dims (freebie matches): {', '.join(forced)}")


def write_outputs(out_dir: Path, report: Dict[str, Any]) -> None:
    (out_dir / "replay_per_dim.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    # Flat per-day comparison CSV.
    csv_path = out_dir / "comparison.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(
            ["day", "meal", "scored_dims",
             "per_dim_correct", "per_dim_corrections", "per_dim_acc",
             "per_dim_acc_predicted_only",
             "joint_correct", "joint_corrections", "joint_acc",
             "hard_rule_forced"]
        )
        for d in report["days"]:
            pd_s = d["per_dim_score"]
            jt_s = d.get("joint_score")
            w.writerow([
                d["day"], d["meal"], pd_s["scored_dims"],
                pd_s["matches"], pd_s["num_mismatches"],
                f"{pd_s['accuracy']:.4f}", f"{pd_s['accuracy_predicted_only']:.4f}",
                (jt_s["matches"] if jt_s else ""),
                (jt_s["num_mismatches"] if jt_s else ""),
                (f"{jt_s['accuracy']:.4f}" if jt_s else ""),
                ";".join(d["forced"]),
            ])
    print(f"\nWrote:\n  {out_dir / 'replay_per_dim.json'}\n  {csv_path}")


def plot_learning_curve(report: Dict[str, Any], out_dir: Path) -> Optional[Path]:
    """Two-panel learning curve across days (per-dim vs live joint): accuracy on
    the left, corrections-needed at m=0 on the right. Matches analyze_day.py's
    Agg/matplotlib style. Skipped (with a note) if matplotlib is absent."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot (pip install matplotlib).")
        return None

    rows = [d for d in report["days"] if d.get("per_dim_bundle")]
    if not rows:
        print("No per-dim results to plot.")
        return None
    rows.sort(key=lambda d: d["day"])
    days = [d["day"] for d in rows]

    pd_acc = [d["per_dim_score"]["accuracy"] for d in rows]
    pd_acc_pred = [d["per_dim_score"]["accuracy_predicted_only"] for d in rows]
    pd_corr = [d["per_dim_score"]["num_mismatches"] for d in rows]

    def _joint(days_rows, key_path):
        xs, ys = [], []
        for d in days_rows:
            js = d.get("joint_score")
            if js is None:
                continue
            v = js
            for k in key_path:
                v = v[k]
            xs.append(d["day"])
            ys.append(v)
        return xs, ys

    jt_acc_x, jt_acc_y = _joint(rows, ["accuracy"])
    jt_corr_x, jt_corr_y = _joint(rows, ["num_mismatches"])

    n_scored = len(report.get("scored_fields", [])) or 21
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Panel A: accuracy
    ax1.plot(days, pd_acc, "b-o", markersize=6, label="per-dim")
    ax1.plot(days, pd_acc_pred, "b--o", markersize=4, alpha=0.5, label="per-dim (predicted-only)")
    if jt_acc_x:
        ax1.plot(jt_acc_x, jt_acc_y, "r-s", markersize=6, label="joint (live)")
    ax1.set_xlabel("Day")
    ax1.set_ylabel(f"Accuracy (m=0, {n_scored} scored dims)")
    ax1.set_title("Prediction accuracy vs day")
    ax1.set_ylim(0.0, 1.0)
    ax1.set_xticks(days)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Panel B: corrections needed at m=0
    ax2.plot(days, pd_corr, "b-o", markersize=6, label="per-dim")
    if jt_corr_x:
        ax2.plot(jt_corr_x, jt_corr_y, "r-s", markersize=6, label="joint (live)")
    ax2.set_xlabel("Day")
    ax2.set_ylabel("Corrections needed at m=0")
    ax2.set_title("Correction burden vs day")
    ax2.set_ylim(0, max(pd_corr + jt_corr_y + [1]) + 1)
    ax2.set_xticks(days)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f"Per-dim vs Joint offline replay — user={report.get('user', '?')}")
    fig.tight_layout()
    out_path = out_dir / "learning_curve.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  {out_path}")
    return out_path


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Faithful offline per-dim replay of a deployment vs live joint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--log-dir", type=Path, default=_DEFAULT_LOG_DIR,
                    help="Deployment log dir (contains day_NN/ and preference_learning/).")
    ap.add_argument("--user", type=str, default=_DEFAULT_USER,
                    help="User subdir under preference_learning/.")
    ap.add_argument("--days", type=int, default=0,
                    help="Replay only the first N finalized days (0 = all).")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output dir (default: <log-dir>/preference_learning/<user>/per_dim_replay).")
    ap.add_argument("--physical_profile_file", type=str, default="",
                    help="Override the auto-recovered profile with this UTF-8 text file.")
    ap.add_argument("--predict-all-dims", action="store_true",
                    help="Also predict (and discard) color/nav dims -- fully literal replay, +8 calls/day.")
    ap.add_argument("--resume", action="store_true",
                    help="Reuse per-dim predictions cached in an existing replay_per_dim.json "
                         "(re-scores + re-plots for free); predict only the missing days. "
                         "With every day cached, makes zero API calls (no key needed).")
    ap.add_argument("--no-plot", action="store_true",
                    help="Skip the learning-curve PNG.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Parse logs, score joint, recover profile, count planned calls -- no API calls.")
    args = ap.parse_args()

    log_dir: Path = args.log_dir
    user: str = args.user
    if not _pref_root(log_dir, user).exists():
        print(f"ERROR: {_pref_root(log_dir, user)} not found.", file=sys.stderr)
        return 1

    days = discover_days(log_dir, user)
    if not days:
        print(f"ERROR: no finalized days under {_pref_root(log_dir, user)}.", file=sys.stderr)
        return 1
    if args.days > 0:
        days = days[: args.days]

    out_dir: Path = args.out or (_pref_root(log_dir, user) / "per_dim_replay")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_logs = out_dir / "replay_logs"

    # Profile (recovered verbatim, or overridden).
    if args.physical_profile_file.strip():
        profile = Path(args.physical_profile_file.strip()).read_text(encoding="utf-8").strip()
        profile_source = f"file:{args.physical_profile_file.strip()}"
    else:
        profile = recover_physical_profile(log_dir, user)
        profile_source = "recovered from logged prompt"

    # Resume cache: reuse per-dim bundles from a prior replay_per_dim.json so
    # already-computed days cost no API calls. Only non-empty bundles count.
    cache: Dict[int, Dict[str, Any]] = {}
    if args.resume:
        prior_path = out_dir / "replay_per_dim.json"
        if prior_path.exists():
            prior = json.loads(prior_path.read_text(encoding="utf-8"))
            for d in prior.get("days", []):
                pdb = d.get("per_dim_bundle")
                if pdb:
                    cache[int(d["day"])] = pdb
            print(f"Resume         : {len(cache)} cached day(s) from {prior_path.name}: "
                  f"{sorted(cache)}")
        else:
            print(f"Resume         : no prior {prior_path.name}; computing all days.")

    predict_days = [] if args.dry_run else [d for d in days if d not in cache]
    scored_per_day = len(SCORED_FIELDS)
    calls_per_day = scored_per_day + len(_UNSCORED) if args.predict_all_dims else scored_per_day
    planned_calls = calls_per_day * len(predict_days)

    print(f"Deployment log : {log_dir}")
    print(f"User           : {user}")
    print(f"Days           : {days}")
    print(f"Scored dims    : {scored_per_day} (categorical + text; color/nav not scored)")
    print(f"Physical profile ({profile_source}):\n  {profile}")
    print(f"Planned per-dim LLM calls: {planned_calls} "
          f"({len(predict_days)} day(s) x {calls_per_day} "
          f"{'all open dims' if args.predict_all_dims else 'scored dims'})")

    if predict_days and not os.environ.get("ANTHROPIC_API_KEY", "").strip():
        print("\nERROR: ANTHROPIC_API_KEY is not set (per-dim makes real Anthropic calls).",
              file=sys.stderr)
        return 1

    if predict_days:
        _seed_isolated_logs(log_dir, user, out_logs)

    day_reports: List[Dict[str, Any]] = []
    for day in days:
        mem = load_day_memory(log_dir, user, day)
        context = mem.get("context", {}) or {}
        truth = mem.get("ground_truth_bundle", {}) or {}
        meal = str(context.get("meal", ""))

        joint_bundle = load_joint_baseline(log_dir, day)
        joint_forced = hard_rule_forced(joint_bundle, meal) if joint_bundle else []
        joint_score = score_bundle(joint_bundle, truth, joint_forced) if joint_bundle else None

        if args.dry_run:
            rec = {
                "day": day, "meal": meal, "truth": truth, "forced": [],
                "per_dim_bundle": None,
                "per_dim_score": {  # placeholder so the report shape is stable
                    "scored_dims": len([f for f in SCORED_FIELDS if f in truth]),
                    "matches": 0, "mismatches": [], "num_mismatches": 0,
                    "accuracy": 0.0, "accuracy_predicted_only": 0.0, "per_dim": {},
                },
                "joint_score": joint_score,
            }
            day_reports.append(rec)
            if joint_score is not None:
                print(f"  [dry-run] day {day} joint baseline: "
                      f"{joint_score['matches']}/{joint_score['scored_dims']} correct, "
                      f"{joint_score['num_mismatches']} corrections")
            continue

        if day in cache:
            bundle = cache[day]
            print(f"\n[day {day}] reusing cached per-dim prediction ({meal!r}) -- no API calls.",
                  flush=True)
        else:
            print(f"\n[day {day}] per-dim predicting {meal!r} ({calls_per_day} calls) ...",
                  flush=True)
            bundle = predict_per_dim_day(
                profile, out_logs, user, day, context, truth, args.predict_all_dims
            )
        forced = hard_rule_forced(bundle, meal)
        per_dim_score = score_bundle(bundle, truth, forced)
        rec = {
            "day": day, "meal": meal, "truth": truth, "forced": forced,
            "per_dim_bundle": {f: bundle.get(f) for f in SCORED_FIELDS if f in truth},
            "per_dim_score": per_dim_score,
            "joint_score": joint_score,
        }
        day_reports.append(rec)
        print_day_report(day, meal, rec)

    # Cross-day rollup.
    def _mean(xs: List[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    pd_accs = [d["per_dim_score"]["accuracy"] for d in day_reports if not args.dry_run]
    pd_corr = [d["per_dim_score"]["num_mismatches"] for d in day_reports if not args.dry_run]
    jt_accs = [d["joint_score"]["accuracy"] for d in day_reports if d.get("joint_score")]
    jt_corr = [d["joint_score"]["num_mismatches"] for d in day_reports if d.get("joint_score")]

    report = {
        "log_dir": str(log_dir),
        "user": user,
        "days": day_reports,
        "physical_profile": profile,
        "physical_profile_source": profile_source,
        "scored_fields": SCORED_FIELDS,
        "predict_all_dims": args.predict_all_dims,
        "dry_run": args.dry_run,
        "summary": {
            "mean_per_dim_accuracy": _mean(pd_accs),
            "mean_per_dim_corrections": _mean(pd_corr),
            "mean_joint_accuracy": _mean(jt_accs),
            "mean_joint_corrections": _mean(jt_corr),
        },
    }

    if not args.dry_run:
        print(f"\n{'#' * 78}")
        print("SUMMARY (mean over days)")
        print(f"{'#' * 78}")
        print(f"  per-dim : acc {_fmt_pct(_mean(pd_accs))} | {_mean(pd_corr):.2f} corrections/day")
        if jt_accs:
            print(f"  joint   : acc {_fmt_pct(_mean(jt_accs))} | {_mean(jt_corr):.2f} corrections/day (live)")

    write_outputs(out_dir, report)
    if not args.dry_run and not args.no_plot:
        plot_learning_curve(report, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
