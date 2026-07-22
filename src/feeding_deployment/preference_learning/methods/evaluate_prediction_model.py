from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from feeding_deployment.preference_learning.methods.metrics import _generate_metrics
from feeding_deployment.preference_learning.methods.utils import (
    _extract_truth_bundle,
    _pred_matches_truth,
    _retry_on_rate_limit,
)

from feeding_deployment.preference_learning.config.physical_capabilities import (
    PHYSICAL_CAPABILITY_PROFILES,
)
from feeding_deployment.preference_learning.methods.prediction_model import PredictionModel, MEMORY_MODES, DEFAULT_MEMORY_MODE
from feeding_deployment.preference_learning.methods.utils import PREF_FIELDS
from feeding_deployment.utils.llm_config import PREDICTION_CLAUDE_MODEL, PREDICTION_EFFORT


def _load_dotenv() -> None:
    """Load KEY=VALUE lines from a .env in the CWD or the repo root into
    os.environ (already-set environment variables win). The Anthropic/OpenAI
    clients read their keys from the environment only, and nothing else in
    the offline eval path loads .env."""
    for env_path in (Path.cwd() / ".env", Path(__file__).resolve().parents[4] / ".env"):
        if not env_path.is_file():
            continue
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val
        break


class _Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, s: str) -> None:
        for st in self._streams:
            st.write(s)
            st.flush()

    def flush(self) -> None:
        for st in self._streams:
            st.flush()

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate LLM-based interactive predictor on synthetic datasets.")
    p.add_argument("--data-file", help="Path to one JSON dataset file.")
    p.add_argument("--data-dir", help="Directory containing JSON dataset files.")
    p.add_argument("--k-retrieve", type=int, default=10)
    p.add_argument("--max-corrections", type=int, default=27)
    p.add_argument("--max-meals", type=int, default=0)
    p.add_argument("--num-rollouts", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--provider",
        choices=["anthropic", "openai"],
        default="anthropic",
        help="Chat provider for the prediction call. 'openai' routes the same "
        "prompts to an OpenAI reasoning model for benchmarking (pass the model "
        "via --openai-model, e.g. gpt-5.6); requires a memory mode with no "
        "LTM-update call (single_full_history / no_memory).",
    )
    p.add_argument(
        "--openai-model",
        default=PREDICTION_CLAUDE_MODEL,
        help="Prediction model id. Defaults to the Anthropic prediction model; "
        "for --provider openai pass an OpenAI model (e.g. gpt-5.6).",
    )
    p.add_argument("--embed-model", default="text-embedding-3-small")
    p.add_argument(
        "--memory-mode", choices=list(MEMORY_MODES), default=None,
        help="Memory backend (PredictionModel.memory_mode); matches run.py's "
             "--pref_memory_mode so offline comparisons mirror deployment. "
             f"Default: {DEFAULT_MEMORY_MODE} (a non-'full' --ablation implies "
             "its mode instead).",
    )
    p.add_argument(
        "--ablation", choices=["full", "ltm_only", "em_only", "no_memory"], default="full",
        help="Three-layer sub-ablation (which of LTM/EM to enable); implies/"
             "requires --memory-mode three_layer. 'no_memory' is a legacy "
             "alias for --memory-mode no_memory.",
    )
    p.add_argument(
        "--prediction-mode",
        choices=["joint", "per_dim"],
        default="joint",
        help="Prediction structure. per_dim = Axis 2 arm: one LLM call per open dimension, "
        "each seeing only that dimension's history; predictions are made once per meal "
        "(m=0) and stay frozen through the correction loop (corrections cannot propagate). "
        "Requires --memory-mode three_layer (top-k retrieved episodes) or "
        "single_full_history (all prior days, no retrieval/summarization).",
    )
    p.add_argument(
        "--per-dim-workers",
        type=int,
        default=9,
        help="Thread pool size for per_dim prediction calls (default: 9).",
    )
    p.add_argument(
        "--profile-label",
        default="",
        choices=[""] + [prof.label for prof in PHYSICAL_CAPABILITY_PROFILES],
        help="Override the dataset's physical_profile_label with a known "
        "capability profile. Needed for deployment logs whose label (e.g. "
        "'manual') is not one of the known profiles the prediction prompt "
        "can describe.",
    )
    p.add_argument(
        "--exclude-missing-dims",
        action="store_true",
        help="Score only the dimensions actually present in each day's "
        "'preferences' record. Dims absent from the dataset (e.g. deployment "
        "logs that predate the plate_color_*/nav_offset_* dims) are excluded "
        "from accuracy/mismatch/correction metrics and from memory updates, "
        "instead of being backfilled with default values.",
    )
    p.add_argument(
        "--resume-dir",
        default="",
        help="Path to an existing reports/run_* directory to resume. Completed days "
        "(those with a day_metrics checkpoint) are replayed from disk with no API "
        "calls; evaluation continues at the first missing day. Requires "
        "--num-rollouts 1 and the same flags as the original run.",
    )
    p.add_argument("--days", type=int, default=0, help="Evaluate only the first N days (0 = use all days in dataset).")

    return p.parse_args()


def _load_files(args: argparse.Namespace) -> List[str]:
    files: List[str] = []
    if args.data_file:
        files.append(args.data_file)
    if args.data_dir:
        files.extend(sorted(glob.glob(os.path.join(args.data_dir, "*.json"))))
    return files


def main() -> int:
    args = parse_args()
    _load_dotenv()

    files = _load_files(args)
    if not files:
        print("No input files provided. Use --data-file or --data-dir.")
        return 1

    # Per-day metric checkpoints (and therefore --resume-dir) require a single
    # rollout: with multiple rollouts the same day runs several times and a
    # checkpoint would be ambiguous.
    checkpoints_enabled = args.num_rollouts == 1
    if args.resume_dir:
        if not checkpoints_enabled:
            raise SystemExit("--resume-dir requires --num-rollouts 1.")
        report_dir = Path(args.resume_dir)
        if not (report_dir / "logs").is_dir():
            raise SystemExit(f"--resume-dir does not look like a run directory (no logs/): {report_dir}")
        run_ts = report_dir.name[len("run_"):] if report_dir.name.startswith("run_") else report_dir.name
        print(f"Resuming run directory: {report_dir}")
    else:
        run_ts = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
        report_dir = Path(__file__).parent / "reports" / f"run_{run_ts}"
        report_dir.mkdir(parents=True, exist_ok=True)

    report_txt_path = report_dir / "report.txt"
    report_txt_file = open(report_txt_path, "a" if args.resume_dir else "w", encoding="utf-8")
    real_stdout = sys.stdout
    sys.stdout = _Tee(real_stdout, report_txt_file)

    logs_dir = report_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # An unset --memory-mode resolves from the ablation (ltm_only/em_only are
    # three-layer sub-ablations; no_memory is the legacy alias for that mode)
    # and otherwise falls back to the deployment default.
    memory_mode = args.memory_mode
    if args.ablation == "no_memory":
        if memory_mode not in (None, "three_layer", "no_memory"):
            raise SystemExit("--ablation no_memory conflicts with --memory-mode " + memory_mode)
        memory_mode = "no_memory"
    elif args.ablation != "full":
        if memory_mode is None:
            memory_mode = "three_layer"
        elif memory_mode != "three_layer":
            raise SystemExit("--ablation ltm_only/em_only only applies to --memory-mode three_layer")
    if memory_mode is None:
        memory_mode = DEFAULT_MEMORY_MODE
    if args.prediction_mode == "per_dim" and memory_mode not in ("three_layer", "single_full_history"):
        raise SystemExit(
            "--prediction-mode per_dim requires --memory-mode three_layer "
            "(top-k retrieved episodes) or single_full_history (all prior days, no retrieval)."
        )

    # The LTM/EM booleans are three-layer sub-ablations; the other modes derive
    # them internally and reject explicit values.
    memory_kwargs: Dict[str, Any] = {"memory_mode": memory_mode}
    if memory_mode == "three_layer":
        memory_kwargs["use_long_term_memory"] = args.ablation != "em_only"
        memory_kwargs["use_episodic_memory"] = args.ablation != "ltm_only"

    try:
        user_reports: List[Dict[str, Any]] = []

        # Wall-clock latency of each live LLM prediction round (one entry per
        # predict_bundle call: the m=0 initial prediction and every joint-mode
        # correction round). per_dim correction rounds are local overlays with
        # no LLM call, so only their m=0 is timed. Days replayed from a
        # checkpoint make no API call and are not timed -- a resumed run
        # therefore averages over its live days only.
        prediction_timings: List[Dict[str, Any]] = []

        for path in files:
            print(f"Evaluating {path} ...", flush=True)
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            user = str(data.get("user", "unknown"))
            physical_profile_label = str(data.get("physical_profile_label", "")).strip()
            if args.profile_label:
                if physical_profile_label and physical_profile_label != args.profile_label:
                    print(
                        f"[user={user}] overriding dataset profile "
                        f"{physical_profile_label!r} with --profile-label "
                        f"{args.profile_label!r}",
                        flush=True,
                    )
                physical_profile_label = args.profile_label
            if not physical_profile_label:
                raise SystemExit(f"Dataset missing required field: 'physical_profile_label' in {path}")

            days: List[Dict[str, Any]] = list(data.get("days", []))
            days.sort(key=lambda r: int(r.get("day", 0)))
            if args.days > 0:
                days = [d for d in days if int(d.get("day", 0)) <= args.days]

            # metrics
            total_meals = 0
            total_corrections = 0
            acc_after_m_sum: Dict[int, float] = {}
            acc_after_m_n: Dict[int, int] = {}

            acc_m0_sum_by_day: Dict[int, float] = defaultdict(float)
            acc_m0_n_by_day: Dict[int, int] = defaultdict(int)
            acc_m1_sum_by_day: Dict[int, float] = defaultdict(float)
            acc_m1_n_by_day: Dict[int, int] = defaultdict(int)

            mismatches_m0_sum_by_day: Dict[int, float] = defaultdict(float)
            mismatches_m0_n_by_day: Dict[int, int] = defaultdict(int)
            mismatches_m1_sum_by_day: Dict[int, float] = defaultdict(float)
            mismatches_m1_n_by_day: Dict[int, int] = defaultdict(int)

            m_star_sum_by_day: Dict[int, float] = defaultdict(float)
            m_star_n_by_day: Dict[int, int] = defaultdict(int)
            affective_state_by_day: Dict[int, str] = {}

            acc_m0_sum_by_state: Dict[str, float] = defaultdict(float)
            acc_m0_n_by_state: Dict[str, int] = defaultdict(int)
            m_star_sum_by_state: Dict[str, float] = defaultdict(float)
            m_star_n_by_state: Dict[str, int] = defaultdict(int)

            per_dim_correct: Dict[str, int] = defaultdict(int)
            per_dim_total: Dict[str, int] = defaultdict(int)

            zero_correction_meals_total = 0
            zero_correction_meals_final_week = 0

            # Per-day metric checkpoints: a day is checkpointed once it is fully
            # finalized (correction loop done + memory updated), so a crashed or
            # resumed run replays completed days from disk with zero API calls.
            day_metrics_dir = logs_dir / user / "day_metrics"
            done_records: Dict[int, Dict[str, Any]] = {}
            if checkpoints_enabled and day_metrics_dir.is_dir():
                for ckpt_path in sorted(day_metrics_dir.glob("day_*.json")):
                    try:
                        rec = json.loads(ckpt_path.read_text(encoding="utf-8"))
                        done_records[int(rec["day"])] = rec
                    except Exception as e:
                        print(f"Warning: unreadable day checkpoint {ckpt_path} ({e}); that day will re-run.", flush=True)

            # Completed days must form a prefix of the evaluation order --
            # memory is cumulative, so a gap would corrupt later days.
            eval_day_order = [int(r.get("day", 0)) for r in days]
            seen_missing = False
            for d in eval_day_order:
                if d not in done_records:
                    seen_missing = True
                elif seen_missing:
                    raise SystemExit(
                        f"Day checkpoints for user {user} are not a contiguous prefix "
                        f"(day {d} is checkpointed after a missing day). Delete the "
                        f"out-of-order files in {day_metrics_dir} and re-run."
                    )
            if done_records:
                print(f"[user={user}] {len(done_records)} day(s) already checkpointed; replaying from disk.", flush=True)

            for rollout_idx in range(args.num_rollouts):
                rng = random.Random(args.seed + rollout_idx)

                prediction_model = PredictionModel(
                    user=user,
                    physical_profile_label=physical_profile_label,
                    chat_model=args.openai_model,
                    embed_model=args.embed_model,
                    provider=args.provider,
                    retry_fn=_retry_on_rate_limit,
                    logs_dir=logs_dir,
                    k_retrieve=args.k_retrieve,
                    prediction_mode=args.prediction_mode,
                    per_dim_workers=args.per_dim_workers,
                    **memory_kwargs,
                )

                meals_this_rollout = 0
                prior_memory_loaded = False
                excluded_logged = False

                def apply_day_result(
                    day: int,
                    affective_state: str,
                    steps: List[Dict[str, Any]],
                    per_dim_m0: Dict[str, bool],
                ) -> None:
                    """Fold one finalized day into the accumulators. Used for both
                    live days and days replayed from a checkpoint, so a resumed
                    run produces the identical report."""
                    nonlocal total_meals, total_corrections, meals_this_rollout
                    nonlocal zero_correction_meals_total, zero_correction_meals_final_week

                    affective_state_by_day[day] = affective_state
                    for st in steps:
                        m_i = int(st["m"])
                        acc_i = float(st["acc"])
                        nm_i = float(st["mismatches"])
                        acc_after_m_sum[m_i] = acc_after_m_sum.get(m_i, 0.0) + acc_i
                        acc_after_m_n[m_i] = acc_after_m_n.get(m_i, 0) + 1
                        if m_i == 0:
                            acc_m0_sum_by_day[day] += acc_i
                            acc_m0_n_by_day[day] += 1
                            mismatches_m0_sum_by_day[day] += nm_i
                            mismatches_m0_n_by_day[day] += 1
                            acc_m0_sum_by_state[affective_state] += acc_i
                            acc_m0_n_by_state[affective_state] += 1
                        elif m_i == 1:
                            acc_m1_sum_by_day[day] += acc_i
                            acc_m1_n_by_day[day] += 1
                            mismatches_m1_sum_by_day[day] += nm_i
                            mismatches_m1_n_by_day[day] += 1

                    for f, ok in per_dim_m0.items():
                        per_dim_total[f] += 1
                        if ok:
                            per_dim_correct[f] += 1

                    last = steps[-1]
                    m_star = int(last["m"])
                    total_meals += 1
                    total_corrections += m_star
                    meals_this_rollout += 1
                    m_star_sum_by_day[day] += m_star
                    m_star_n_by_day[day] += 1
                    m_star_sum_by_state[affective_state] += m_star
                    m_star_n_by_state[affective_state] += 1

                    if m_star == 0 and float(last["mismatches"]) == 0:
                        acc_m1_sum_by_day[day] += 1.0
                        acc_m1_n_by_day[day] += 1
                        mismatches_m1_sum_by_day[day] += 0.0
                        mismatches_m1_n_by_day[day] += 1
                        zero_correction_meals_total += 1
                        if day >= 24:
                            zero_correction_meals_final_week += 1

                for day_rec in days:
                    day = int(day_rec.get("day", 0))
                    context = day_rec.get("context", {}) or {}
                    truth = _extract_truth_bundle(day_rec)
                    if args.exclude_missing_dims:
                        present = set((day_rec.get("preferences") or {}).keys())
                        truth = {f: v for f, v in truth.items() if f in present}
                        excluded = [f for f in PREF_FIELDS if f not in truth]
                        if excluded and not excluded_logged:
                            print(
                                f"[user={user}] excluding {len(excluded)} dim(s) missing "
                                f"from dataset: {', '.join(excluded)}",
                                flush=True,
                            )
                            excluded_logged = True
                    # The scored dims; truth holds every PREF_FIELD unless
                    # --exclude-missing-dims filtered some out.
                    eval_fields = [f for f in PREF_FIELDS if f in truth]

                    if day in done_records:
                        rec = done_records[day]
                        apply_day_result(
                            day,
                            str(rec.get("affective_state", "unknown")),
                            list(rec.get("steps", [])),
                            dict(rec.get("per_dim_m0", {}) or {}),
                        )
                        print(
                            f"[Day {day}] replayed from checkpoint "
                            f"(m*={rec['steps'][-1]['m']}, no API calls)",
                            flush=True,
                        )
                        if args.max_meals and meals_this_rollout >= args.max_meals:
                            break
                        continue

                    if done_records and not prior_memory_loaded:
                        # First live day after checkpointed ones: rehydrate LTM /
                        # episodic / full-history memory from the prior run's day files.
                        prediction_model.load_prior_memory(day)
                        prior_memory_loaded = True

                    corrected: Dict[str, str] = {}
                    m = 0
                    steps: List[Dict[str, Any]] = []
                    per_dim_m0: Dict[str, bool] = {}
                    affective_state = str(context.get("transient_affective_state") or "unknown").strip() or "unknown"

                    print(
                        f"[Day {day}] Meal: {context.get('meal', 'unknown')} | "
                        f"Setting: {context.get('setting', 'unknown')} | "
                        f"Time: {context.get('time_of_day', 'unknown')} | "
                        f"Affective state: {affective_state}",
                        flush=True,
                    )
                    print(f"  Ground truth bundle: {json.dumps(truth, indent=2)}", flush=True)

                    pred: Dict[str, Any] = {}
                    while True:
                        if m == 0 or args.prediction_mode == "joint":
                            print(f"  [Predict] Calling Claude for bundle (day {day}, m={m}) ...", flush=True)
                            _t0 = time.perf_counter()
                            pred = prediction_model.predict_bundle(
                                context=context,
                                corrected=corrected,
                            )
                            _elapsed = time.perf_counter() - _t0
                            prediction_timings.append(
                                {"user": user, "day": day, "m": m, "seconds": _elapsed}
                            )
                            print(f"  [Predict] round latency: {_elapsed:.1f}s (m={m})", flush=True)
                        else:
                            # per_dim: no cross-dim visibility, so a correction
                            # cannot change any other dim's prediction --
                            # overlay the correction + hard rules on the frozen
                            # m=0 bundle instead of re-predicting (no LLM calls).
                            print(f"  [Predict] per_dim: reapplying constraints (day {day}, m={m}) ...", flush=True)
                            pred = prediction_model.reapply_constraints(pred, context, corrected)
                        print(f"  [Predict] Prediction: {json.dumps(pred, indent=2)}", flush=True)

                        unrevealed = [f for f in eval_fields if f not in corrected]
                        acc = (
                            (
                                sum(1 for f in unrevealed if _pred_matches_truth(f, pred.get(f), truth.get(f)))
                                / float(len(unrevealed))
                            )
                            if unrevealed
                            else 1.0
                        )

                        mismatches = [
                            f
                            for f in eval_fields
                            if f not in corrected and not _pred_matches_truth(f, pred.get(f), truth.get(f))
                        ]
                        num_mismatches = len(mismatches)

                        steps.append({"m": m, "acc": acc, "mismatches": num_mismatches})
                        if m == 0:
                            per_dim_m0 = {
                                f: _pred_matches_truth(f, pred.get(f), truth.get(f)) for f in unrevealed
                            }

                        print(
                            f"  [user={user}] rollout {rollout_idx+1}/{args.num_rollouts} "
                            f"day {day} m={m}: acc_unrevealed={acc:.3f} "
                            f"mismatches={num_mismatches}",
                            flush=True,
                        )

                        if not mismatches or m >= args.max_corrections:
                            print(
                                f"  Meal finished after {m} corrections\n",
                                flush=True,
                            )
                            break

                        f_corr = rng.choice(mismatches)
                        corrected_value = truth.get(f_corr, "")

                        print(
                            f"    correcting {f_corr} -> {corrected_value}",
                            flush=True,
                        )

                        corrected[f_corr] = corrected_value
                        m += 1

                    # Build episode text (used for long_term_memory_model update and retrieval history)
                    prediction_model.update(day, context, corrected, truth)

                    apply_day_result(day, affective_state, steps, per_dim_m0)

                    if checkpoints_enabled:
                        # Checkpoint AFTER update(): a checkpointed day always has
                        # its memory day-files on disk. Atomic write so a crash
                        # mid-write never leaves a corrupt checkpoint.
                        day_metrics_dir.mkdir(parents=True, exist_ok=True)
                        ckpt = {
                            "day": day,
                            "affective_state": affective_state,
                            "context": context,
                            "steps": steps,
                            "per_dim_m0": per_dim_m0,
                        }
                        ckpt_path = day_metrics_dir / f"day_{day:04d}.json"
                        tmp_path = ckpt_path.with_suffix(".json.tmp")
                        tmp_path.write_text(json.dumps(ckpt, ensure_ascii=False, indent=2), encoding="utf-8")
                        os.replace(tmp_path, ckpt_path)

                    if args.max_meals and meals_this_rollout >= args.max_meals:
                        break

            mean_corr = (total_corrections / float(total_meals)) if total_meals else 0.0
            acc_after_m = {str(mm): (acc_after_m_sum[mm] / float(acc_after_m_n[mm])) for mm in sorted(acc_after_m_sum)}

            all_days = sorted(
                set(acc_m0_sum_by_day)
                | set(acc_m1_sum_by_day)
                | set(mismatches_m0_sum_by_day)
                | set(mismatches_m1_sum_by_day)
                | set(m_star_sum_by_day)
            )

            per_day_metrics: List[Dict[str, Any]] = []
            for d in all_days:
                rec: Dict[str, Any] = {"day": d, "affective_state": affective_state_by_day.get(d, "unknown")}
                if acc_m0_n_by_day[d]:
                    rec["acc_m0"] = acc_m0_sum_by_day[d] / float(acc_m0_n_by_day[d])
                if acc_m1_n_by_day[d]:
                    rec["acc_m1"] = acc_m1_sum_by_day[d] / float(acc_m1_n_by_day[d])
                if mismatches_m0_n_by_day[d]:
                    rec["mismatches_m0"] = mismatches_m0_sum_by_day[d] / float(mismatches_m0_n_by_day[d])
                if mismatches_m1_n_by_day[d]:
                    rec["mismatches_m1"] = mismatches_m1_sum_by_day[d] / float(mismatches_m1_n_by_day[d])
                if m_star_n_by_day[d]:
                    rec["m_star"] = m_star_sum_by_day[d] / float(m_star_n_by_day[d])
                per_day_metrics.append(rec)

            by_affective_state: Dict[str, Dict[str, float]] = {}
            for state in sorted(set(acc_m0_n_by_state) | set(m_star_n_by_state)):
                by_affective_state[state] = {}
                if acc_m0_n_by_state[state]:
                    by_affective_state[state]["acc_m0"] = acc_m0_sum_by_state[state] / float(acc_m0_n_by_state[state])
                if m_star_n_by_state[state]:
                    by_affective_state[state]["mean_m_star"] = m_star_sum_by_state[state] / float(m_star_n_by_state[state])

            # Only dims that were ever scored: with --exclude-missing-dims the
            # absent dims would otherwise show as misleading 0.0 bars.
            per_dimension_m0_accuracy: Dict[str, float] = {
                f: per_dim_correct[f] / float(per_dim_total[f])
                for f in PREF_FIELDS
                if per_dim_total[f]
            }

            summary_statistics = {
                "zero_correction_meals_total": zero_correction_meals_total,
                "zero_correction_meals_final_week": zero_correction_meals_final_week,
            }

            user_reports.append(
                {
                    "file": path,
                    "user": user,
                    "physical_profile_label": physical_profile_label,
                    "meals": total_meals,
                    "num_rollouts": args.num_rollouts,
                    "mean_corrections_to_stop": mean_corr,
                    "accuracy_after_m": acc_after_m,
                    "per_day_metrics": per_day_metrics,
                    "by_affective_state": by_affective_state,
                    "per_dimension_m0_accuracy": per_dimension_m0_accuracy,
                    "summary_statistics": summary_statistics,
                }
            )

        def _latency_stats(xs: List[float]) -> Dict[str, float]:
            xs_sorted = sorted(xs)
            n = len(xs_sorted)
            median = (
                xs_sorted[n // 2]
                if n % 2
                else (xs_sorted[n // 2 - 1] + xs_sorted[n // 2]) / 2.0
            )
            return {
                "count": n,
                "mean_s": sum(xs_sorted) / n,
                "median_s": median,
                "min_s": xs_sorted[0],
                "max_s": xs_sorted[-1],
            }

        timing_summary: Dict[str, Any] = {}
        if prediction_timings:
            all_s = [t["seconds"] for t in prediction_timings]
            m0_s = [t["seconds"] for t in prediction_timings if t["m"] == 0]
            corr_s = [t["seconds"] for t in prediction_timings if t["m"] > 0]
            timing_summary = {
                "provider": args.provider,
                "effort": PREDICTION_EFFORT,
                "model": args.openai_model,
                "all_rounds": _latency_stats(all_s),
                "initial_m0": _latency_stats(m0_s) if m0_s else None,
                "correction_rounds": _latency_stats(corr_s) if corr_s else None,
            }
            allr = timing_summary["all_rounds"]
            print("\n=== Prediction latency (this run) ===", flush=True)
            print(f"  provider={args.provider}  model={args.openai_model}  effort={PREDICTION_EFFORT}", flush=True)
            print(
                f"  all rounds:        n={allr['count']}  mean={allr['mean_s']:.1f}s  "
                f"median={allr['median_s']:.1f}s  min={allr['min_s']:.1f}s  max={allr['max_s']:.1f}s",
                flush=True,
            )
            if corr_s:
                cr = timing_summary["correction_rounds"]
                print(
                    f"  correction rounds: n={cr['count']}  mean={cr['mean_s']:.1f}s  median={cr['median_s']:.1f}s",
                    flush=True,
                )
            if m0_s:
                im = timing_summary["initial_m0"]
                print(f"  initial (m=0):     n={im['count']}  mean={im['mean_s']:.1f}s", flush=True)

        report_path = report_dir / "report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "users": user_reports,
                    "run_timestamp": run_ts,
                    "memory_mode": memory_mode,
                    "ablation": args.ablation,
                    "prediction_mode": args.prediction_mode,
                    "exclude_missing_dims": args.exclude_missing_dims,
                    "k_retrieve": args.k_retrieve,
                    "num_rollouts": args.num_rollouts,
                    "prediction_timing": timing_summary,
                    "prediction_timings": prediction_timings,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        print(f"\nWrote report: {report_path}")
        print(f"Terminal output saved to: {report_txt_path}")

        _generate_metrics(user_reports, report_dir)
        return 0

    finally:
        sys.stdout = real_stdout
        report_txt_file.close()


if __name__ == "__main__":
    raise SystemExit(main())
