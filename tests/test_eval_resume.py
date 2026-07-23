"""End-to-end test of per-day checkpointing + --resume-dir in
evaluate_prediction_model (no API calls; PredictionModel is stubbed).

Run with:
    PYTHONPATH=src ANTHROPIC_API_KEY=dummy python -m pytest tests/test_eval_resume.py -v
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("ANTHROPIC_API_KEY", "dummy")

import feeding_deployment.preference_learning.methods.evaluate_prediction_model as evalmod
from feeding_deployment.preference_learning.config.preference_bundle import PREFERENCE_BUNDLE
from feeding_deployment.preference_learning.methods.utils import PREF_FIELDS

PROFILE = "limited_arms_no_trunk_good_head"


def _truth_bundle() -> dict:
    out = {}
    for d in PREFERENCE_BUNDLE:
        if d.kind == "color":
            out[d.field] = {"h": 90, "s": 200, "v": 150, "range": 0.1}
        elif d.kind == "nav_offset":
            out[d.field] = {"dx": 0.1, "dy": -0.05, "dyaw": 0.2}
        elif d.kind == "text":
            out[d.field] = "no particular order"
        else:
            out[d.field] = d.options[0]
    return out


def _make_dataset(path: Path, n_days: int) -> None:
    truth = _truth_bundle()
    days = [
        {
            "day": d,
            "context": {
                "meal": "chicken nuggets",
                "setting": "Personal",
                "time_of_day": "noon",
                "transient_affective_state": "Neutral",
            },
            "preferences": {f: {"choice": truth[f], "rationale": ""} for f in PREF_FIELDS},
        }
        for d in range(1, n_days + 1)
    ]
    path.write_text(
        json.dumps({"user": "u1", "physical_profile_label": PROFILE, "days": days}), encoding="utf-8"
    )


class StubPredictionModel:
    """Deterministic stand-in: predicts truth everywhere except robot_speed
    (wrong until corrected), so every meal converges in exactly one correction
    and the correction order is rng-independent (singleton mismatch list)."""

    instances: list = []

    def __init__(self, **kwargs):
        self.logs_dir = kwargs["logs_dir"]
        self.user = kwargs["user"]
        self.predict_days: list = []
        self.loaded_prior_from: int | None = None
        StubPredictionModel.instances.append(self)

    def load_prior_memory(self, current_day: int) -> None:
        self.loaded_prior_from = current_day

    def predict_bundle(self, context, corrected, **kwargs):
        truth = _truth_bundle()
        pred = dict(truth)
        if "robot_speed" not in corrected:
            pred["robot_speed"] = "medium" if truth["robot_speed"] != "medium" else "fast"
        pred.update(corrected)
        return pred

    def reapply_constraints(self, pred, context, corrected, **kwargs):
        out = dict(pred)
        out.update(corrected)
        return out

    def update(self, day, context, corrected, ground_truth_bundle):
        self.predict_days.append(day)


def _run_eval(monkeypatch, base: Path, argv_extra: list) -> dict:
    """Run evalmod.main() with reports rooted at ``base`` (each logical run gets
    its own base so two runs started within the same second cannot collide on
    the second-resolution run_<ts> directory name)."""
    base.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(evalmod, "PredictionModel", StubPredictionModel)
    monkeypatch.setattr(evalmod, "__file__", str(base / "evaluate_prediction_model.py"))
    monkeypatch.setattr(evalmod, "_generate_metrics", lambda *a, **k: None)
    monkeypatch.setattr(sys, "argv", ["prog"] + argv_extra)
    rc = evalmod.main()
    assert rc == 0
    runs = sorted((base / "reports").glob("run_*"))
    report = json.loads((runs[-1] / "report.json").read_text(encoding="utf-8"))
    return {"report": report, "run_dir": runs[-1]}


def _strip_volatile(report: dict) -> dict:
    out = dict(report)
    out.pop("run_timestamp", None)
    # Wall-clock prediction timings: an aggregate summary and the per-prediction
    # list. Both are inherently run-dependent (and only the live day is timed on
    # resume), so they are not part of the reproducible metrics.
    out.pop("prediction_timing", None)
    out.pop("prediction_timings", None)
    return out


class TestEvalResume:
    def test_resume_produces_identical_report(self, monkeypatch, tmp_path):
        data_file = tmp_path / "dataset.json"
        _make_dataset(data_file, 3)

        # Reference: uninterrupted 3-day run.
        ref = _run_eval(monkeypatch, tmp_path / "ref", ["--data-file", str(data_file), "--days", "3"])

        # Interrupted: 2 days, then resume the same run dir for day 3.
        StubPredictionModel.instances.clear()
        part = _run_eval(monkeypatch, tmp_path / "part", ["--data-file", str(data_file), "--days", "2"])
        ckpts = sorted((part["run_dir"] / "logs" / "u1" / "day_metrics").glob("day_*.json"))
        assert [p.name for p in ckpts] == ["day_0001.json", "day_0002.json"]

        StubPredictionModel.instances.clear()
        resumed = _run_eval(
            monkeypatch,
            tmp_path / "part",
            ["--data-file", str(data_file), "--days", "3", "--resume-dir", str(part["run_dir"])],
        )
        assert resumed["run_dir"] == part["run_dir"]  # same directory reused

        # Only day 3 ran live, after rehydrating memory from day 3's viewpoint.
        stub = StubPredictionModel.instances[-1]
        assert stub.predict_days == [3]
        assert stub.loaded_prior_from == 3

        # Metrics identical to the uninterrupted run.
        assert _strip_volatile(resumed["report"]) == _strip_volatile(ref["report"])

    def test_gap_in_checkpoints_rejected(self, monkeypatch, tmp_path):
        data_file = tmp_path / "dataset.json"
        _make_dataset(data_file, 3)
        part = _run_eval(monkeypatch, tmp_path / "gap", ["--data-file", str(data_file), "--days", "2"])
        (part["run_dir"] / "logs" / "u1" / "day_metrics" / "day_0001.json").unlink()

        StubPredictionModel.instances.clear()
        monkeypatch.setattr(
            sys, "argv",
            ["prog", "--data-file", str(data_file), "--days", "3", "--resume-dir", str(part["run_dir"])],
        )
        with pytest.raises(SystemExit, match="contiguous prefix"):
            evalmod.main()

    def test_resume_requires_single_rollout(self, monkeypatch, tmp_path):
        data_file = tmp_path / "dataset.json"
        _make_dataset(data_file, 1)
        monkeypatch.setattr(
            sys, "argv",
            ["prog", "--data-file", str(data_file), "--num-rollouts", "2", "--resume-dir", str(tmp_path)],
        )
        with pytest.raises(SystemExit, match="num-rollouts 1"):
            evalmod.main()
