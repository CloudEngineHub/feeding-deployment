"""Tests for the Axis 1 (single_memory) and Axis 2 (per_dim) experiment arms.
All LLM/embedding calls are mocked -- these tests verify information flow,
not model quality.

Run with:
    PYTHONPATH=src ANTHROPIC_API_KEY=dummy python -m pytest tests/test_memory_prediction_modes.py -v
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

os.environ.setdefault("ANTHROPIC_API_KEY", "dummy")

from feeding_deployment.preference_learning.config.preference_bundle import (
    PREFERENCE_BUNDLE,
    parse_color,
    parse_nav_offset,
)
from feeding_deployment.preference_learning.methods.episodic_memory import EpisodicMemoryModel
from feeding_deployment.preference_learning.methods.prediction_model import PredictionModel
from feeding_deployment.preference_learning.methods.prompts.dim_prediction import (
    get_dim_prediction_prompt,
)
from feeding_deployment.preference_learning.methods.utils import PREF_FIELDS

PROFILE = "limited_arms_no_trunk_good_head"
CONTEXT = {"meal": "chicken nuggets", "setting": "Personal", "time_of_day": "noon", "transient_affective_state": "Neutral"}
KIND_BY_FIELD = {d.field: d.kind for d in PREFERENCE_BUNDLE}
OPTIONS_BY_FIELD = {d.field: d.options for d in PREFERENCE_BUNDLE}


def _fake_truth() -> dict:
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


def _make_meta(day: int, corrected_fields=()):
    return {"day": day, "context": dict(CONTEXT), "prefs": _fake_truth(), "corrected_fields": list(corrected_fields)}


def _fake_response(payload: dict):
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=json.dumps(payload))],
        usage=SimpleNamespace(speed="standard"),
    )


class TestEpisodicRecords:
    def _model(self):
        m = EpisodicMemoryModel(client=None, embed_model="x", cache_path=Path("/nonexistent/cache.json"),
                                retry_fn=lambda fn: fn(), k_retrieve=2)
        # Deterministic fake embeddings: episodes mentioning "noon" align with a noon query.
        m._embed = lambda text: [1.0, 0.0] if "noon" in text else [0.0, 1.0]
        return m

    def test_retrieve_and_retrieve_records_rank_identically(self):
        m = self._model()
        m.add_episode("day=1; time_of_day=noon; prefs", meta=_make_meta(1))
        m.add_episode("day=2; time_of_day=evening; prefs", meta=_make_meta(2))
        m.add_episode("day=3; time_of_day=noon; prefs", meta=_make_meta(3))
        joined = m.retrieve(CONTEXT, {})
        records = m.retrieve_records(CONTEXT, {})
        assert [r["episode_text"] for r in records] == joined.split("\n\n")
        assert [r["meta"]["day"] for r in records] == [1, 3]  # noon episodes win, stable order

    def test_meta_alignment_enforced(self):
        m = self._model()
        with pytest.raises(ValueError):
            m.load_history(["a", "b"], metas=[None])

    def test_missing_meta_is_none(self):
        m = self._model()
        m.add_episode("day=1; time_of_day=noon; prefs")  # no meta
        recs = m.retrieve_records(CONTEXT, {})
        assert recs[0]["meta"] is None


class TestDimPredictionPrompt:
    def test_no_cross_dim_leakage(self):
        records = [{"episode_text": "irrelevant", "meta": _make_meta(1, corrected_fields=["robot_speed"])}]
        ltm = json.dumps({f: {"default": "x", "user_tendencies": "y"} for f in PREF_FIELDS})
        prompt = get_dim_prediction_prompt(
            field="robot_speed", physical_profile_label=PROFILE, ltm_summary=ltm,
            records=records, context=CONTEXT, option_line="- robot_speed: [slow, medium, fast]",
        )
        for other in PREF_FIELDS:
            if other != "robot_speed":
                assert other not in prompt, f"per-dim prompt for robot_speed leaks {other}"
        assert "(user actively corrected this that day)" in prompt

    def test_metaless_episodes_skipped(self):
        prompt = get_dim_prediction_prompt(
            field="robot_speed", physical_profile_label=PROFILE, ltm_summary="",
            records=[{"episode_text": "day=1; transfer_mode=inside mouth transfer", "meta": None}],
            context=CONTEXT, option_line="- robot_speed: [slow, medium, fast]",
        )
        assert "transfer_mode" not in prompt
        assert "(empty)" in prompt

    def test_dip_rule_only_in_its_own_prompt(self):
        kw = dict(physical_profile_label=PROFILE, ltm_summary="", records=[], context=CONTEXT)
        p_dip = get_dim_prediction_prompt(field="bite_dipping_preference", option_line="- x", **kw)
        p_speed = get_dim_prediction_prompt(field="robot_speed", option_line="- x", **kw)
        assert "HARD RULE" in p_dip and "HARD RULE" not in p_speed


class TestPerDimPrediction:
    def _model(self, tmp_path: Path) -> PredictionModel:
        return PredictionModel(
            user="u1", physical_profile_label=PROFILE, logs_dir=tmp_path,
            use_long_term_memory=False, use_episodic_memory=False,
            prediction_mode="per_dim", per_dim_workers=4,
        )

    def test_calls_only_open_fields_and_pins(self, tmp_path):
        pm = self._model(tmp_path)
        called: list = []

        def fake_create(**kwargs):
            prompt = kwargs["messages"][0]["content"]
            field = prompt.split("CURRENT MEAL: ", 1)[1].split(".", 1)[0]
            called.append(field)
            if KIND_BY_FIELD[field] == "color":
                value = {"h": 10, "s": 20, "v": 30, "range": 0.1}
            elif KIND_BY_FIELD[field] == "nav_offset":
                value = {"dx": 0.01, "dy": 0.02, "dyaw": 0.03}
            elif KIND_BY_FIELD[field] == "text":
                value = "chicken first"
            else:
                value = OPTIONS_BY_FIELD[field][-1]
            return _fake_response({"explanation": "test", "value": value})

        corrected = {"robot_speed": "slow", "plate_color_table": "h=90,s=200,v=150,range=0.10"}
        with patch.object(pm, "_create_prediction_message", side_effect=fake_create):
            pred = pm.predict_bundle(context=CONTEXT, corrected=corrected)

        assert sorted(called) == sorted(f for f in PREF_FIELDS if f not in corrected)
        assert pred["robot_speed"] == "slow"  # pinned verbatim
        assert pred["plate_color_table"] == parse_color("h=90,s=200,v=150,range=0.10")  # string pin canonicalized
        assert pred["bite_ordering"] == "chicken first"
        assert pm.last_latent_inference == ""
        assert pm.last_explanations["microwave_time"] == "test"
        # per-call logs carry the field name
        logs = list((tmp_path / "u1" / "prediction_model_llm_calls").glob("*_robot_speed.txt"))
        assert not logs  # pinned dim never called/logged
        assert list((tmp_path / "u1" / "prediction_model_llm_calls").glob("*_microwave_time.txt"))

    def test_cross_dim_hard_rule_applied_post_hoc(self, tmp_path):
        pm = self._model(tmp_path)

        def fake_create(**kwargs):
            prompt = kwargs["messages"][0]["content"]
            field = prompt.split("CURRENT MEAL: ", 1)[1].split(".", 1)[0]
            if field == "transfer_mode":
                value = "inside mouth transfer"
            elif field == "outside_mouth_distance":
                value = "near"  # contradicts transfer_mode; must be overridden post-hoc
            elif KIND_BY_FIELD[field] == "color":
                value = {"h": 1, "s": 1, "v": 1, "range": 0.1}
            elif KIND_BY_FIELD[field] == "nav_offset":
                value = {"dx": 0, "dy": 0, "dyaw": 0}
            elif KIND_BY_FIELD[field] == "text":
                value = "no particular order"
            else:
                value = OPTIONS_BY_FIELD[field][0]
            return _fake_response({"explanation": "t", "value": value})

        with patch.object(pm, "_create_prediction_message", side_effect=fake_create):
            pred = pm.predict_bundle(context=CONTEXT, corrected={})
        assert pred["transfer_mode"] == "inside mouth transfer"
        assert pred["outside_mouth_distance"] == "not applicable"

    def test_reapply_constraints_no_llm(self, tmp_path):
        pm = self._model(tmp_path)
        pred = _fake_truth()
        pred["transfer_mode"] = "outside mouth transfer"
        with patch.object(pm, "_create_prediction_message", side_effect=AssertionError("no LLM calls allowed")):
            out = pm.reapply_constraints(
                pred, CONTEXT,
                corrected={"transfer_mode": "inside mouth transfer", "nav_offset_dining_table": "dx=0.213,dy=-0.081,dyaw=0.140"},
            )
        assert out["transfer_mode"] == "inside mouth transfer"
        assert out["outside_mouth_distance"] == "not applicable"  # rule fires from the correction
        assert out["nav_offset_dining_table"] == parse_nav_offset("dx=0.213,dy=-0.081,dyaw=0.140")
        # untouched open dims stay frozen
        assert out["robot_speed"] == pred["robot_speed"]


class TestSingleMemoryMode:
    def test_prompt_contains_all_episodes_chronologically(self, tmp_path):
        pm = PredictionModel(user="u1", physical_profile_label=PROFILE, logs_dir=tmp_path,
                             memory_mode="single_memory")
        truth = _fake_truth()
        for day in (1, 2, 3):
            ctx = dict(CONTEXT, meal=f"meal{day}")
            pm.update(day, ctx, {}, truth)

        captured = {}

        def fake_create(**kwargs):
            captured["prompt"] = kwargs["messages"][0]["content"]
            return _fake_response({f: truth[f] for f in PREF_FIELDS})

        with patch.object(pm, "_create_prediction_message", side_effect=fake_create):
            pm.predict_bundle(context=CONTEXT, corrected={})

        p = captured["prompt"]
        assert "MEMORY: all prior meals in chronological order" in p
        assert p.index("meal=meal1") < p.index("meal=meal2") < p.index("meal=meal3")
        assert "SEMANTIC MEMORY" not in p and "EPISODIC MEMORY" not in p
