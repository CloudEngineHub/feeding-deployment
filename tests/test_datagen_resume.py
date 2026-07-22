"""Per-day persistence + in-place resume for the deployment dataset generator
(no API calls; the joint-generation LLM call is stubbed).

Run with:
    PYTHONPATH=src python -m pytest tests/test_datagen_resume.py -v
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

import feeding_deployment.preference_learning.data_generation.generate_deployment_dataset_llm as gen
from feeding_deployment.preference_learning.data_generation.continuous_prefs import (
    sample_continuous_tables,
)

PROFILE = "limited_arms_no_trunk_good_head"


def _stub_llm(monkeypatch, calls: list) -> None:
    def fake_generate(client, **kwargs):
        calls.append(kwargs["meal_info"]["meal"])
        choices = {}
        for dim in gen.LLM_DIMS:
            choices[dim.field] = "no particular order" if dim.kind == "text" else dim.options[0]
        rationales = {f: "stub" for f in choices}
        return choices, rationales

    monkeypatch.setattr(gen, "generate_joint_preferences_with_llm", fake_generate)


def _run(tmp_path: Path, days: int) -> str:
    tables = sample_continuous_tables(random.Random("0:1"))
    return gen.run_deployment(
        client=None,
        user_name="u1",
        deployment_id="dep1",
        physical_profile_label=PROFILE,
        user_preference_encoding={},
        continuous_tables=tables,
        seed=0,
        days=days,
        model="stub-model",
        output_dir=str(tmp_path),
        output_filename="u1.json",
    )


class TestDatagenResume:
    def test_file_written_after_every_day(self, monkeypatch, tmp_path):
        calls: list = []
        _stub_llm(monkeypatch, calls)
        out = _run(tmp_path, 2)
        data = json.loads(Path(out).read_text(encoding="utf-8"))
        assert len(data["days"]) == 2 and len(calls) == 2

    def test_resume_generates_only_missing_days(self, monkeypatch, tmp_path):
        calls: list = []
        _stub_llm(monkeypatch, calls)
        _run(tmp_path, 2)
        first = json.loads((tmp_path / "u1.json").read_text(encoding="utf-8"))

        calls.clear()
        out = _run(tmp_path, 4)  # resume in place, extend to 4 days
        data = json.loads(Path(out).read_text(encoding="utf-8"))
        assert len(calls) == 2  # only days 3-4 hit the LLM
        assert [d["day"] for d in data["days"]] == [1, 2, 3, 4]
        assert data["days"][:2] == first["days"]  # recorded days kept verbatim
        # rng stream stayed aligned: day 3-4 contexts match a fresh 4-day run
        fresh_dir = tmp_path / "fresh"
        fresh_dir.mkdir()
        calls.clear()
        fresh = json.loads(Path(_run(fresh_dir, 4)).read_text(encoding="utf-8"))
        assert [d["context"] for d in data["days"]] == [d["context"] for d in fresh["days"]]

    def test_resume_rejects_config_mismatch(self, monkeypatch, tmp_path):
        calls: list = []
        _stub_llm(monkeypatch, calls)
        _run(tmp_path, 1)
        tables = sample_continuous_tables(random.Random("0:1"))
        with pytest.raises(ValueError, match="seed"):
            gen.run_deployment(
                client=None, user_name="u1", deployment_id="dep1",
                physical_profile_label=PROFILE, user_preference_encoding={},
                continuous_tables=tables, seed=99, days=2, model="stub-model",
                output_dir=str(tmp_path), output_filename="u1.json",
            )
