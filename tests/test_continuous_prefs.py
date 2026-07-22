"""Tests for the programmatic continuous-dim ground truth (Phase 0.0).

Run with:
    PYTHONPATH=src python -m pytest tests/test_continuous_prefs.py -v
"""

from __future__ import annotations

import json
import random

import pytest

from feeding_deployment.preference_learning.config.affective_state import AFFECTIVE_STATES
from feeding_deployment.preference_learning.config.mealtime_context import TIMES_OF_DAY
from feeding_deployment.preference_learning.config.preference_bundle import (
    COLOR_FIELD_BY_LOCATION,
    OFFSET_FIELD_BY_LOCATION,
)
from feeding_deployment.preference_learning.data_generation.continuous_prefs import (
    CONTINUOUS_FIELDS,
    continuous_truth,
    render_continuous_tendencies,
    sample_continuous_tables,
)

COLOR_FIELDS = list(COLOR_FIELD_BY_LOCATION.values())
NAV_FIELDS = list(OFFSET_FIELD_BY_LOCATION.values())


def _all_contexts():
    for tod in TIMES_OF_DAY:
        for state in AFFECTIVE_STATES:
            yield tod, state


class TestSampling:
    def test_deterministic_given_seed(self):
        t1 = sample_continuous_tables(random.Random(42))
        t2 = sample_continuous_tables(random.Random(42))
        assert t1 == t2
        assert sample_continuous_tables(random.Random(43)) != t1

    def test_json_round_trip(self):
        tables = sample_continuous_tables(random.Random(0))
        assert json.loads(json.dumps(tables)) == tables

    def test_reference_components_are_zero(self):
        tables = sample_continuous_tables(random.Random(0))
        assert all(v == 0 for v in tables["time_color_shifts"]["noon"].values())
        assert all(v == 0.0 for v in tables["time_nav_shifts"]["noon"].values())
        assert all(v == 0.0 for v in tables["affect_nav_shifts"]["Neutral"].values())
        assert all(v == 0 for v in tables["location_color_offsets"]["table"].values())


class TestTruth:
    @pytest.mark.parametrize("seed", range(50))
    def test_no_clipping_over_full_context_space(self, seed):
        """continuous_truth asserts values survive parse_* unchanged; sweep
        every (tod, affect) for many seeds so a bad sampling range fails here,
        not mid-regeneration."""
        tables = sample_continuous_tables(random.Random(seed))
        for tod, state in _all_contexts():
            vals = continuous_truth(tables, tod, state)
            assert sorted(vals) == CONTINUOUS_FIELDS

    def test_same_context_same_value(self):
        tables = sample_continuous_tables(random.Random(7))
        assert continuous_truth(tables, "morning", "Hurried") == continuous_truth(
            tables, "morning", "Hurried"
        )

    def test_color_time_shift_shared_across_locations(self):
        tables = sample_continuous_tables(random.Random(7))
        noon = continuous_truth(tables, "noon", "Neutral")
        morning = continuous_truth(tables, "morning", "Neutral")
        deltas = {
            f: (morning[f]["h"] - noon[f]["h"], morning[f]["s"] - noon[f]["s"], morning[f]["v"] - noon[f]["v"])
            for f in COLOR_FIELDS
        }
        assert len(set(deltas.values())) == 1  # one shared lighting shift
        assert deltas[COLOR_FIELDS[0]] != (0, 0, 0)

    def test_color_has_no_affect_term(self):
        tables = sample_continuous_tables(random.Random(7))
        a = continuous_truth(tables, "noon", "Neutral")
        b = continuous_truth(tables, "noon", "Fatigued")
        for f in COLOR_FIELDS:
            assert a[f] == b[f]

    def test_nav_affect_shift_shared_across_locations(self):
        tables = sample_continuous_tables(random.Random(7))
        neutral = continuous_truth(tables, "noon", "Neutral")
        fatigued = continuous_truth(tables, "noon", "Fatigued")
        deltas = {
            f: tuple(round(fatigued[f][k] - neutral[f][k], 3) for k in ("dx", "dy", "dyaw"))
            for f in NAV_FIELDS
        }
        assert len(set(deltas.values())) == 1  # one shared affect component
        assert deltas[NAV_FIELDS[0]] != (0.0, 0.0, 0.0)

    def test_nav_bases_independent_across_locations(self):
        tables = sample_continuous_tables(random.Random(7))
        neutral = continuous_truth(tables, "noon", "Neutral")
        bases = {tuple(neutral[f].values()) for f in NAV_FIELDS}
        assert len(bases) == len(NAV_FIELDS)

    def test_unknown_context_raises(self):
        tables = sample_continuous_tables(random.Random(0))
        with pytest.raises(KeyError):
            continuous_truth(tables, "midnight", "Neutral")
        with pytest.raises(KeyError):
            continuous_truth(tables, "noon", "Grumpy")


class TestRenderTendencies:
    def test_all_fields_standard_shape(self):
        tables = sample_continuous_tables(random.Random(3))
        enc = render_continuous_tendencies(tables)
        assert sorted(enc) == CONTINUOUS_FIELDS
        for field, entry in enc.items():
            assert set(entry) == {"default", "user_tendencies"}
            assert entry["default"]
            assert "IF" in entry["user_tendencies"]
