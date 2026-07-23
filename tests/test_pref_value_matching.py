"""Tests for field-aware truth extraction / prediction matching (Phase 0.3 of
the memory-vs-prediction-structure experiment).

The eval loop pins truth values as *formatted strings* ("h=..,s=.." /
"dx=.."), while predictions carry canonical dicts for color/nav dims. These
tests lock in that every comparison and prompt-rendering path canonicalizes
both representations instead of comparing a dict against its string encoding
(guaranteed mismatch) or collapsing a string correction to the factory default.

Run with:
    PYTHONPATH=src python -m pytest tests/test_pref_value_matching.py -v
"""

from __future__ import annotations

import pytest

from feeding_deployment.preference_learning.config.preference_bundle import (
    DEFAULT_COLOR,
    DEFAULT_NAV_OFFSET,
    format_color,
    format_nav_offset,
    parse_color,
    parse_nav_offset,
)
from feeding_deployment.preference_learning.methods.prediction_model import (
    _format_corrected_block,
)
from feeding_deployment.preference_learning.methods.utils import (
    _extract_truth_bundle,
    _format_pref_value,
    _pred_matches_truth,
)

COLOR_DICT = {"h": 101, "s": 200, "v": 198, "range": 0.1}
COLOR_STR = "h=101,s=200,v=198,range=0.10"
NAV_DICT = {"dx": 0.213, "dy": -0.081, "dyaw": 0.14}
NAV_STR = "dx=0.213,dy=-0.081,dyaw=0.140"


class TestFormatPrefValue:
    def test_color_dict_and_string_agree(self):
        assert _format_pref_value("plate_color_fridge", COLOR_DICT) == COLOR_STR
        assert _format_pref_value("plate_color_fridge", COLOR_STR) == COLOR_STR

    def test_nav_dict_and_string_agree(self):
        assert _format_pref_value("nav_offset_dining_table", NAV_DICT) == NAV_STR
        assert _format_pref_value("nav_offset_dining_table", NAV_STR) == NAV_STR

    def test_categorical_passthrough(self):
        assert _format_pref_value("robot_speed", "slow") == "slow"


class TestExtractTruthBundle:
    def test_color_and_nav_are_compact_strings(self):
        day_rec = {
            "preferences": {
                "plate_color_table": {"choice": COLOR_DICT, "rationale": ""},
                "nav_offset_sink": {"choice": NAV_DICT, "rationale": ""},
                "robot_speed": {"choice": "medium", "rationale": ""},
            }
        }
        truth = _extract_truth_bundle(day_rec)
        assert truth["plate_color_table"] == COLOR_STR
        assert truth["nav_offset_sink"] == NAV_STR
        assert truth["robot_speed"] == "medium"
        # Never a Python dict repr.
        assert not truth["nav_offset_sink"].startswith("{")


class TestPredMatchesTruth:
    def test_color_dict_pred_vs_string_truth(self):
        assert _pred_matches_truth("plate_color_table", COLOR_DICT, COLOR_STR)
        wrong = dict(COLOR_DICT, v=100)
        assert not _pred_matches_truth("plate_color_table", wrong, COLOR_STR)

    def test_nav_dict_pred_vs_string_truth(self):
        assert _pred_matches_truth("nav_offset_fridge", NAV_DICT, NAV_STR)
        wrong = dict(NAV_DICT, dx=0.214)
        assert not _pred_matches_truth("nav_offset_fridge", wrong, NAV_STR)

    def test_text_normalizes_case_and_whitespace_only(self):
        assert _pred_matches_truth("bite_ordering", "  No Particular Order ", "no particular order")
        assert not _pred_matches_truth("bite_ordering", "chicken first", "no particular order")

    def test_categorical_exact(self):
        assert _pred_matches_truth("robot_speed", "slow", "slow")
        assert not _pred_matches_truth("robot_speed", "slow", "fast")


class TestCorrectedBlockRendering:
    def test_string_corrections_render_faithfully(self):
        """A pinned truth string must render as itself, not the factory default."""
        block = _format_corrected_block(
            {"plate_color_fridge": COLOR_STR, "nav_offset_dining_table": NAV_STR}
        )
        assert f"plate_color_fridge={COLOR_STR}" in block
        assert f"nav_offset_dining_table={NAV_STR}" in block
        assert format_color(DEFAULT_COLOR) not in block

    def test_dict_corrections_render_faithfully(self):
        block = _format_corrected_block({"plate_color_fridge": COLOR_DICT})
        assert f"plate_color_fridge={COLOR_STR}" in block


class TestPinningOverlayRoundTrip:
    """predict_bundle pins via parse_color/parse_nav_offset; formatted-string
    corrections must survive the round trip exactly."""

    def test_color_string_round_trip(self):
        assert parse_color(COLOR_STR) == COLOR_DICT
        assert format_color(parse_color(COLOR_STR)) == COLOR_STR

    def test_nav_string_round_trip(self):
        assert parse_nav_offset(NAV_STR) == pytest.approx(NAV_DICT) or parse_nav_offset(NAV_STR) == NAV_DICT
        assert format_nav_offset(parse_nav_offset(NAV_STR)) == NAV_STR

    def test_defaults_when_unparseable(self):
        assert parse_color(None) == DEFAULT_COLOR
        assert parse_nav_offset(None) == DEFAULT_NAV_OFFSET
