"""Programmatic ground truth for the continuous preference dims (3 plate
colors + 4 nav offsets).

Unlike the categorical dims (LLM-generated against an option list that
quantizes away sampling noise), a continuous HSV/offset value has to be exactly
reproducible across 30 independent days for a stable f(x)=y to exist -- the
eval compares formatted strings digit-for-digit. So these 7 dims are generated
by a seeded pure function, no LLM anywhere.

Structure (the correlated-bundle hypothesis, instantiated):

    truth_color(loc, tod)          = base_color + location_offset[loc] + time_shift[tod]
    truth_nav(loc, tod, affect)    = base_nav[loc] + time_shift[tod] + affect_shift[affect]

- The color time shift is SHARED across the 3 pickup locations (same physical
  handle, one lighting change): correcting one plate color reveals the shift
  the other two need. No affect term -- lighting is not mood-dependent.
- The nav time and affect shifts are SHARED across the 4 locations, on top of
  independent per-location bases (matching the bundle docs: locations are
  independent, residuals are not).
- The affect shift is driven by the transient affective state Y_t, which is
  HIDDEN from the predictor (never in the prompt or episode text). It can only
  be inferred within a meal from corrections on other dims that share it --
  the persistent joint-vs-per-dim signal.

All components are sampled once per user (ints for HSV, 3-decimal floats for
offsets) with ranges chosen so sums never reach the parse_color /
parse_nav_offset clip bounds; clipping would collapse distinct contexts onto
one value and break the additive structure (continuous_truth asserts this).
Minimum shift magnitudes keep every factor distinguishable at the integer /
3-decimal resolution the eval compares at.
"""

from __future__ import annotations

import random
from typing import Any, Dict

from feeding_deployment.preference_learning.config.affective_state import AFFECTIVE_STATES
from feeding_deployment.preference_learning.config.mealtime_context import TIMES_OF_DAY
from feeding_deployment.preference_learning.config.preference_bundle import (
    COLOR_FIELD_BY_LOCATION,
    OFFSET_FIELD_BY_LOCATION,
    format_color,
    format_nav_offset,
    parse_color,
    parse_nav_offset,
)

CONTINUOUS_FIELDS = sorted(
    list(COLOR_FIELD_BY_LOCATION.values()) + list(OFFSET_FIELD_BY_LOCATION.values())
)

_NEUTRAL_STATE = "Neutral"
# The zero-shift baseline time-of-day. Must be a member of TIMES_OF_DAY;
# "afternoon" is the midday, neutral-lighting reference.
REFERENCE_TOD = "afternoon"

_ZERO_COLOR_SHIFT = {"dh": 0, "ds": 0, "dv": 0}
_ZERO_NAV_SHIFT = {"dx": 0.0, "dy": 0.0, "dyaw": 0.0}


def _signed(rng: random.Random, lo: float, hi: float) -> float:
    """Magnitude in [lo, hi] (3 decimals) with a random sign."""
    return rng.choice((-1, 1)) * round(rng.uniform(lo, hi), 3)


def sample_continuous_tables(rng: random.Random) -> Dict[str, Any]:
    """Sample one user's frozen component tables. JSON-serializable; stored in
    the encoding payload under "continuous_tables" and replayed verbatim by
    the dataset generator.

    Sums stay strictly inside the clip bounds (h in [0,179], s/v in [0,255]) by
    construction, over all four times of day:
      h: base [10,165] + microwave [4,8] + evening [2,5] <= 178;  min 10-3-5 = 2
      s: base [150,235] + evening [-40,-20]              -> [110, 235]
      v: base [120,195] - fridge 50 - night 45 >= 25;    + morning 25 + mw 10 <= 230
      nav dx/dy: 0.20 + 0.12 + 0.10 = 0.42 < 0.5;  dyaw: 0.30 + 0.15 + 0.15 = 0.60 < 0.785
    (Base hue is [10,165], not the plan's [10,169] -- 169+8+5 = 182 would clip.)
    """
    base_color = {
        "h": rng.randint(10, 165),
        "s": rng.randint(150, 235),
        "v": rng.randint(120, 195),
        "range": rng.choice((0.05, 0.10, 0.15)),
    }
    location_color_offsets = {
        "table": dict(_ZERO_COLOR_SHIFT),  # reference location
        "fridge": {"dh": rng.randint(-3, 3), "ds": 0, "dv": rng.randint(-50, -30)},
        "microwave": {"dh": rng.randint(4, 8), "ds": 0, "dv": rng.randint(-10, 10)},
    }
    time_color_shifts = {
        "afternoon": dict(_ZERO_COLOR_SHIFT),  # reference time (midday, neutral light)
        "morning": {"dh": rng.randint(-5, -2), "ds": 0, "dv": rng.randint(10, 25)},
        "evening": {"dh": rng.randint(2, 5), "ds": rng.randint(-40, -20), "dv": rng.randint(-35, -15)},
        "night": {"dh": rng.randint(-5, -2), "ds": rng.randint(-30, -10), "dv": rng.randint(-45, -25)},
    }

    base_nav = {
        loc: {
            "dx": round(rng.uniform(-0.20, 0.20), 3),
            "dy": round(rng.uniform(-0.20, 0.20), 3),
            "dyaw": round(rng.uniform(-0.30, 0.30), 3),
        }
        for loc in OFFSET_FIELD_BY_LOCATION
    }
    time_nav_shifts = {
        tod: (
            dict(_ZERO_NAV_SHIFT)
            if tod == REFERENCE_TOD
            else {
                "dx": _signed(rng, 0.06, 0.12),
                "dy": _signed(rng, 0.06, 0.12),
                "dyaw": _signed(rng, 0.08, 0.15),
            }
        )
        for tod in TIMES_OF_DAY
    }
    affect_nav_shifts = {
        state: (
            dict(_ZERO_NAV_SHIFT)
            if state == _NEUTRAL_STATE
            else {
                "dx": _signed(rng, 0.05, 0.10),
                "dy": _signed(rng, 0.05, 0.10),
                "dyaw": _signed(rng, 0.08, 0.15),
            }
        )
        for state in AFFECTIVE_STATES
    }

    # Guard against time-of-day vocabulary drift: every shift table must cover
    # exactly TIMES_OF_DAY, and the zero-shift baseline must be one of them.
    # (continuous_truth looks up an arbitrary tod in both tables at runtime.)
    assert REFERENCE_TOD in TIMES_OF_DAY, (REFERENCE_TOD, TIMES_OF_DAY)
    assert set(time_color_shifts) == set(TIMES_OF_DAY), set(time_color_shifts)
    assert set(time_nav_shifts) == set(TIMES_OF_DAY), set(time_nav_shifts)

    return {
        "base_color": base_color,
        "location_color_offsets": location_color_offsets,
        "time_color_shifts": time_color_shifts,
        "base_nav": base_nav,
        "time_nav_shifts": time_nav_shifts,
        "affect_nav_shifts": affect_nav_shifts,
    }


def continuous_truth(
    tables: Dict[str, Any], time_of_day: str, affective_state: str
) -> Dict[str, Dict[str, Any]]:
    """The 7 canonical continuous values for one day's context. Deterministic:
    the same (time_of_day, affective_state) always yields the same exact
    values, so they repeat verbatim in episodic memory and stay learnable."""
    time_color = tables["time_color_shifts"][time_of_day]
    time_nav = tables["time_nav_shifts"][time_of_day]
    affect_nav = tables["affect_nav_shifts"][affective_state]
    base_c = tables["base_color"]

    out: Dict[str, Dict[str, Any]] = {}

    for loc, field in COLOR_FIELD_BY_LOCATION.items():
        loc_off = tables["location_color_offsets"][loc]
        raw = {
            "h": int(base_c["h"]) + int(loc_off["dh"]) + int(time_color["dh"]),
            "s": int(base_c["s"]) + int(loc_off["ds"]) + int(time_color["ds"]),
            "v": int(base_c["v"]) + int(loc_off["dv"]) + int(time_color["dv"]),
            "range": float(base_c["range"]),
        }
        val = parse_color(raw)
        assert val == raw, f"{field} clipped: {raw} -> {val} (sampling ranges must keep sums interior)"
        out[field] = val

    for loc, field in OFFSET_FIELD_BY_LOCATION.items():
        base_n = tables["base_nav"][loc]
        raw = {
            k: round(float(base_n[k]) + float(time_nav[k]) + float(affect_nav[k]), 3)
            for k in ("dx", "dy", "dyaw")
        }
        val = parse_nav_offset(raw)
        assert val == raw, f"{field} clipped: {raw} -> {val} (sampling ranges must keep sums interior)"
        out[field] = val

    return out


def render_continuous_tendencies(tables: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    """Human-readable mirror of the tables in the standard encoding shape
    ({default, user_tendencies} per field), so the stored encoding stays
    complete and debuggable. The predictor never sees this -- ground truth
    comes from continuous_truth."""
    out: Dict[str, Dict[str, str]] = {}

    neutral = _NEUTRAL_STATE
    for loc, field in COLOR_FIELD_BY_LOCATION.items():
        rules = []
        for tod in TIMES_OF_DAY:
            val = continuous_truth(tables, tod, neutral)[field]
            because = (
                "reference lighting" if tod == REFERENCE_TOD else f"{tod} lighting shifts the handle's apparent color"
            )
            rules.append(f"IF time_of_day={tod} THEN {format_color(val)} BECAUSE {because}.")
        rules.append(
            "The time-of-day shift is shared across the fridge/microwave/table pickups "
            "(same physical handle, one lighting change)."
        )
        out[field] = {
            "default": format_color(continuous_truth(tables, REFERENCE_TOD, neutral)[field]),
            "user_tendencies": " ".join(rules),
        }

    for loc, field in OFFSET_FIELD_BY_LOCATION.items():
        rules = []
        for tod in TIMES_OF_DAY:
            for state in AFFECTIVE_STATES:
                val = continuous_truth(tables, tod, state)[field]
                rules.append(
                    f"IF time_of_day={tod} AND affective_state={state} THEN {format_nav_offset(val)}."
                )
        rules.append(
            "The time-of-day and affective-state components are shared across the "
            "table/microwave/sink/fridge offsets (household routine and posture, not the location)."
        )
        out[field] = {
            "default": format_nav_offset(continuous_truth(tables, REFERENCE_TOD, neutral)[field]),
            "user_tendencies": " ".join(rules),
        }

    return out
