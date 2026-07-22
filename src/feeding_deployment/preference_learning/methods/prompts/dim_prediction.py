"""Prompt builder for per-dimension prediction (Axis 2).

Each call predicts ONE dimension and sees ONLY that dimension's history:
- system description rendered for just this dimension (no cross-dim leakage);
- the LTM summary sliced by field (the summary JSON is keyed by field, and
  ltm_update.txt constrains each entry to context/food/affect conditions, so
  slicing leaks nothing about other dims);
- retrieved episodes rendered as one line each -- context + THIS field's value
  only -- from the structured episode metas.

No corrected/confirmed blocks: pinned dims are never called, and showing other
dims' corrections is exactly the cross-dimension channel this arm removes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from feeding_deployment.preference_learning.config.physical_capabilities import (
    PHYSICAL_CAPABILITY_PROFILES,
)
from feeding_deployment.preference_learning.config.preference_bundle import PREFERENCE_BUNDLE
from feeding_deployment.preference_learning.data_generation.prompts.system_description import (
    get_system_description_prompt,
)
from feeding_deployment.preference_learning.methods.long_term_memory import _extract_json_object
from feeding_deployment.preference_learning.methods.utils import _format_pref_value

DIM_PREDICTION_PROMPT_PATH = Path(__file__).parent / "dim_prediction.txt"

_DIM_BY_FIELD = {dim.field: dim for dim in PREFERENCE_BUNDLE}
_PHYSICAL_CAPABILITY_BY_LABEL = {p.label: p for p in PHYSICAL_CAPABILITY_PROFILES}

# The no-dippables rule is context-only (meal contents are already in the
# prompt), so including it for its own dimension leaks nothing. The cross-dim
# transfer_mode rule is deliberately absent here -- it is applied post-hoc in
# _predict_bundle_per_dim, the one sanctioned cross-dim touch point.
_DIM_LOCAL_HARD_RULES: Dict[str, str] = {
    "bite_dipping_preference": (
        "\nHARD RULE (must always be satisfied): if the meal contents above "
        'indicate no dippable items or no sauces, the value MUST be "do not dip".\n'
    ),
}


def _slice_ltm(ltm_summary: str, field: str) -> str:
    """This field's entry from the field-keyed LTM summary JSON; empty-block
    markers when the summary is missing, unparseable, or lacks the field."""
    if not ltm_summary or not ltm_summary.strip() or ltm_summary.strip() == "N/A":
        return "(empty)"
    try:
        data = json.loads(_extract_json_object(ltm_summary))
    except Exception:
        return "(empty)"
    entry = data.get(field) if isinstance(data, dict) else None
    if entry is None:
        return "(no entry for this dimension yet)"
    return json.dumps({field: entry}, ensure_ascii=False, indent=2)


def _episode_lines(records: List[Dict[str, Any]], field: str) -> str:
    """One line per retrieved episode: context + THIS field's value. Episodes
    without structured metas (pre-change logs) are skipped -- rendering their
    flat episode text would leak every other dimension."""
    lines: List[str] = []
    for rec in records:
        meta = rec.get("meta")
        if not isinstance(meta, dict) or not isinstance(meta.get("prefs"), dict):
            continue
        ctx = meta.get("context", {}) or {}
        value = _format_pref_value(field, meta["prefs"].get(field, ""))
        line = (
            f"day={meta.get('day')}; meal={ctx.get('meal')}; setting={ctx.get('setting')}; "
            f"time_of_day={ctx.get('time_of_day')}; {field}={value}"
        )
        if field in (meta.get("corrected_fields") or []):
            line += " (user actively corrected this that day)"
        lines.append(line)
    return "\n".join(lines) if lines else "(empty)"


def get_dim_prediction_prompt(
    field: str,
    physical_profile_label: str,
    ltm_summary: str,
    records: List[Dict[str, Any]],
    context: dict,
    option_line: str,
    meal_contents: str = "(not provided)",
    *,
    physical_profile_description: Optional[str] = None,
) -> str:
    if field not in _DIM_BY_FIELD:
        raise ValueError(f"Unknown preference field: {field!r}")
    template = DIM_PREDICTION_PROMPT_PATH.read_text(encoding="utf-8")

    system_description = get_system_description_prompt(dims=[_DIM_BY_FIELD[field]])

    if physical_profile_description is not None:
        desc = physical_profile_description.strip()
        if not desc:
            raise ValueError("physical_profile_description is empty")
    else:
        if physical_profile_label not in _PHYSICAL_CAPABILITY_BY_LABEL:
            valid = ", ".join(sorted(_PHYSICAL_CAPABILITY_BY_LABEL.keys()))
            raise ValueError(f"Unknown physical_profile_label={physical_profile_label!r}. Valid: {valid}")
        desc = _PHYSICAL_CAPABILITY_BY_LABEL[physical_profile_label].description

    return template.format(
        system_description=system_description,
        physical_profile=desc,
        field=field,
        ltm_field_block=_slice_ltm(ltm_summary, field),
        episodic_field_block=_episode_lines(records, field),
        meal=context.get("meal"),
        setting=context.get("setting"),
        time_of_day=context.get("time_of_day"),
        meal_contents=meal_contents,
        option_line=option_line,
        hard_rule_block=_DIM_LOCAL_HARD_RULES.get(field, ""),
    )
