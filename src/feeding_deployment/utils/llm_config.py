"""Shared LLM configuration."""

import os

DEFAULT_CLAUDE_MODEL = "claude-opus-4-8"

# Synthetic data generation (user preference encodings + deployment datasets)
# runs on Opus: the encoding must hold a coherent personality with consistent
# tie-break policies across ~20 dims, and each day's bundle must apply those
# rules faithfully -- rule-application slips in the ground truth become noise
# that every eval arm pays for. Generation is a one-off cost (~155 calls per
# regen), not a runtime cost, so quality wins over price here. The robot
# runtime keeps DEFAULT_CLAUDE_MODEL.
DATA_GENERATION_CLAUDE_MODEL = "claude-opus-4-8"

# Preference learning (bundle prediction + LTM updates) runs on Opus: these run
# once per user correction / once per meal, and Haiku reasons correctly about
# cross-dimension correlations but hedges on committing to the implied values
# (see docs/preference_learning.md and scripts/replay_pref_stresstest.py).
# Everything else (FLAIR planning, transparency, adaptability) stays on
# DEFAULT_CLAUDE_MODEL.
#
# Effort "medium": a 2026-07-03 sweep of low/medium/high/xhigh over the recorded
# 3-day stress test (scratchpad effort_sweep; see docs §2.5) found prediction
# quality FLAT across levels -- corrections/day 12/11/12/13, identical day-3
# convergence, full explanation coverage, equivalent color propagation by the
# next pickup -- while mean call latency scaled 23s/30s/38s/68s. Medium halves
# every user-visible wait vs the old xhigh at no measured quality cost. If a
# future scenario shows under-thinking (propagation misses, shallow latent
# inference), raise to "high" before "xhigh".
PREDICTION_CLAUDE_MODEL = "claude-opus-4-8"

# Default "medium" (see the sweep rationale above). Overridable per-process via
# the PREDICTION_EFFORT env var so an offline effort sweep can time low/high
# without editing this file (and without changing the deployment default that
# run.py reads from the same constant). The robot deployment leaves the env var
# unset and therefore always runs at "medium".
_ALLOWED_EFFORT = {"low", "medium", "high", "xhigh", "max"}
PREDICTION_EFFORT = os.environ.get("PREDICTION_EFFORT", "medium").strip().lower()
if PREDICTION_EFFORT not in _ALLOWED_EFFORT:
    raise ValueError(
        f"PREDICTION_EFFORT must be one of {sorted(_ALLOWED_EFFORT)}; "
        f"got {os.environ.get('PREDICTION_EFFORT')!r}"
    )

# Serve the prediction call with fast mode (research preview: same Opus 4.8
# weights at up to 2.5x output tokens/sec, at 2x price -- $10/$50 per MTok vs
# $5/$25). Access is gated (account manager / claude.com/fast-mode waitlist);
# without it the fast request fails and PredictionModel falls back to standard
# speed automatically, so this is safe to leave on before access is granted.
# The meal-end LTM update always runs at standard speed (nobody waits on it).
PREDICTION_FAST_MODE = True
