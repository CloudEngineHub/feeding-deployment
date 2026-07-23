"""Tests for scan_gate's pure GateCore logic (no ROS, no lidar).

Run with:
    PYTHONPATH=src python -m pytest tests/test_scan_gate.py -v

GateCore is the replay-testable core of scripts/scan_gate.py: feed it per-lidar
coverage + timestamps (and nonzero base commands via note_cmd) and read .open.
These lock the two freeze triggers -- occlusion (coverage < min_cells) and the
parked timer (no cmd_vel for park_freeze_s) -- and the reopen hysteresis.
"""

from __future__ import annotations

import os
import sys
import types

# scan_gate.py lives in scripts/, not the installed package.
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts")
)


# scan_gate imports rospy + the ROS message types at module load, none of which
# GateCore (the pure logic under test) touches. Stub them so the test runs in a
# plain Python env without a sourced ROS workspace.
def _stub(name, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


_stub("rospy")
_stub("geometry_msgs")
_stub("geometry_msgs.msg", Twist=object)
_stub("sensor_msgs")
_stub("sensor_msgs.msg", LaserScan=object)
_stub("std_msgs")
_stub("std_msgs.msg", Bool=object)

from scan_gate import GateCore  # noqa: E402

HEALTHY = 50  # cells, comfortably above the default min_cells=25
OCCLUDED = 10  # feeding-occlusion regime


def _feed(core, t, cov_l=HEALTHY, cov_r=HEALTHY):
    """Deliver one scan from each lidar at time t; return whether the gate is
    open afterwards. `_feed.last_transition` holds any flip during the tick
    (the node consumes .transition per single update; feeding two here would
    otherwise clobber a flip that fired on the first)."""
    core.update("l", cov_l, t)
    tr = core.transition
    is_open = core.update("r", cov_r, t)
    _feed.last_transition = core.transition or tr
    return is_open


_feed.last_transition = None


def _drive_open(core, t0=0.0, dt=0.5, ticks=8):
    """Feed healthy scans (with a live command each tick) until the gate opens."""
    t = t0
    for _ in range(ticks):
        core.note_cmd(t)
        if _feed(core, t):
            return t
        t += dt
    return None


def test_opens_when_healthy_and_commanded():
    core = GateCore(reopen_after_s=2.0, park_freeze_s=5.0)
    assert not core.open  # starts closed
    opened_at = _drive_open(core)
    assert core.open
    # reopen hysteresis honored: not before reopen_after_s of continuous health.
    assert opened_at is not None and opened_at >= 2.0


def test_startup_opens_without_any_command():
    """park_freeze must stay disabled until the first command, so a freshly
    launched robot still locks onto the loaded map on coverage alone."""
    core = GateCore(reopen_after_s=2.0, park_freeze_s=5.0)
    t = 0.0
    for _ in range(8):
        # note: never call note_cmd -> last_cmd_t stays None -> never "parked"
        if _feed(core, t):
            break
        t += 0.5
    assert core.open


def test_parked_freezes_even_with_clear_view():
<<<<<<< HEAD
    core = GateCore(reopen_after_s=2.0, park_freeze_s=5.0)
=======
    core = GateCore(reopen_after_s=2.0, park_freeze_s=5.0, startup_grace_s=0.0)
>>>>>>> carto-park-freeze
    _drive_open(core)
    assert core.open
    # Base stops being commanded; keep feeding healthy scans. Once we cross
    # park_freeze_s past the last command, the gate must close.
    last_cmd_t = core.last_cmd_t
    t = last_cmd_t
    closed = False
    for _ in range(20):
        t += 0.5
        if not _feed(core, t):
            closed = True
            break
    assert closed
    assert (t - last_cmd_t) > 5.0
    assert not core.open
    # transition reason names the parked trigger, not coverage.
    assert _feed.last_transition is not None
    assert "parked" in _feed.last_transition[1]


def test_parked_reopens_when_commanded_again():
    core = GateCore(reopen_after_s=2.0, park_freeze_s=5.0, startup_grace_s=0.0)
    _drive_open(core)
    # let it park-freeze
    t = core.last_cmd_t + 6.0
    _feed(core, t)
    assert not core.open
    # command motion again -> reopens after the hysteresis
    reopened = _drive_open(core, t0=t + 0.5)
    assert core.open and reopened is not None


def test_startup_grace_holds_open_through_initial_localization():
    """The parked freeze must NOT engage during the startup warmup, even after
    an early command -- Cartographer's initial global lock can take far longer
    than park_freeze_s, and freezing then would starve it."""
    core = GateCore(reopen_after_s=2.0, park_freeze_s=5.0, startup_grace_s=30.0)
    _drive_open(core)                 # localizes; last cmd ~t=2, start_t=0
    assert core.open
    # No more commands. Feed healthy well past park_freeze_s but inside the
    # 30 s grace -> must stay OPEN.
    t = core.last_cmd_t
    while t < core.start_t + 25.0:
        t += 0.5
        _feed(core, t)
    assert core.open, "startup grace should hold the gate open during warmup"
    # Past the grace and still parked -> now it freezes.
    while t < core.start_t + 33.0:
        t += 0.5
        _feed(core, t)
    assert not core.open


def test_park_freeze_disabled_when_zero():
    """park_freeze_s=0 falls back to coverage-only gating: a parked-but-clear
    base stays open indefinitely."""
    core = GateCore(reopen_after_s=2.0, park_freeze_s=0.0)
    _drive_open(core)
    assert core.open
    # No further commands, long idle, healthy view -> stays open.
    t = core.last_cmd_t
    for _ in range(40):
        t += 0.5
        _feed(core, t)
    assert core.open


def test_occlusion_still_closes_regardless_of_commands():
    """Regression: the original coverage trigger must still fire, even while
    the base is actively being commanded (occluded mid-approach)."""
    core = GateCore(reopen_after_s=2.0, park_freeze_s=5.0)
    _drive_open(core)
    assert core.open
    t = core.last_cmd_t
    core.note_cmd(t)  # base still moving
    assert not _feed(core, t, cov_l=OCCLUDED, cov_r=HEALTHY)
    assert _feed.last_transition is not None
    assert "coverage" in _feed.last_transition[1]


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
