"""Tests for map_odom_clamp's pure SE2 + ClampCore logic (no ROS, no TF).

Run with:
    PYTHONPATH=src python -m pytest tests/test_map_odom_clamp.py -v

ClampCore takes the latest map_carto->odom and returns the map->map_carto
correction; net map->odom = compose(correction, carto). These lock: identity
start, pass-through when moving/disabled, jump absorption (held map->odom) while
parked, small-change pass-through, and release continuity.
"""

from __future__ import annotations

import math
import os
import sys
import types

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts")
)


def _stub(name, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


# map_odom_clamp imports rospy/tf2_ros/msg types at load; ClampCore uses none.
_stub("rospy")
_stub("tf2_ros")
_stub("geometry_msgs")
_stub("geometry_msgs.msg", Twist=object, TransformStamped=object)
_stub("std_msgs")
_stub("std_msgs.msg", Bool=object)

from map_odom_clamp import (  # noqa: E402
    ClampCore,
    se2_compose,
    se2_inverse,
    wrap,
    yaw_from_quat,
)

TOL = 1e-9
IDENT = (0.0, 0.0, 0.0)


def _close(a, b, tol=TOL):
    return (abs(a[0] - b[0]) < tol and abs(a[1] - b[1]) < tol
            and abs(wrap(a[2] - b[2])) < tol)


def _net(core, carto):
    """Net map->odom that consumers see = compose(correction, carto)."""
    return se2_compose(core.correction, carto)


# ---- SE2 helpers ----

def test_se2_inverse_roundtrip():
    a = (1.3, -0.7, 0.9)
    assert _close(se2_compose(a, se2_inverse(a)), IDENT)
    assert _close(se2_compose(se2_inverse(a), a), IDENT)
    assert _close(se2_inverse(se2_inverse(a)), a)


def test_se2_compose_with_rotation():
    # Translate (1,0) in a frame rotated +90 deg -> (0,1).
    r90 = (0.0, 0.0, math.pi / 2)
    assert _close(se2_compose(r90, (1.0, 0.0, 0.0)), (0.0, 1.0, math.pi / 2))


def test_yaw_from_quat():
    # +90 deg about z
    assert abs(yaw_from_quat(0.0, 0.0, math.sin(math.pi / 4),
                             math.cos(math.pi / 4)) - math.pi / 2) < 1e-9


# ---- ClampCore ----

def test_starts_identity_and_adopts_first_carto():
    core = ClampCore(park_freeze_s=5.0)
    assert core.correction == IDENT
    corr = core.update((2.0, 1.0, 0.3), 1.0)
    assert corr == IDENT  # first sample: adopt, no correction
    assert _close(_net(core, (2.0, 1.0, 0.3)), (2.0, 1.0, 0.3))


def test_flows_when_not_parked():
    """No command ever -> never parked -> Cartographer's changes pass through
    (identity correction), even a large one (this is startup localization)."""
    core = ClampCore(park_freeze_s=5.0)
    core.update((1.0, 0.0, 0.0), 1.0)
    core.update((5.0, 2.0, 0.5), 2.0)  # big change, but not parked
    assert core.correction == IDENT
    assert _close(_net(core, (5.0, 2.0, 0.5)), (5.0, 2.0, 0.5))


def test_absorbs_jump_while_parked():
    core = ClampCore(park_freeze_s=5.0, lin_jump_m=0.05, ang_jump_rad=0.03,
                     startup_grace_s=0.0)
    core.note_cmd(0.0)
    core.update((1.0, 0.0, 0.0), 1.0)          # adopt; net = (1,0,0)
    core.update((1.0, 0.0, 0.0), 2.0)          # not parked yet, no change
    assert _close(_net(core, (1.0, 0.0, 0.0)), (1.0, 0.0, 0.0))
    # Parked (t > 5) and Cartographer yanks map_carto->odom by 2 m:
    core.update((3.0, 0.0, 0.0), 6.0)
    assert core.last_absorbed is not None
    # Net map->odom must be HELD at the pre-jump value (1,0,0), not (3,0,0).
    assert _close(_net(core, (3.0, 0.0, 0.0)), (1.0, 0.0, 0.0))


def test_small_change_flows_while_parked():
    core = ClampCore(park_freeze_s=5.0, lin_jump_m=0.05, ang_jump_rad=0.03,
                     startup_grace_s=0.0)
    core.note_cmd(0.0)
    core.update((1.0, 0.0, 0.0), 1.0)
    core.update((1.02, 0.0, 0.0), 6.0)         # parked, 2 cm < 5 cm threshold
    assert core.last_absorbed is None
    assert core.correction == IDENT
    assert _close(_net(core, (1.02, 0.0, 0.0)), (1.02, 0.0, 0.0))


def test_release_is_continuous_then_follows():
    core = ClampCore(park_freeze_s=5.0, lin_jump_m=0.05, ang_jump_rad=0.03,
                     startup_grace_s=0.0)
    core.note_cmd(0.0)
    core.update((1.0, 0.0, 0.0), 1.0)
    core.update((3.0, 0.0, 0.0), 6.0)          # parked jump absorbed -> held at (1,0,0)
    assert _close(_net(core, (3.0, 0.0, 0.0)), (1.0, 0.0, 0.0))
    # Commanded motion again -> released. At the release instant (carto still
    # (3,0,0)) net is unchanged (continuous, no snap).
    core.note_cmd(10.0)
    core.update((3.0, 0.0, 0.0), 10.0)
    assert _close(_net(core, (3.0, 0.0, 0.0)), (1.0, 0.0, 0.0))
    # As Cartographer now moves with the base, net follows 1:1 from the held pose.
    core.update((3.5, 0.0, 0.0), 11.0)
    assert _close(_net(core, (3.5, 0.0, 0.0)), (1.5, 0.0, 0.0))


def test_startup_grace_keeps_clamp_transparent_during_warmup():
    """During the startup warmup the clamp must NOT absorb, even after a command
    and past park_freeze_s -- so Cartographer's initial global lock flows to the
    output. After the grace, parked jumps are absorbed as usual."""
    core = ClampCore(park_freeze_s=5.0, lin_jump_m=0.05, ang_jump_rad=0.03,
                     startup_grace_s=30.0)
    core.note_cmd(0.0)
    core.update((1.0, 0.0, 0.0), 1.0)          # start_t = 1.0
    # Parked (t-cmd > 5) but still inside the 30 s grace -> a big jump FLOWS.
    core.update((4.0, 0.0, 0.0), 20.0)
    assert core.last_absorbed is None
    assert core.correction == IDENT
    assert _close(_net(core, (4.0, 0.0, 0.0)), (4.0, 0.0, 0.0))
    # Past the grace (t - start_t > 30) and parked -> now a jump is absorbed.
    core.update((9.0, 0.0, 0.0), 40.0)
    assert core.last_absorbed is not None
    assert _close(_net(core, (9.0, 0.0, 0.0)), (4.0, 0.0, 0.0))  # held


def test_park_freeze_zero_disables_clamp():
    core = ClampCore(park_freeze_s=0.0)
    core.note_cmd(0.0)
    core.update((1.0, 0.0, 0.0), 1.0)
    core.update((9.0, 0.0, 0.0), 100.0)        # long idle, big jump -> still flows
    assert core.correction == IDENT
    assert _close(_net(core, (9.0, 0.0, 0.0)), (9.0, 0.0, 0.0))


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
