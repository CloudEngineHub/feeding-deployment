"""Tests for mid-skill takeover during bite transfer.

Run with:
    PYTHONPATH=src python -m pytest tests/test_transfer_takeover.py -v
or directly:
    PYTHONPATH=src python tests/test_transfer_takeover.py

Background. The feel_the_bite transfer controllers command the arm directly
through ``execute_command`` (via ``move_to_ee_pose``), bypassing the takeover
polling in ``HighLevelAction.execute_robot_command``. A takeover pressed during
the two transfer moves -- the approach to the mouth, and the retract afterwards
-- was therefore never surfaced: the skill ran to completion and the executive
fell through to its idle handler, where there is no skill left to redo, so
"Redo Skill" silently became "next". Observed on day 17 (2026-08-10, 21:22:31),
where the press landed during the retract and the redo turned into the next
skill.

These tests are deliberately ROS-free and robot-free: they drive the proxy and
the filter-flag lifecycle through stubs, so they can run on the deployment
machine without touching the live roscore or publishing to real topics.
"""

from __future__ import annotations

import sys
import types
import unittest


# --- Stub out the ROS + heavy deps that base.py pulls in ---------------------
# We only exercise TakeoverAwareArmInterface and the filter-flag lifecycle, so
# importing the real modules (rospy, pybullet, torch, ...) is neither possible
# nor useful here. Only what the code under test actually touches is stubbed.

class _StubBool:
    def __init__(self, data=False):
        self.data = data


class WebInterfaceTakeoverInterrupt(Exception):
    """Mirror of the real exception (importing the real one drags in rospy)."""


class _FakeEvent:
    def __init__(self, value=False):
        self._value = value

    def set(self):
        self._value = True

    def clear(self):
        self._value = False

    def is_set(self):
        return self._value


class _FakeWebInterface:
    def __init__(self):
        self.takeover_event = _FakeEvent()


class _FakeArm:
    """Records commands and never fails, so a raise can only come from the proxy."""

    def __init__(self):
        self.commands = []
        self.speed = "high"

    def execute_command(self, command):
        self.commands.append(command)
        return True

    def set_speed(self, speed):
        self.speed = speed

    def switch_out_of_compliant_mode(self):
        return "switched"


# The proxy under test, transcribed from feeding_deployment.actions.base. Kept
# in sync by test_proxy_matches_source below, which diffs it against the real
# definition so this copy cannot silently drift.
class TakeoverAwareArmInterface:
    def __init__(self, inner, web_interface):
        self._inner = inner
        self._web_interface = web_interface

    def _raise_if_takeover(self):
        web = self._web_interface
        if web is not None and web.takeover_event.is_set():
            raise WebInterfaceTakeoverInterrupt()

    def execute_command(self, *args, **kwargs):
        self._raise_if_takeover()
        result = self._inner.execute_command(*args, **kwargs)
        self._raise_if_takeover()
        return result

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _takeover_aware_arm_interface(robot_interface, web_interface, enabled=True):
    if enabled and robot_interface is not None and web_interface is not None:
        return TakeoverAwareArmInterface(robot_interface, web_interface)
    return robot_interface


class _FakeTransferController:
    """Stands in for OutsideMouthTransfer, mirroring the real control flow of
    move_to_transfer_state: raise the sticky filter flag, move, lower it in a
    finally. move_to_before_transfer_state is the bare move (the day-17 case)."""

    def __init__(self, robot_interface, filter_log):
        self.robot_interface = robot_interface
        self.filter_log = filter_log

    def _publish_filter(self, value):
        if self.robot_interface is not None:
            self.filter_log.append(value)

    def move_to_ee_pose(self, pose):
        if self.robot_interface is None:
            return
        self.robot_interface.execute_command(pose)

    def move_to_transfer_state(self, pose="approach"):
        self._publish_filter(True)
        try:
            self.move_to_ee_pose(pose)
        finally:
            self._publish_filter(False)

    def move_to_before_transfer_state(self, pose="retract"):
        self.move_to_ee_pose(pose)


class _FakeGestureDetector:
    """Mirrors the real detectors' loop contract (mouth_open line 424,
    poll_until_detected line 365): poll until detected, until the caller's
    termination_event is set, or until the timeout. Returns whether it fired."""

    def __init__(self, fires_after=None):
        self.fires_after = fires_after   # poll count at which the gesture fires
        self.polls = 0

    def __call__(self, perception_interface, termination_event, timeout, **kw):
        deadline = timeout
        while self.polls < deadline and (
            termination_event is None or not termination_event.is_set()
        ):
            self.polls += 1
            if self.fires_after is not None and self.polls >= self.fires_after:
                return True
        return False


class _FakeHLA:
    """The two helpers added to HighLevelAction, transcribed."""

    def __init__(self, web_interface):
        self.web_interface = web_interface

    def takeover_termination_event(self):
        return self.web_interface.takeover_event if self.web_interface is not None else None

    def raise_if_takeover(self):
        if self.web_interface is not None and self.web_interface.takeover_event.is_set():
            raise WebInterfaceTakeoverInterrupt()


class GestureWaitTest(unittest.TestCase):
    """A takeover must end a gesture wait. Before this, mouth_open/head_nod were
    called with termination_event=None, so the executive sat in the wait for up
    to 10 minutes with no teleop session running -- the arm-control screen was
    inert until the user performed the very gesture they were interrupting."""

    def setUp(self):
        self.web = _FakeWebInterface()
        self.hla = _FakeHLA(self.web)

    def _wait(self, detector):
        detector(None, self.hla.takeover_termination_event(), 600)
        self.hla.raise_if_takeover()

    def test_gesture_detected_proceeds(self):
        detector = _FakeGestureDetector(fires_after=3)
        self._wait(detector)          # no raise
        self.assertEqual(detector.polls, 3)

    def test_takeover_ends_the_wait_and_raises(self):
        detector = _FakeGestureDetector(fires_after=None)  # user never signals
        self.web.takeover_event.set()
        with self.assertRaises(WebInterfaceTakeoverInterrupt):
            self._wait(detector)
        self.assertEqual(detector.polls, 0, "wait should end immediately, not run to timeout")

    def test_takeover_midway_ends_the_wait(self):
        detector = _FakeGestureDetector(fires_after=None)
        web = self.web

        class _Detector(_FakeGestureDetector):
            def __call__(inner, pi, ev, timeout, **kw):
                # the press lands after a few polls
                for _ in range(5):
                    inner.polls += 1
                web.takeover_event.set()
                return super().__call__(pi, ev, timeout, **kw)

        with self.assertRaises(WebInterfaceTakeoverInterrupt):
            self._wait(_Detector(fires_after=None))

    def test_timeout_does_not_raise(self):
        # The Fix-1/Fix-3 boundary, pinned deliberately: a plain timeout is NOT a
        # takeover. What should happen when the user never signals at all is a
        # separate question; this change must not silently start aborting there.
        detector = _FakeGestureDetector(fires_after=None)
        self._wait(detector)          # runs to timeout, no raise
        self.assertEqual(detector.polls, 600)

    def test_sim_path_unchanged(self):
        hla = _FakeHLA(None)
        self.assertIsNone(hla.takeover_termination_event())
        hla.raise_if_takeover()       # no web interface -> never raises


class ForceTriggerTest(unittest.TestCase):
    """detect_force_trigger is the fork-at-the-mouth wait. It has no timeout at
    all, so without a termination event a takeover could not end it: the user had
    to bite the fork to get control back."""

    @staticmethod
    def _force_trigger(exceeded_after, termination_event=None, simulation=False,
                       max_polls=1000):
        """The loop from perception_interface.detect_force_trigger."""
        if simulation:
            return True, 0
        polls = 0
        exceeded = False
        while (polls < max_polls and not exceeded
               and (termination_event is None or not termination_event.is_set())):
            polls += 1
            if exceeded_after is not None and polls >= exceeded_after:
                exceeded = True
        return exceeded, polls

    def test_bite_detected_returns_true(self):
        triggered, polls = self._force_trigger(exceeded_after=10)
        self.assertTrue(triggered)
        self.assertEqual(polls, 10)

    def test_takeover_ends_the_unbounded_wait(self):
        web = _FakeWebInterface()
        web.takeover_event.set()
        triggered, polls = self._force_trigger(
            exceeded_after=None, termination_event=web.takeover_event)
        self.assertFalse(triggered, "must report it did not fire, not a false bite")
        self.assertEqual(polls, 0)

    def test_without_event_it_still_blocks(self):
        # Documents why the event is needed: no termination_event -> runs forever
        # (bounded here by max_polls only).
        triggered, polls = self._force_trigger(exceeded_after=None, max_polls=500)
        self.assertFalse(triggered)
        self.assertEqual(polls, 500)

    def test_simulation_short_circuits(self):
        triggered, _ = self._force_trigger(exceeded_after=None, simulation=True)
        self.assertTrue(triggered)


class WaitCoverageTest(unittest.TestCase):
    """Every blocking user-wait in transfer_tool must pass a real termination
    event and immediately surface a takeover. This is the guard that matters:
    the failure mode is someone adding a sixth wait and forgetting."""

    GUARDED = {"mouth_open", "head_nod", "gesture_fn", "detect_force_trigger"}

    def _wait_calls(self):
        import ast
        import pathlib

        src = pathlib.Path(__file__).resolve().parents[1] / (
            "src/feeding_deployment/actions/transfer_tool.py"
        )
        tree = ast.parse(src.read_text())
        found = []
        for node in ast.walk(tree):
            body = getattr(node, "body", None)
            if not isinstance(body, list):
                continue
            for i, stmt in enumerate(body):
                if not isinstance(stmt, ast.Expr) or not isinstance(stmt.value, ast.Call):
                    continue
                call = stmt.value
                func = call.func
                name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
                if name not in self.GUARDED:
                    continue
                nxt = body[i + 1] if i + 1 < len(body) else None
                found.append((name, call, nxt))
        return found

    def test_every_wait_is_guarded(self):
        import ast

        calls = self._wait_calls()
        self.assertGreaterEqual(len(calls), 6, f"expected the known waits, found {len(calls)}")
        for name, call, nxt in calls:
            kwargs = {k.arg: k.value for k in call.keywords}
            self.assertIn("termination_event", kwargs,
                          f"{name}() must pass termination_event")
            value = kwargs["termination_event"]
            self.assertFalse(
                isinstance(value, ast.Constant) and value.value is None,
                f"{name}() passes termination_event=None -- takeover cannot end this wait",
            )
            self.assertTrue(
                isinstance(nxt, ast.Expr) and isinstance(nxt.value, ast.Call)
                and getattr(nxt.value.func, "attr", "") == "raise_if_takeover",
                f"{name}() must be followed immediately by self.raise_if_takeover()",
            )

    def test_force_trigger_loop_honours_the_event(self):
        import ast
        import pathlib

        src = pathlib.Path(__file__).resolve().parents[1] / (
            "src/feeding_deployment/interfaces/perception_interface.py"
        )
        fn = next(
            (n for n in ast.walk(ast.parse(src.read_text()))
             if isinstance(n, ast.FunctionDef) and n.name == "detect_force_trigger"),
            None,
        )
        self.assertIsNotNone(fn, "detect_force_trigger not found")
        self.assertIn("termination_event", [a.arg for a in fn.args.args],
                      "detect_force_trigger must accept termination_event")
        loops = [n for n in ast.walk(fn) if isinstance(n, ast.While)]
        self.assertTrue(loops, "expected a wait loop")
        self.assertTrue(
            any("termination_event" in ast.unparse(loop.test) for loop in loops),
            "the wait loop must test termination_event, or a takeover cannot end it",
        )


class TakeoverProxyTest(unittest.TestCase):
    def setUp(self):
        self.arm = _FakeArm()
        self.web = _FakeWebInterface()
        self.proxy = _takeover_aware_arm_interface(self.arm, self.web)

    def test_passes_through_when_no_takeover(self):
        self.assertTrue(self.proxy.execute_command("pose"))
        self.assertEqual(self.arm.commands, ["pose"])

    def test_raises_before_the_move_when_already_latched(self):
        self.web.takeover_event.set()
        with self.assertRaises(WebInterfaceTakeoverInterrupt):
            self.proxy.execute_command("pose")
        # Nothing was commanded: the press beat the move.
        self.assertEqual(self.arm.commands, [])

    def test_raises_after_a_move_interrupted_mid_flight(self):
        # stop_action aborts the in-flight move and the flag is still latched
        # when execute_command returns -- the day-17 sequence.
        class _ArmThatGetsStopped(_FakeArm):
            def execute_command(inner_self, command):
                super().execute_command(command)
                self.web.takeover_event.set()
                return True

        arm = _ArmThatGetsStopped()
        proxy = _takeover_aware_arm_interface(arm, self.web)
        with self.assertRaises(WebInterfaceTakeoverInterrupt):
            proxy.execute_command("pose")
        self.assertEqual(arm.commands, ["pose"])

    def test_only_peeks_never_consumes(self):
        # execute_action owns the consume; the proxy must leave the flag set.
        self.web.takeover_event.set()
        with self.assertRaises(WebInterfaceTakeoverInterrupt):
            self.proxy.execute_command("pose")
        self.assertTrue(self.web.takeover_event.is_set())

    def test_delegates_everything_else(self):
        self.assertIsNotNone(self.proxy)
        self.assertTrue(bool(self.proxy))
        self.assertEqual(self.proxy.switch_out_of_compliant_mode(), "switched")
        self.assertEqual(self.proxy.speed, "high")
        self.proxy.set_speed("low")
        self.assertEqual(self.arm.speed, "low")

    def test_falls_back_to_raw_interface_where_takeover_cannot_apply(self):
        # sim (no arm), no web interface, and feature disabled: unchanged paths.
        self.assertIsNone(_takeover_aware_arm_interface(None, self.web))
        self.assertIs(_takeover_aware_arm_interface(self.arm, None), self.arm)
        self.assertIs(
            _takeover_aware_arm_interface(self.arm, self.web, enabled=False), self.arm
        )


class FilterFlagLifecycleTest(unittest.TestCase):
    """The noisy-readings filter is sticky on the head-perception node. If a
    takeover raises out of the approach move and the flag is left on, every
    later transfer can block in the (deliberately unbounded)
    wait_for_head_perception_data."""

    def setUp(self):
        self.arm = _FakeArm()
        self.web = _FakeWebInterface()
        self.filter_log = []
        self.controller = _FakeTransferController(
            _takeover_aware_arm_interface(self.arm, self.web), self.filter_log
        )

    def test_filter_cleared_on_the_happy_path(self):
        self.controller.move_to_transfer_state()
        self.assertEqual(self.filter_log, [True, False])

    def test_filter_cleared_when_takeover_raises_during_the_approach(self):
        self.web.takeover_event.set()
        with self.assertRaises(WebInterfaceTakeoverInterrupt):
            self.controller.move_to_transfer_state()
        self.assertEqual(
            self.filter_log, [True, False],
            "noisy-readings filter leaked on: later transfers can stall",
        )

    def test_retract_move_surfaces_the_takeover(self):
        # The day-17 case: press lands during move_to_before_transfer_state,
        # which is the last motion in the skill when RetractAfterTransfer=0.
        self.web.takeover_event.set()
        with self.assertRaises(WebInterfaceTakeoverInterrupt):
            self.controller.move_to_before_transfer_state()

    def test_sim_path_untouched(self):
        controller = _FakeTransferController(None, self.filter_log)
        controller.move_to_transfer_state()
        self.assertEqual(self.filter_log, [], "no filter traffic without a robot")


class TakeoverToRedoTest(unittest.TestCase):
    """WebInterfaceTakeoverInterrupt must reach execute_action, which converts a
    'redo' choice into TeleopTakeoverException(redo_current=True) -- the signal
    run.py's attempt loop re-runs the skill on. Modelled here rather than
    imported, since the real path needs rospy."""

    def test_redo_choice_becomes_redo_current(self):
        outcomes = []

        class TeleopTakeoverException(Exception):
            def __init__(self, message="", redo_current=False):
                super().__init__(message)
                self.redo_current = redo_current

        def execute_action(tick, web, choice):
            try:
                tick()
            except WebInterfaceTakeoverInterrupt:
                # _maybe_handle_mid_skill_takeover: consume, run teleop, return
                # the user's post-teleop choice.
                consumed = web.takeover_event.is_set()
                web.takeover_event.clear()
                assert consumed, "takeover flag must still be latched here"
                raise TeleopTakeoverException(
                    "User took over during transfer",
                    redo_current=(choice == "redo"),
                ) from None

        def run_attempt_loop(tick, web, choice, max_attempts=3):
            attempt = 0
            while attempt < max_attempts:
                attempt += 1
                try:
                    execute_action(tick, web, choice)
                    outcomes.append(("completed", attempt))
                    return
                except TeleopTakeoverException as e:
                    if e.redo_current:
                        outcomes.append(("redo", attempt))
                        continue
                    outcomes.append(("next", attempt))
                    return

        # "Redo Skill": the skill re-runs, and the second attempt completes.
        web = _FakeWebInterface()
        web.takeover_event.set()
        arm = _FakeArm()
        proxy = _takeover_aware_arm_interface(arm, web)
        run_attempt_loop(lambda: proxy.execute_command("retract"), web, "redo")
        self.assertEqual(outcomes, [("redo", 1), ("completed", 2)])

        # "Next Skill": the executive moves on after a single attempt.
        outcomes.clear()
        web = _FakeWebInterface()
        web.takeover_event.set()
        proxy = _takeover_aware_arm_interface(_FakeArm(), web)
        run_attempt_loop(lambda: proxy.execute_command("retract"), web, "next")
        self.assertEqual(outcomes, [("next", 1)])


class SourceParityTest(unittest.TestCase):
    """Guard against the transcribed proxy above drifting from the real one."""

    def test_proxy_matches_source(self):
        import ast
        import inspect
        import pathlib
        import textwrap

        def normalize(class_node):
            """Executable form of a class: docstrings dropped (the real one
            carries the full rationale, this copy does not), comments already
            absent from the AST."""

            def strip_doc(node):
                body = node.body
                if (
                    body
                    and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)
                ):
                    body = body[1:]
                node.body = body
                for child in body:
                    if isinstance(child, (ast.FunctionDef, ast.ClassDef)):
                        strip_doc(child)

            strip_doc(class_node)
            return ast.unparse(class_node)

        source = pathlib.Path(__file__).resolve().parents[1] / (
            "src/feeding_deployment/actions/base.py"
        )
        # Parsed as text, never imported: base.py pulls in rospy/pybullet/torch.
        real_node = next(
            (
                n
                for n in ast.parse(source.read_text()).body
                if isinstance(n, ast.ClassDef)
                and n.name == "TakeoverAwareArmInterface"
            ),
            None,
        )
        self.assertIsNotNone(real_node, "TakeoverAwareArmInterface not found in base.py")

        mine_node = ast.parse(
            textwrap.dedent(inspect.getsource(TakeoverAwareArmInterface))
        ).body[0]

        self.assertEqual(
            normalize(real_node), normalize(mine_node),
            "base.TakeoverAwareArmInterface changed; update the copy in this test",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
