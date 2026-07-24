"""Recreate the 2026-07-23 18:17:26 OpenMicrowave stall and test whether it can be
worked around by re-sending / resuming the trajectory instead of handing to teleop.

Background (from session bundle session_20260723_173323_jul23_deployment_day4):
    open_microwave() grasped the handle, then sent an 11-waypoint Cartesian
    door-opening arc (Speed=high) as ONE ExecuteWaypointTrajectory. The arm ended
    >1 cm from the final waypoint, so kinova.move_cartesian_trajectory's endpoint
    check (kinova.py:729-734) returned False -> "Arm did not reach desired
    position" -> the executive (base.py:704-731) handed control to the user.
    The real firmware abort reason is swallowed (kinova.py:247-256 treats
    ACTION_END and ACTION_ABORT identically), so we only know "the endpoint was
    missed", not which waypoint physically stalled.

Hunch under test: a firmware failure like this can be worked around by simply
sending the goal again, or by resuming from where the arm stalled and sending the
rest of the waypoints one at a time (each point-to-point reach_pose re-solves IK
from the current configuration, which can clear a batched-trajectory stall).

This is a STANDALONE test harness. It talks to the arm server over the same RPC as
the deployment (ArmInterfaceClient) so the watchdog/bulldog e-stop chain stays live,
but it does NOT touch the executive / behaviour tree / main workflow.

The trajectory + the preamble that precedes it are read from the sibling JSON
(failed_trajectories/microwave_open_20260723_181726.json). Point --traj at another
extracted trajectory to replay a different incident with the same machinery.

Safety:
    * The end effector sweeps a ~35 cm arc at z ~= 0.52 m in the arm base frame.
      Clear that volume (or have the real microwave in place -- see below).
    * Free-space replay reproduces a KINEMATIC stall (joint limit / singularity).
      If the original stall was the door physically resisting, free space will not
      reproduce it -- run in front of the real microwave with the handle grasped.
    * Default speed is "high" to match the run. Consider --speed low first.

Usage:
    # Dry run: print the plan + the trajectory, connect to nothing, move nothing.
    python resume_microwave_open_traj.py --dry-run

    # Reproduce only: preamble + full arc, report whether it stalls. No recovery.
    python resume_microwave_open_traj.py --mode reproduce

    # Full hunch test (default): reproduce, then on failure resume from the stall
    # point by re-sending each remaining waypoint point-to-point (--retries each).
    python resume_microwave_open_traj.py

    # Other recovery strategies:
    python resume_microwave_open_traj.py --resume-strategy subtrajectory
    python resume_microwave_open_traj.py --resume-strategy retry-full --retries 3

    # Walk the whole arc point-to-point from the start and report the first
    # waypoint the arm cannot reach (localise the kinematic wall directly).
    python resume_microwave_open_traj.py --mode localize

    # Quick kinematic test without grasping a handle: skip the staging/grasp
    # preamble and just go to the arc start, then run.
    python resume_microwave_open_traj.py --no-preamble --skip-gripper --speed low
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

DEFAULT_TRAJ = Path(__file__).parent / "failed_trajectories" / "microwave_open_20260723_181726.json"


# --------------------------------------------------------------------------- #
# Trajectory loading                                                          #
# --------------------------------------------------------------------------- #
def load_trajectory(path: Path) -> dict:
    """Read the incident JSON into plain python: speed, preamble, and the arc as a
    list of (pos[3], quat[4]) tuples of floats."""
    with open(path, "r") as f:
        data = json.load(f)

    waypoints = [
        (list(map(float, wp["pos"])), list(map(float, wp["quat"])))
        for wp in data["opening_waypoints"]
    ]
    for pos, quat in waypoints:
        assert len(pos) == 3 and len(quat) == 4, "each waypoint needs pos[3] + quat[4]"

    pre = data.get("preamble", {})
    preamble = {
        "staging_joint_pos": [float(x) for x in pre["staging_joint_pos"]] if pre.get("staging_joint_pos") else None,
        "pre_grasp_pose": (
            ([float(x) for x in pre["pre_grasp_pose"]["pos"]], [float(x) for x in pre["pre_grasp_pose"]["quat"]])
            if pre.get("pre_grasp_pose") else None
        ),
        "grasp_pose": (
            ([float(x) for x in pre["grasp_pose"]["pos"]], [float(x) for x in pre["grasp_pose"]["quat"]])
            if pre.get("grasp_pose") else None
        ),
    }
    return {
        "speed": data.get("speed", "medium"),
        "preamble": preamble,
        "waypoints": waypoints,
        "meta": {k: data.get(k) for k in ("incident", "when", "failure_mode")},
    }


# --------------------------------------------------------------------------- #
# Small helpers                                                               #
# --------------------------------------------------------------------------- #
def confirm(prompt: str, assume_yes: bool) -> bool:
    if assume_yes:
        print(f"{prompt}  [auto-yes]")
        return True
    return input(f"{prompt} [y/N] ").strip().lower() == "y"


def ee_xyz(client) -> np.ndarray:
    return np.asarray(client.get_state()["ee_pos"][:3], dtype=float)


def print_state(client, label: str) -> None:
    state = client.get_state()
    print(f"  {label} joints:", np.round(state["position"], 5).tolist())
    print(f"  {label} ee_pos:", np.round(state["ee_pos"], 5).tolist())


def dist_cm(a, b) -> float:
    return 100.0 * float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


class Sender:
    """Wraps ArmInterfaceClient so every send returns a uniform (ok, note) and never
    raises out of a step. `ok` is True only when the driver reported success."""

    def __init__(self, client, commands, dry_run: bool):
        self.client = client
        self.C = commands  # command_interface module
        self.dry_run = dry_run

    def _run(self, label, cmd):
        print(f"--> {label}")
        if self.dry_run:
            print("    [dry-run] not sent")
            return True, "dry-run"
        start = time.time()
        try:
            result = self.client.execute_command(cmd)
        except Exception as e:  # RPC re-raises a simplified Exception on firmware error
            print(f"    raised after {time.time() - start:.1f}s: {e}")
            return False, f"exception: {e}"
        ok = result is True
        print(f"    done in {time.time() - start:.1f}s, returned {result!r}")
        return ok, f"returned {result!r}"

    def joint(self, pos, label):
        return self._run(label, self.C.JointCommand(pos))

    def cartesian(self, pos, quat, label):
        return self._run(label, self.C.CartesianCommand(pos, quat))

    def cartesian_traj(self, waypoints, label):
        return self._run(label, self.C.CartesianTrajectoryCommand(waypoints))

    def open_gripper(self):
        return self._run("open gripper", self.C.OpenGripperCommand())

    def close_gripper(self):
        return self._run("close gripper", self.C.CloseGripperCommand())


# --------------------------------------------------------------------------- #
# Phases                                                                      #
# --------------------------------------------------------------------------- #
def run_preamble(sender, preamble, args) -> None:
    """Put the arm in the same starting configuration the run had before the arc."""
    if args.no_preamble:
        pos, quat = None, None
        # move straight to the arc start instead
        return
    if preamble["staging_joint_pos"] is not None:
        sender.joint(preamble["staging_joint_pos"], "staging joint move (fridge_door_staging_pos)")
    if preamble["pre_grasp_pose"] is not None:
        sender.cartesian(*preamble["pre_grasp_pose"], label="pre-grasp pose")
    if not args.skip_gripper:
        sender.open_gripper()
    if preamble["grasp_pose"] is not None:
        sender.cartesian(*preamble["grasp_pose"], label="grasp pose")
    if not args.skip_gripper:
        sender.close_gripper()


def reach_and_measure(sender, waypoints, i, tol_cm, retries):
    """Point-to-point reach of waypoint i (reach_pose), retrying up to `retries`
    extra times -- this is the 're-send the goal' hunch. Returns (reached, attempts,
    dist_cm)."""
    pos, quat = waypoints[i]
    n = len(waypoints)
    for attempt in range(1, retries + 2):  # 1 initial + `retries` extra
        tag = f"waypoint {i + 1}/{n}" + ("" if attempt == 1 else f" (re-send #{attempt - 1})")
        ok, _ = sender.cartesian(pos, quat, label=f"{tag} {np.round(pos, 4).tolist()}")
        if sender.dry_run:
            return True, attempt, 0.0
        d = dist_cm(ee_xyz(sender.client), pos)
        print(f"    reached within {d:.2f} cm (tol {tol_cm} cm), driver_ok={ok}")
        if ok or d <= tol_cm:
            return True, attempt, d
    return False, retries + 1, d


def nearest_waypoint(client, waypoints) -> int:
    cur = ee_xyz(client)
    dists = [dist_cm(cur, wp[0]) for wp in waypoints]
    k = int(np.argmin(dists))
    print(f"  Arm is closest to waypoint {k + 1}/{len(waypoints)} ({dists[k]:.2f} cm away).")
    return k




def stepwise_resume(sender, waypoints, start_index, args):
    """Send waypoints[start_index:] one at a time, re-sending each on failure and
    continuing past any that stay unreachable. Returns a per-waypoint report."""
    report = []
    n = len(waypoints)
    for i in range(start_index, n):
        reached, attempts, d = reach_and_measure(sender, waypoints, i, args.tol_cm, args.retries)
        report.append({"index": i, "reached": reached, "attempts": attempts, "dist_cm": d})
        if not reached:
            print(f"    waypoint {i + 1}/{n} NOT reached after {attempts} attempts; continuing with the rest.")
    return report


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traj", type=Path, default=DEFAULT_TRAJ, help="incident trajectory JSON")
    ap.add_argument("--mode", choices=["reproduce", "localize", "resume"], default="resume")
    ap.add_argument("--resume-strategy", choices=["stepwise", "subtrajectory", "retry-full"], default="stepwise",
                    help="how to recover after the batched arc fails")
    ap.add_argument("--speed", choices=["low", "medium", "high"], default=None,
                    help="arm speed (default: the speed from the incident JSON)")
    ap.add_argument("--retries", type=int, default=2, help="extra re-sends of a stalled goal")
    ap.add_argument("--tol-cm", type=float, default=1.0, help="reached tolerance (matches kinova's 1 cm check)")
    ap.add_argument("--no-preamble", action="store_true", help="skip staging/grasp; go straight to the arc start")
    ap.add_argument("--skip-gripper", action="store_true", help="do not open/close the gripper in the preamble")
    ap.add_argument("--yes", action="store_true", help="skip confirmation prompts")
    ap.add_argument("--dry-run", action="store_true", help="print the plan; connect to nothing, move nothing")
    args = ap.parse_args()

    traj = load_trajectory(args.traj)
    waypoints = traj["waypoints"]
    speed = args.speed or traj["speed"]
    n = len(waypoints)

    print("=" * 78)
    print(f"Incident : {traj['meta'].get('incident')}")
    print(f"When     : {traj['meta'].get('when')}")
    print(f"Traj file: {args.traj}")
    print(f"Mode     : {args.mode}   resume-strategy: {args.resume_strategy}   speed: {speed}")
    print(f"Arc      : {n} waypoints, z ~= {np.round(waypoints[0][0][2], 4)} m (arm_base_link frame)")
    print(f"Final wp : {np.round(waypoints[-1][0], 5).tolist()}")
    print("=" * 78)

    # Import the ROS/arm stack lazily so --dry-run works off-robot.
    if not args.dry_run:
        import rospy
        from feeding_deployment.control.robot_controller.arm_client import ArmInterfaceClient
        from feeding_deployment.control.robot_controller import command_interface

        rospy.init_node("resume_microwave_open_traj", anonymous=True)
        client = ArmInterfaceClient()
        sender = Sender(client, command_interface, dry_run=False)

        print("\nCurrent state:")
        print_state(client, "start")
        print("\nSAFETY: the EE sweeps a ~35 cm arc at z ~= 0.52 m. Make sure that volume is")
        print("clear, or that the real microwave handle is grasped as in the run.")
        if not confirm(f"Proceed at speed={speed}?", args.yes):
            return
        client.set_speed(speed)
    else:
        # dry-run: use a tiny shim so Sender can format command labels without ROS.
        class _Shim:
            def __getattr__(self, _):  # JointCommand(...) etc. -> a no-op callable
                return lambda *a, **k: None
        sender = Sender(None, _Shim(), dry_run=True)

    # --- preamble ---------------------------------------------------------- #
    print("\n--- Preamble (reach the arc's starting configuration) ---")
    run_preamble(sender, traj["preamble"], args)
    if args.no_preamble:
        sender.cartesian(*waypoints[0], label=f"move to arc start (waypoint 1/{n})")

    if not args.dry_run:
        print("\nAt arc start:")
        print_state(client, "arc-start")

    # --- localize mode: walk the whole arc point-to-point ------------------ #
    if args.mode == "localize":
        print("\n--- Localise: point-to-point through the full arc ---")
        report = stepwise_resume(sender, waypoints, 0, args)
        _summary(report, n, args, reproduced=None)
        return

    # --- reproduce the batched send --------------------------------------- #
    print("\n--- Reproduce: send all waypoints as ONE trajectory (as the run did) ---")
    ok_batch, _ = sender.cartesian_traj(waypoints, label=f"batched arc ({n} waypoints)")
    if not args.dry_run:
        d_final = dist_cm(ee_xyz(client), waypoints[-1][0])
        print(f"  Final EE is {d_final:.2f} cm from the goal waypoint (kinova check: <= 1 cm).")
        reproduced = not ok_batch
        print(f"  RESULT: batched arc {'FAILED (stall reproduced)' if reproduced else 'completed (stall NOT reproduced)'}.")
    else:
        reproduced = True  # assume for planning purposes

    if args.mode == "reproduce":
        _summary([], n, args, reproduced=reproduced)
        return

    if not reproduced:
        print("\nBatched arc completed, so there is nothing to resume. "
              "Try --speed high, or run with the real microwave present.")
        _summary([], n, args, reproduced=False)
        return

    # --- resume ------------------------------------------------------------ #
    print(f"\n--- Resume (strategy: {args.resume_strategy}) ---")
    report = []
    if args.resume_strategy == "retry-full":
        for attempt in range(1, args.retries + 2):
            ok, _ = sender.cartesian_traj(waypoints, label=f"re-send full arc (attempt {attempt})")
            if args.dry_run:
                break
            d = dist_cm(ee_xyz(client), waypoints[-1][0])
            print(f"    final {d:.2f} cm from goal, driver_ok={ok}")
            report.append({"attempt": attempt, "ok": ok, "dist_cm": d})
            if ok or d <= args.tol_cm:
                break
    elif args.resume_strategy == "subtrajectory":
        if args.dry_run:
            start_index = 1
        else:
            # The stall point is the nearest waypoint; the arm has effectively
            # traversed everything up to it. Resume from the NEXT waypoint: sending
            # the tail only, and skipping the coincident stall waypoint (a ~0 first
            # segment makes Kinova abort a blended trajectory at init).
            stall = nearest_waypoint(client, waypoints)
            start_index = min(stall + 1, n)
            print(f"  Resuming sub-trajectory from waypoint {start_index + 1}/{n} (the tail past the stall).")
        if start_index >= n:
            print("  Stall was at the final waypoint; nothing left to resume as a trajectory.")
        else:
            rest = waypoints[start_index:]
            ok, _ = sender.cartesian_traj(rest, label=f"send the rest as one sub-trajectory (wp {start_index + 1}..{n})")
            if not args.dry_run:
                d = dist_cm(ee_xyz(client), waypoints[-1][0])
                print(f"    final {d:.2f} cm from goal, driver_ok={ok}")
                report.append({"start_index": start_index, "ok": ok, "dist_cm": d})
    else:  # stepwise: re-sending the stall waypoint is harmless, so start at nearest
        start_index = 0 if args.dry_run else nearest_waypoint(client, waypoints)
        report = stepwise_resume(sender, waypoints, start_index, args)

    _summary(report, n, args, reproduced=reproduced)


def _summary(report, n, args, reproduced) -> None:
    print("\n" + "=" * 78)
    print("SUMMARY")
    if reproduced is not None:
        print(f"  batched arc reproduced the stall: {reproduced}")
    if args.dry_run:
        print("  [dry-run] no arm activity; re-run without --dry-run on the robot.")
        print("=" * 78)
        return
    if not report:
        # e.g. --mode reproduce: nothing was resumed/localised, so there is no
        # point-to-point result to report and no hunch verdict to draw.
        print("=" * 78)
        return
    if args.mode in ("localize",) or args.resume_strategy == "stepwise":
        reached = [r for r in report if r.get("reached")]
        print(f"  waypoints reached point-to-point: {len(reached)}/{len(report)} attempted")
        for r in report:
            state = "ok" if r.get("reached") else "MISS"
            print(f"    wp {r['index'] + 1:>2}/{n}: {state:>4}  attempts={r.get('attempts')}  dist={r.get('dist_cm'):.2f} cm")
        first_miss = next((r for r in report if not r.get("reached")), None)
        if first_miss is not None:
            print(f"  first unreachable waypoint: {first_miss['index'] + 1}/{n} "
                  f"({np.round(first_miss.get('dist_cm', 0), 2)} cm short)")
        else:
            print("  HUNCH SUPPORTED: point-to-point re-sending walked the whole arc past the batch stall.")
    else:
        for r in report:
            print(f"    {r}")
        if report and report[-1].get("ok"):
            print("  HUNCH SUPPORTED: re-sending recovered the arc.")
    print("=" * 78)


if __name__ == "__main__":
    main()
