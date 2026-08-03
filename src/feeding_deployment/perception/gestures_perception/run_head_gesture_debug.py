"""Stream the head nod / head shake detector's internals live, for threshold calibration.

Brings up just enough of the stack to get head perception running (arm client for TF +
RealSense, then the head-perception thread) and runs the detector with debug printing on.
By default it never returns on a detection: it reports the hit, clears the buffer and keeps
watching, so several attempts can be tried in one run.

    python3 run_head_gesture_debug.py --gesture nod
    python3 run_head_gesture_debug.py --gesture shake --tool drink
    python3 run_head_gesture_debug.py --gesture nod --once          # stop at first hit
    python3 run_head_gesture_debug.py --gesture nod | tee /tmp/nod.log

Reading the stream. One line per distinct DECA frame:

    dt        seconds since the previous distinct frame. If this is much above 0.1, DECA is
              delivering slower than the poll rate, and `deg/s` gets noisier as a result.
    roll/pitch/yaw
              the raw head_pose angles, all three, so the axis mapping can be re-checked:
              nod should move `pitch`, shake should move `yaw`. If the wrong column moves
              when you gesture, the mapping -- not the thresholds -- is the problem.
    unwrap    pitch (or yaw) after +-180 unwrapping. Should be continuous even when the raw
              column jumps ~358.
    step      change in `unwrap` since the last frame; `deg/s` is that over `dt`.
    base/dev  the window median and the current deviation from it. `dev` is what gets
              compared against the enter threshold.
    dir/hc    current direction, and counted direction changes out of those required.
              '^' marks a frame that counted one.
    p2p_pri/p2p_crs/ratio
              peak-to-peak of each axis over the window, and their ratio, which must beat
              the dominance threshold.
    verdict   which gate is holding the detection back, or *** DETECTED ***.

Calibrating. Sit still for ~15 s first: `p2p_pri` is then your live noise floor, and
NOD_MIN_AMPLITUDE_DEG / NOD_ENTER_DEG in `improved_static_head_detectors.py` need to sit
above it. Then nod normally and read off the `p2p_pri` and `dev` a real gesture reaches.
Watch for `SLEW RESET` lines during a deliberate gesture -- those mean the motion is being
discarded as a glitch.
"""

import argparse
from pathlib import Path
import time

import rospy

from feeding_deployment.control.robot_controller.arm_client import ArmInterfaceClient
from feeding_deployment.integration.data_logger import DataLogger
from feeding_deployment.interfaces.perception_interface import PerceptionInterface
from feeding_deployment.perception.gestures_perception.improved_static_head_detectors import (
    head_nod,
    head_shake,
)

DETECTORS = {"nod": head_nod, "shake": head_shake}


def _wait_for_head_perception(perception_interface, timeout=60.0):
    """Block until DECA produces a face, so the stream doesn't start on empty polls."""
    print(f"Waiting up to {timeout:.0f}s for the first head-perception frame ...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        if perception_interface.get_head_perception_data() is not None:
            print("Head perception is live.")
            return True
        time.sleep(0.2)
    print("No head-perception frame arrived -- is a face in view of the camera?")
    return False


def _main(gesture: str, tool: str, timeout: float, once: bool, simulate_head_perception: bool) -> None:
    rospy.init_node("run_head_gesture_debug")

    log_dir = Path(__file__).parents[1].parent / "integration" / "log" / "head_gesture_debug"
    log_dir.mkdir(parents=True, exist_ok=True)

    robot_interface = ArmInterfaceClient()
    data_logger = DataLogger(state_dir=log_dir)
    perception_interface = PerceptionInterface(
        robot_interface=robot_interface,
        simulate_head_perception=simulate_head_perception,
        data_logger=data_logger,
    )
    # get_head_perception_data() writes its pose pickle to a per-tool filename, so the tool
    # has to be set before the thread starts.
    perception_interface.set_head_perception_tool(tool)
    perception_interface.start_head_perception_thread()

    try:
        if not _wait_for_head_perception(perception_interface):
            return
        detected = DETECTORS[gesture](
            perception_interface,
            None,
            timeout,
            debug=True,
            continuous=not once,
        )
        print(f"\n{gesture}: {'detected at least once' if detected else 'never detected'}")
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        perception_interface.stop_head_perception_thread()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gesture", type=str, default="nod", choices=sorted(DETECTORS))
    parser.add_argument("--tool", type=str, default="fork", choices=["fork", "drink", "wipe"],
                        help="tool head perception aims for (affects the tool-tip target only)")
    parser.add_argument("--timeout", type=float, default=600.0,
                        help="how long to keep watching, in seconds")
    parser.add_argument("--once", action="store_true",
                        help="return on the first detection instead of watching continuously")
    parser.add_argument("--simulate_head_perception", action="store_true",
                        help="replay the logged head_perception_data pickle instead of live DECA")
    args = parser.parse_args()

    _main(args.gesture, args.tool, args.timeout, args.once, args.simulate_head_perception)
