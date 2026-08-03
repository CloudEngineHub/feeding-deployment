"""Record labelled head-gesture examples on the robot, Enter to start, Enter to stop.

Writes the same pickle layout as the clips already in `gestures_examples/`
(`{gesture_label, gesture_description, positive_examples, negative_examples}`, each example
`{head_pose: [...frames], face_keypoints: [...frames]}`), so
`test_improved_static_head_detectors.py --data <file>` scores the result directly and
`MockPerceptionInterface` can replay it.

    python3 record_head_gestures.py --gesture nod                  # 10 positives, then 10 negatives
    python3 record_head_gestures.py --gesture nod --count 6
    python3 record_head_gestures.py --gesture shake --only negative --out my_shakes.pkl

Per example: Enter to start, Enter to stop, then Enter to keep / `r` to redo / `q` to move
on. The file is rewritten after every accepted example, so a crash mid-session keeps
whatever was recorded. At the end it replays everything through the current detector and
prints recall, false positives, and the gate that blocked each miss.

Record negatives that actually stress the detector -- the script cycles through
suggestions. A detector is only as good as the hardest negative in the set.
"""

import argparse
import pickle
import threading
import time
from pathlib import Path

import numpy as np
import rospy

from feeding_deployment.control.robot_controller.arm_client import ArmInterfaceClient
from feeding_deployment.integration.data_logger import DataLogger
from feeding_deployment.interfaces.perception_interface import PerceptionInterface
from feeding_deployment.perception.gestures_perception.improved_static_head_detectors import (
    NOD_PARAMETERS,
    SHAKE_PARAMETERS,
    _HeadOscillationTracker,
    _PITCH_INDEX,
    _POLL_PERIOD,
    _YAW_INDEX,
)

GESTURES = {
    "nod": {
        "label": "detect_head_nod",
        "description": "nodding head up and down",
        "parameters": NOD_PARAMETERS,
        "positive_prompt": "NOD clearly -- two or three nods, as you would to confirm a bite",
        "negative_prompts": [
            "sit still, looking at the robot",
            "sit still, looking down at the plate",
            "talk for a few seconds",
            "chew, as if eating a bite",
            "shake your head 'no'",
            "lean slowly forward, then back",
            "look left, then right, slowly",
            "one single small nod (should NOT count as a confirmation)",
            "blink and raise your eyebrows",
            "turn to talk to someone beside you",
        ],
    },
    "shake": {
        "label": "detect_head_shake",
        "description": "head is shaking from left to right",
        "parameters": SHAKE_PARAMETERS,
        "positive_prompt": "SHAKE your head 'no' -- two or three shakes",
        "negative_prompts": [
            "sit still, looking at the robot",
            "nod your head 'yes'",
            "talk for a few seconds",
            "chew, as if eating a bite",
            "look slowly left, hold, then slowly right",
            "turn to talk to someone beside you",
            "one single small head turn",
            "lean slowly forward, then back",
            "blink and raise your eyebrows",
            "look down at the plate and back up",
        ],
    },
}


def _unwrap(angles):
    """Unwrap a degree series the same way the detector does, for honest peak-to-peak."""
    angles = np.asarray(angles, dtype=float)
    if angles.size < 2:
        return angles
    steps = ((np.diff(angles) + 180.0) % 360.0) - 180.0
    return np.concatenate([[0.0], np.cumsum(steps)]) + angles[0]


class _Recorder(threading.Thread):
    """Collects distinct head-perception frames until asked to stop."""

    def __init__(self, perception_interface, quiet=False):
        super().__init__(daemon=True)
        self.perception_interface = perception_interface
        self.quiet = quiet
        self.stop_event = threading.Event()
        self.head_poses = []
        self.face_keypoints = []
        self.dropouts = 0
        self.repeats = 0
        self.start_time = None

    def run(self):
        self.start_time = time.time()
        last_head_pose = None
        while not self.stop_event.is_set():
            head_perception_data = self.perception_interface.get_head_perception_data()
            time.sleep(_POLL_PERIOD)

            if head_perception_data is None:
                self.dropouts += 1
                continue
            head_pose = head_perception_data.get("head_pose")
            keypoints = head_perception_data.get("face_keypoints")
            if head_pose is None or keypoints is None:
                self.dropouts += 1
                continue

            head_pose = tuple(float(value) for value in head_pose)
            if head_pose == last_head_pose:
                self.repeats += 1
                continue
            last_head_pose = head_pose

            self.head_poses.append(head_pose)
            # Only pose and landmarks: head_perception_data also carries the full RGB
            # frame, which would balloon the file for no benefit here.
            self.face_keypoints.append(np.asarray(keypoints))
            if not self.quiet:
                print(f"\r  recording: {len(self.head_poses):4d} frames  "
                      f"{time.time() - self.start_time:5.1f}s  "
                      f"pitch p2p {self._extent(_PITCH_INDEX):5.1f}  "
                      f"yaw p2p {self._extent(_YAW_INDEX):5.1f}   "
                      f"[Enter to stop]", end="", flush=True)

    def _extent(self, index):
        if len(self.head_poses) < 2:
            return 0.0
        return float(_unwrap([pose[index] for pose in self.head_poses]).ptp())

    def summary(self):
        duration = len(self.head_poses) * _POLL_PERIOD
        return (f"{len(self.head_poses)} frames (~{duration:.1f}s)  "
                f"pitch p2p {self._extent(_PITCH_INDEX):.1f} deg  "
                f"yaw p2p {self._extent(_YAW_INDEX):.1f} deg  "
                f"dropped {self.dropouts}  repeats {self.repeats}")

    def example(self):
        return {"head_pose": list(self.head_poses), "face_keypoints": list(self.face_keypoints)}


def _record_one(perception_interface, prompt):
    input(f"\n{prompt}\n  Enter to START ... ")
    recorder = _Recorder(perception_interface)
    recorder.start()
    input()
    recorder.stop_event.set()
    recorder.join()
    print(f"\r  stopped: {recorder.summary()}" + " " * 12)
    return recorder


def _collect(perception_interface, prompts, count, kind, save):
    """Record up to `count` examples, letting the user redo or cut the set short."""
    examples = []
    while len(examples) < count:
        index = len(examples)
        prompt = (f"[{kind} {index + 1}/{count}] {prompts[index % len(prompts)]}")
        recorder = _record_one(perception_interface, prompt)

        if not recorder.head_poses:
            print("  no frames captured -- is a face in view? redoing.")
            continue
        if len(recorder.head_poses) < 10:
            print(f"  only {len(recorder.head_poses)} frames; a gesture needs a couple of "
                  f"seconds. Keep anyway with Enter, or redo with 'r'.")

        choice = input("  Enter = keep, 'r' = redo, 'q' = done with this set: ").strip().lower()
        if choice.startswith("r"):
            continue
        if choice.startswith("q"):
            break
        examples.append(recorder.example())
        save(examples)
        print(f"  kept ({len(examples)}/{count} {kind}s)")
    return examples


def _score(gesture, positives, negatives):
    """Replay everything through the current detector and report what it would do."""
    settings = dict(GESTURES[gesture]["parameters"])
    print("\n" + "=" * 78)
    print(f"Current detector on the data just recorded ({gesture})")
    print("=" * 78)

    for kind, examples, want in (("positive", positives, True), ("negative", negatives, False)):
        if not examples:
            continue
        correct = 0
        print(f"\n{kind} examples ({len(examples)}):")
        for index, example in enumerate(examples):
            detected_at, blocked_by = _HeadOscillationTracker.replay(
                example["head_pose"], **settings)
            fired = detected_at is not None
            correct += int(fired == want)
            pitch = _unwrap([p[_PITCH_INDEX] for p in example["head_pose"]]).ptp()
            yaw = _unwrap([p[_YAW_INDEX] for p in example["head_pose"]]).ptp()
            verdict = f"fired at {detected_at:.1f}s" if fired else f"no fire ({blocked_by})"
            flag = "   " if fired == want else " <-"
            print(f"  {index:2d}  {len(example['head_pose']):4d} frames  "
                  f"pitch p2p {pitch:6.1f}  yaw p2p {yaw:6.1f}  {verdict}{flag}")
        label = "recall" if want else "correctly rejected"
        print(f"  {label}: {correct}/{len(examples)}")

    print("\nRows marked '<-' are the ones to tune against. Thresholds live in "
          "improved_static_head_detectors.py\n(NOD_* / SHAKE_* constants); re-score without "
          "re-recording via:")
    print(f"  python3 test_improved_static_head_detectors.py --data <this file>")


def _main(gesture: str, count: int, only: str, out: str, tool: str, overwrite: bool) -> None:
    spec = GESTURES[gesture]
    output_path = Path(out) if out else (
        Path(__file__).parent / "gestures_examples" / f"{gesture}_recorded.pkl")
    if output_path.exists() and not overwrite:
        raise SystemExit(f"{output_path} exists -- pass --overwrite, or --out to write elsewhere")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rospy.init_node("record_head_gestures")
    log_dir = Path(__file__).parents[1].parent / "integration" / "log" / "head_gesture_recording"
    log_dir.mkdir(parents=True, exist_ok=True)

    robot_interface = ArmInterfaceClient()
    perception_interface = PerceptionInterface(
        robot_interface=robot_interface,
        data_logger=DataLogger(state_dir=log_dir),
    )
    perception_interface.set_head_perception_tool(tool)
    perception_interface.start_head_perception_thread()

    dataset = {
        "gesture_label": spec["label"],
        "gesture_description": spec["description"],
        "positive_examples": [],
        "negative_examples": [],
    }

    def save():
        with open(output_path, "wb") as handle:
            pickle.dump(dataset, handle)

    try:
        print(f"Waiting for head perception ...")
        while perception_interface.get_head_perception_data() is None:
            time.sleep(0.2)
        print("Head perception is live.")

        if only in ("both", "positive"):
            print(f"\n{'=' * 78}\nPOSITIVE examples -- do the gesture\n{'=' * 78}")
            dataset["positive_examples"] = _collect(
                perception_interface, [spec["positive_prompt"]], count, "positive",
                lambda examples: (dataset.__setitem__("positive_examples", examples), save()))

        if only in ("both", "negative"):
            print(f"\n{'=' * 78}\nNEGATIVE examples -- everything that must NOT trigger it\n{'=' * 78}")
            dataset["negative_examples"] = _collect(
                perception_interface, spec["negative_prompts"], count, "negative",
                lambda examples: (dataset.__setitem__("negative_examples", examples), save()))

        save()
        print(f"\nSaved {len(dataset['positive_examples'])} positive and "
              f"{len(dataset['negative_examples'])} negative examples to {output_path}")
        _score(gesture, dataset["positive_examples"], dataset["negative_examples"])
    except KeyboardInterrupt:
        save()
        print(f"\nInterrupted -- kept what was recorded so far in {output_path}")
    finally:
        perception_interface.stop_head_perception_thread()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gesture", type=str, default="nod", choices=sorted(GESTURES))
    parser.add_argument("--count", type=int, default=10, help="examples per class")
    parser.add_argument("--only", type=str, default="both", choices=["both", "positive", "negative"],
                        help="record just one class (use --out to avoid clobbering the other)")
    parser.add_argument("--out", type=str, default=None,
                        help="output pickle (default gestures_examples/<gesture>_recorded.pkl)")
    parser.add_argument("--tool", type=str, default="fork", choices=["fork", "drink", "wipe"])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    _main(args.gesture, args.count, args.only, args.out, args.tool, args.overwrite)
