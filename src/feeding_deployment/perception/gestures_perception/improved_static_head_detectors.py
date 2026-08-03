"""Head-gesture detectors that key off a real oscillation rather than a raw angle delta.

Drop-in replacement for `static_gesture_detectors`: same module layout, same
`(perception_interface, termination_event, timeout) -> bool` detector signature, and a
`function_name_to_label` mapping, so `inspect.getmembers(module, inspect.isfunction)`
gesture discovery in `actions/base.py` / `actions/emulate_transfer.py` keeps working.
`debug` and `continuous` are keyword-only extras with defaults, so a 3-positional-argument
call from the existing call sites behaves exactly as before.

Why this module exists. `head_pose` is built in
`perception/head_perception/deca_perception.py:get_head_pose_from_neck_frame()` by taking
`Rotation.from_matrix(neck_frame[:3, :3]).as_euler("xyz", degrees=True)`. In deployment the
pitch channel sits right at +-180 degrees, so that decomposition flips branch constantly:
a perfectly still head emits single-frame steps of ~358 degrees. Across the 80 clips in
`gestures_examples/*.pkl`, 16 contain such a flip, while a genuine nod is only 16-20
degrees peak-to-peak. Any detector comparing raw angles against a fixed threshold is
therefore measuring branch flips, not head motion.

So every reading here is first unwrapped, and a gesture must be an *oscillation*: repeated
alternating excursions about the recent median, at gesture pace, on the expected axis.

Calibrating on the robot. The thresholds below were tuned against the recorded clips, whose
noise floor may not match a live session. Run
`python3 run_head_gesture_debug.py --gesture nod` to stream every quantity the decision
uses, one line per frame, plus the gate that blocked each non-detection.
"""

import time
from collections import deque

import numpy as np

function_name_to_label = {
    "mouth_open": "mouth open",
    "head_nod": "head nod",
    "head_shake": "head shake",
}

# head_pose is (x, y, z, roll, pitch, yaw), angles in degrees. Nodding shows up on pitch
# (16-20 deg peak-to-peak, yaw near-static at 2-4) and shaking on yaw (56-62 deg, pitch
# 14-32); both confirmed against the recorded clips rather than assumed from the frame.
# If a live session disagrees, the debug stream prints all three angles so the mapping can
# be re-checked directly.
_ROLL_INDEX = 3
_PITCH_INDEX = 4
_YAW_INDEX = 5

_POLL_PERIOD = 0.1  # 10 Hz, matching the head-perception thread.

# A step faster than this is a tracking glitch, not a head: genuine nods reach ~140 deg/s
# on pitch and shakes ~290 deg/s on yaw. Expressed per second rather than per sample
# because the gap between *distinct* DECA frames is not guaranteed to be _POLL_PERIOD --
# at 10 Hz this is the same 45 deg/sample as before, but it no longer tightens into real
# gesture territory when frames arrive more slowly.
_MAX_SLEW_DEG_PER_SEC = 450.0

# Losing the face for longer than this discards accumulated evidence, so no gesture is
# ever assembled from readings taken either side of a dropout.
_MAX_FRAME_GAP = 1.0

# Floor on how fast successive direction changes may arrive. A glitch can flip sign in one
# or two frames; a human neck cannot.
_MIN_HALF_PERIOD = 0.15

_MIN_WINDOW_SAMPLES = 3

# Three half-cycles is one and a half nods, which is what a real confirmation looks like.
# These started at four half-cycles past +-4 deg, tuned on gestures_examples/head_nod.pkl,
# where each clip runs ~4.7 s and holds 8 half-cycles -- someone nodding continuously for
# five seconds. A live session (gestures_examples/nod_recorded.pkl) nods for ~2.0 s with a
# median of 3 half-cycles, so the old pair recognised 2 of 10 real nods. Amplitudes agreed
# across both sets (10-22 deg peak-to-peak), so only the count and the enter threshold
# moved; dropping enter to 3 deg is what lifts the smallest nods (9.7 deg p2p, so barely
# +-4.9 about the median) over the line. Verified on both sets at once: 10/10 and 5/5
# recall, 10/10 live non-nods rejected, still nothing on shakes.
NOD_ENTER_DEG = 3.0
NOD_REQUIRED_HALF_CYCLES = 3
NOD_WINDOW_SECONDS = 3.0
NOD_DOMINANCE = 1.2
NOD_MIN_AMPLITUDE_DEG = 8.0

# Shake is left at three half-cycles: it swings +-28 deg, where three is already
# unambiguous. Not yet checked against a live recording -- run record_head_gestures.py
# --gesture shake to confirm these hold up the way the nod thresholds did not.
SHAKE_ENTER_DEG = 12.0
SHAKE_REQUIRED_HALF_CYCLES = 3
SHAKE_WINDOW_SECONDS = 2.5
SHAKE_DOMINANCE = 1.5
SHAKE_MIN_AMPLITUDE_DEG = 24.0

# Single source of truth for what each detector runs with, so calibration is one edit here
# and the offline tools (`test_improved_static_head_detectors.py`,
# `record_head_gestures.py`) score exactly what deployment would do.
NOD_PARAMETERS = dict(
    primary_index=_PITCH_INDEX,
    cross_index=_YAW_INDEX,
    enter_deg=NOD_ENTER_DEG,
    required_half_cycles=NOD_REQUIRED_HALF_CYCLES,
    window=NOD_WINDOW_SECONDS,
    dominance=NOD_DOMINANCE,
    min_amplitude_deg=NOD_MIN_AMPLITUDE_DEG,
)

SHAKE_PARAMETERS = dict(
    primary_index=_YAW_INDEX,
    cross_index=_PITCH_INDEX,
    enter_deg=SHAKE_ENTER_DEG,
    required_half_cycles=SHAKE_REQUIRED_HALF_CYCLES,
    window=SHAKE_WINDOW_SECONDS,
    dominance=SHAKE_DOMINANCE,
    min_amplitude_deg=SHAKE_MIN_AMPLITUDE_DEG,
)


class _HeadOscillationTracker:
    """Streaming test for a deliberate repeated head oscillation on one axis.

    Deliberately a class, not a set of module-level helpers: gesture discovery collects
    every module-level function as a user-selectable gesture, and `emulate_transfer`
    then raises `KeyError` on any that is missing from `function_name_to_label`.

    Feed samples with `update()`; it returns True the moment the axis has completed
    `required_half_cycles` alternating excursions past `+/- enter_deg` about the median of
    the trailing `window`, at plausible pace, with enough amplitude, and with the gesture
    axis moving more than the other one. Every call also leaves a snapshot in
    `last_reading` naming the gate that blocked detection.

    Measured with the tuned constants above: nod 10/10 on a live recording
    (`gestures_examples/nod_recorded.pkl`, median latency 1.1 s) and 5/5 on the older
    `head_nod.pkl` clips, with all 10 live non-nod examples rejected; shake 5/5 on
    `shake_my_head_from_left_to_right.pkl`; neither detector fires on the other's gesture.
    The detector this replaces managed 3/5 on the old nod clips -- and those three were
    exactly the three containing a +-180 branch flip.
    """

    def __init__(self, enter_deg, required_half_cycles, window, dominance, min_amplitude_deg):
        self.enter_deg = enter_deg
        self.required_half_cycles = required_half_cycles
        self.window = window
        self.dominance = dominance
        self.min_amplitude_deg = min_amplitude_deg
        self.last_reading = {}
        self.reset()

    def reset(self):
        """Drop all accumulated evidence."""
        self._samples = deque()      # (timestamp, primary unwrapped, cross unwrapped)
        self._half_cycles = deque()  # timestamps of confirmed direction changes
        self._last_time = None
        self._previous_raw = None
        self._unwrapped = None
        self._direction = 0

    def settings_summary(self, primary_name, cross_name):
        """One-line description of every threshold in play."""
        return (f"axis {primary_name} vs {cross_name} | enter +-{self.enter_deg:.1f} deg | "
                f"{self.required_half_cycles} half-cycles within {self.window:.1f}s | "
                f"min p2p {self.min_amplitude_deg:.1f} deg | dominance {self.dominance:.2f}x | "
                f"slew limit {_MAX_SLEW_DEG_PER_SEC:.0f} deg/s | "
                f"gap reset {_MAX_FRAME_GAP:.1f}s | min half-period {_MIN_HALF_PERIOD:.2f}s")

    @staticmethod
    def _shortest_angle(delta):
        """Map a difference of two angles in degrees onto (-180, 180]."""
        return ((delta + 180.0) % 360.0) - 180.0

    def update(self, timestamp, primary_raw, cross_raw):
        """Feed one reading; return True once an oscillation is confirmed."""
        interval = None if self._last_time is None else timestamp - self._last_time
        stale = interval is not None and interval > _MAX_FRAME_GAP

        primary_step = cross_step = 0.0
        slew = 0.0
        glitch = False
        if self._previous_raw is not None and not stale:
            primary_step = self._shortest_angle(primary_raw - self._previous_raw[0])
            cross_step = self._shortest_angle(cross_raw - self._previous_raw[1])
            elapsed = max(interval, 1e-3) if interval else _POLL_PERIOD
            slew = max(abs(primary_step), abs(cross_step)) / elapsed
            glitch = slew > _MAX_SLEW_DEG_PER_SEC

        reading = {
            "time": timestamp,
            "interval": interval,
            "primary_raw": primary_raw,
            "cross_raw": cross_raw,
            "step": primary_step,
            "slew": slew,
        }

        # An anchoring frame starts the history over, so it can never itself be a
        # detection however good the numbers on it look.
        anchoring = self._previous_raw is None or stale or glitch

        if anchoring:
            # Nothing before this reading can be trusted, so re-anchor on it.
            self.reset()
            self._unwrapped = (primary_raw, cross_raw)
            if glitch:
                reading["blocked_by"] = f"SLEW RESET {slew:.0f}>{_MAX_SLEW_DEG_PER_SEC:.0f} deg/s"
            elif stale:
                reading["blocked_by"] = f"GAP RESET {interval:.1f}s>{_MAX_FRAME_GAP:.1f}s"
            else:
                reading["blocked_by"] = "anchor"
        else:
            # Accumulate shortest-path steps, which is what makes a +-180 branch flip read
            # as the ~0 degrees of motion it actually is.
            self._unwrapped = (self._unwrapped[0] + primary_step,
                               self._unwrapped[1] + cross_step)

        self._previous_raw = (primary_raw, cross_raw)
        self._last_time = timestamp

        primary, cross = self._unwrapped
        self._samples.append((timestamp, primary, cross))
        while self._samples and timestamp - self._samples[0][0] > self.window:
            self._samples.popleft()
        while self._half_cycles and timestamp - self._half_cycles[0] > self.window:
            self._half_cycles.popleft()

        reading.update({"unwrapped": primary, "cross_unwrapped": cross,
                        "samples": len(self._samples)})

        if len(self._samples) < _MIN_WINDOW_SAMPLES:
            reading.setdefault("blocked_by", f"warmup {len(self._samples)}/{_MIN_WINDOW_SAMPLES}")
            self.last_reading = reading
            return False

        primary_window = [sample[1] for sample in self._samples]
        cross_window = [sample[2] for sample in self._samples]

        # Median of the trailing window, not an all-time extreme: it self-centres, so
        # repeated one-sided nods still alternate about it, and slow drift (leaning in,
        # looking down at the plate) is subtracted instead of accumulating forever.
        baseline = float(np.median(primary_window))
        deviation = primary - baseline

        # Only a crossing opposite to the previous one counts, which makes alternation
        # structural -- a single monotonic sweep can never satisfy the count.
        crossed = False
        if self._direction <= 0 and deviation > self.enter_deg:
            self._direction = 1
            self._half_cycles.append(timestamp)
            crossed = True
        elif self._direction >= 0 and deviation < -self.enter_deg:
            self._direction = -1
            self._half_cycles.append(timestamp)
            crossed = True

        primary_extent = max(primary_window) - min(primary_window)
        cross_extent = max(cross_window) - min(cross_window)
        span = (self._half_cycles[-1] - self._half_cycles[0]) if self._half_cycles else 0.0

        reading.update({
            "baseline": baseline,
            "deviation": deviation,
            "direction": self._direction,
            "crossed": crossed,
            "half_cycles": len(self._half_cycles),
            "span": span,
            "primary_extent": primary_extent,
            "cross_extent": cross_extent,
        })

        blocked_by = None
        if len(self._half_cycles) < self.required_half_cycles:
            blocked_by = f"half-cycles {len(self._half_cycles)}/{self.required_half_cycles}"
        else:
            required_span = self._half_cycles[-1] - self._half_cycles[-self.required_half_cycles]
            floor = _MIN_HALF_PERIOD * (self.required_half_cycles - 1)
            if required_span < floor:
                blocked_by = f"too fast {required_span:.2f}<{floor:.2f}s"
            elif primary_extent < self.min_amplitude_deg:
                blocked_by = f"amplitude {primary_extent:.1f}<{self.min_amplitude_deg:.1f}"
            elif primary_extent < self.dominance * cross_extent:
                # Nods swing yaw a little and shakes swing pitch a lot, so requiring the
                # gesture axis to dominate is what keeps nod and shake from triggering
                # each other.
                blocked_by = (f"dominance {primary_extent:.1f}<"
                              f"{self.dominance:.2f}x{cross_extent:.1f}")

        if not anchoring:
            reading["blocked_by"] = blocked_by

        self.last_reading = reading
        return blocked_by is None and not anchoring

    DEBUG_HEADER = (f"{'t':>7s} {'dt':>5s} | {'roll':>7s} {'pitch':>7s} {'yaw':>7s} | "
                    f"{'unwrap':>8s} {'step':>6s} {'deg/s':>6s} | {'base':>8s} {'dev':>6s} | "
                    f"{'dir':>3s} {'hc':>5s} {'span':>5s} | {'n':>3s} {'p2p_pri':>7s} "
                    f"{'p2p_crs':>7s} {'ratio':>5s} | verdict")

    def format_reading(self, head_pose, start_time, detected):
        """Render `last_reading` as one aligned line of the debug stream."""
        reading = self.last_reading

        def number(key, width, digits):
            value = reading.get(key)
            return f"{'-':>{width}s}" if value is None else f"{value:{width}.{digits}f}"

        cross_extent = reading.get("cross_extent")
        primary_extent = reading.get("primary_extent")
        if primary_extent is None or cross_extent is None:
            ratio = f"{'-':>5s}"
        elif cross_extent < 1e-6:
            ratio = f"{'inf':>5s}"
        else:
            ratio = f"{primary_extent / cross_extent:5.2f}"

        verdict = "*** DETECTED ***" if detected else (reading.get("blocked_by") or "-")
        marker = "^" if reading.get("crossed") else " "

        return (f"{reading['time'] - start_time:7.2f} {number('interval', 5, 3)} | "
                f"{head_pose[_ROLL_INDEX]:7.1f} {head_pose[_PITCH_INDEX]:7.1f} "
                f"{head_pose[_YAW_INDEX]:7.1f} | "
                f"{number('unwrapped', 8, 2)} {number('step', 6, 2)} {number('slew', 6, 0)} | "
                f"{number('baseline', 8, 2)} {number('deviation', 6, 2)} | "
                f"{reading.get('direction', 0):+3d} "
                f"{reading.get('half_cycles', 0):2d}/{self.required_half_cycles:<2d}{marker} "
                f"{number('span', 5, 2)} | {reading.get('samples', 0):3d} "
                f"{number('primary_extent', 7, 2)} {number('cross_extent', 7, 2)} {ratio} | "
                f"{verdict}")

    @classmethod
    def replay(cls, head_poses, primary_index, cross_index, frame_period=_POLL_PERIOD,
               **parameters):
        """Score a recorded clip offline, exactly as the live loop would.

        `head_poses` is a sequence of head_pose tuples in capture order. Returns
        `(detected_at_seconds, blocking_gate)`, where a detection gives
        `(time, None)` and a miss gives `(None, reason)`.
        """
        tracker = cls(**parameters)
        for frame, head_pose in enumerate(head_poses):
            timestamp = frame * frame_period
            if tracker.update(timestamp, float(head_pose[primary_index]),
                              float(head_pose[cross_index])):
                return timestamp, None
        return None, tracker.last_reading.get("blocked_by")

    @classmethod
    def poll_until_detected(cls, perception_interface, termination_event, timeout,
                            primary_index, cross_index, label, debug=False,
                            continuous=False, **parameters):
        """Poll head perception until the oscillation is seen, the caller stops us, or we time out.

        With `debug`, prints one line per distinct frame showing every quantity the
        decision uses. With `continuous`, a detection clears the buffer and polling
        carries on until the timeout instead of returning, so gestures can be tried
        repeatedly in one run; the return value is then "did anything fire at all".
        """
        tracker = cls(**parameters)
        start_time = time.time()
        deadline = start_time + timeout
        last_head_pose = None
        detections = 0
        lines_printed = 0

        if debug:
            axis_names = {_ROLL_INDEX: "roll[3]", _PITCH_INDEX: "pitch[4]", _YAW_INDEX: "yaw[5]"}
            print(f"\n[{label}] {tracker.settings_summary(axis_names[primary_index], axis_names[cross_index])}")
            print(f"[{label}] continuous={continuous} timeout={timeout:.0f}s "
                  f"poll={1.0 / _POLL_PERIOD:.0f} Hz -- '^' marks a counted direction change")

        while time.time() < deadline and (termination_event is None or not termination_event.is_set()):
            head_perception_data = perception_interface.get_head_perception_data()
            # Sleep every iteration, including when no face was found -- the original
            # detector only slept on the success branch and so spun a core while the face
            # was missing.
            time.sleep(_POLL_PERIOD)

            if head_perception_data is None:
                # Transient dropout. Keep waiting: update() re-anchors by itself if the
                # gap outlasts _MAX_FRAME_GAP.
                continue

            head_pose = head_perception_data.get("head_pose")
            if head_pose is None or len(head_pose) <= max(primary_index, cross_index):
                continue

            # DECA reuses its last neck frame on some fallback paths, so the same reading
            # can be polled twice; a repeat is not new evidence.
            head_pose = tuple(float(value) for value in head_pose)
            if head_pose == last_head_pose:
                continue
            last_head_pose = head_pose

            detected = tracker.update(time.time(), head_pose[primary_index],
                                      head_pose[cross_index])

            if debug:
                if lines_printed % 25 == 0:
                    print(cls.DEBUG_HEADER)
                print(tracker.format_reading(head_pose, start_time, detected))
                lines_printed += 1

            if detected:
                detections += 1
                if not continuous:
                    return True
                if debug:
                    print(f"[{label}] detection #{detections} -- buffer cleared, still watching")
                tracker.reset()
                last_head_pose = None

        if debug:
            print(f"[{label}] finished after {time.time() - start_time:.1f}s "
                  f"with {detections} detection(s)")

        return detections > 0


def mouth_open(perception_interface, termination_event, timeout):
    """ Detect mouth open """
    threshold = 0.45

    def gesture_detector(perception_interface, termination_event, timeout, threshold):

        def euclidean_distance(p1, p2):
            """Calculate Euclidean distance between two points."""
            return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

        start_time = time.time()
        while time.time() - start_time < timeout and (termination_event is None or not termination_event.is_set()):
            head_perception_data = perception_interface.get_head_perception_data()
            if head_perception_data is None:
                continue
            else:
                time.sleep(0.1) # Maintain 10 Hz rate
            face_keypoints = head_perception_data["face_keypoints"]

            # Indices for mouth landmarks
            mouth_points = face_keypoints[48:68]

            # Calculate vertical distances
            A = euclidean_distance(mouth_points[2], mouth_points[10])  # 51, 59
            B = euclidean_distance(mouth_points[4], mouth_points[8])   # 53, 57

            # Calculate horizontal distance
            C = euclidean_distance(mouth_points[0], mouth_points[6])   # 49, 55

            mar = (A + B) / (2.0 * C)
            if mar > threshold:
                return True

        return False

    return gesture_detector(perception_interface, termination_event, timeout, threshold)


def head_nod(perception_interface, termination_event, timeout, debug=False, continuous=False):
    """Detect a head nod: about two nods' worth of pitch oscillation, with yaw held still.

    `debug` streams every quantity behind the decision; `continuous` keeps watching after
    a detection (clearing the buffer) instead of returning. Both default off, so the
    existing three-argument call sites are unchanged.
    """
    detected = _HeadOscillationTracker.poll_until_detected(
        perception_interface,
        termination_event,
        timeout,
        label="head nod",
        debug=debug,
        continuous=continuous,
        **NOD_PARAMETERS,
    )
    if detected and not debug:
        print("Head nod detected")
    return detected


def head_shake(perception_interface, termination_event, timeout, debug=False, continuous=False):
    """Detect a head shake: side-to-side yaw oscillation dominating pitch.

    See `head_nod` for `debug` / `continuous`.
    """
    detected = _HeadOscillationTracker.poll_until_detected(
        perception_interface,
        termination_event,
        timeout,
        label="head shake",
        debug=debug,
        continuous=continuous,
        **SHAKE_PARAMETERS,
    )
    if detected and not debug:
        print("Head shake detected")
    return detected
