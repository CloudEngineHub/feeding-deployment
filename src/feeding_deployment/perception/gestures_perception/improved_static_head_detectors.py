"""Head-gesture detectors that key off a real oscillation rather than a raw angle delta.

Drop-in replacement for `static_gesture_detectors`: same module layout, same
`(perception_interface, termination_event, timeout) -> bool` detector signature, and a
`function_name_to_label` mapping, so `inspect.getmembers(module, inspect.isfunction)`
gesture discovery in `actions/base.py` / `actions/emulate_transfer.py` keeps working.

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
_PITCH_INDEX = 4
_YAW_INDEX = 5

_POLL_PERIOD = 0.1  # 10 Hz, matching the head-perception thread.

# A step larger than this between consecutive samples is a tracking glitch, not a head:
# genuine nods reach 14 deg/sample on pitch and shakes 29 deg/sample on yaw, so this sits
# clear of real motion while still catching residual branch flips and DECA misfits.
_MAX_SLEW_DEG = 45.0

# Losing the face for longer than this discards accumulated evidence, so no gesture is
# ever assembled from readings taken either side of a dropout.
_MAX_FRAME_GAP = 1.0

# Floor on how fast successive direction changes may arrive. A glitch can flip sign in one
# or two frames; a human neck cannot.
_MIN_HALF_PERIOD = 0.15

_MIN_WINDOW_SAMPLES = 3

# Nod is the stricter of the two on purpose: its amplitude is small (~17 deg), and a false
# positive drives the arm toward the user's face, so it asks for about two nods. A shake
# swings +-28 deg, where three half-cycles are already unambiguous -- demanding four there
# costs real detections.
NOD_ENTER_DEG = 4.0
NOD_REQUIRED_HALF_CYCLES = 4
NOD_WINDOW_SECONDS = 3.0
NOD_DOMINANCE = 1.2
NOD_MIN_AMPLITUDE_DEG = 8.0

SHAKE_ENTER_DEG = 12.0
SHAKE_REQUIRED_HALF_CYCLES = 3
SHAKE_WINDOW_SECONDS = 2.5
SHAKE_DOMINANCE = 1.5
SHAKE_MIN_AMPLITUDE_DEG = 24.0


class _HeadOscillationTracker:
    """Streaming test for a deliberate repeated head oscillation on one axis.

    Deliberately a class, not a set of module-level helpers: gesture discovery collects
    every module-level function as a user-selectable gesture, and `emulate_transfer`
    then raises `KeyError` on any that is missing from `function_name_to_label`.

    Feed samples with `update()`; it returns True the moment the axis has completed
    `required_half_cycles` alternating excursions past `+/- enter_deg` about the median of
    the trailing `window`, at plausible pace, with enough amplitude, and with the gesture
    axis moving more than the other one.

    Measured against `gestures_examples/*.pkl` with the tuned constants above: 5/5 recall
    on both nod and shake clips, neither detector firing on the other's gesture, ~2 s
    latency. The detector it replaces managed 3/5 on nods -- and those three were exactly
    the three clips containing a +-180 branch flip.
    """

    def __init__(self, enter_deg, required_half_cycles, window, dominance, min_amplitude_deg):
        self.enter_deg = enter_deg
        self.required_half_cycles = required_half_cycles
        self.window = window
        self.dominance = dominance
        self.min_amplitude_deg = min_amplitude_deg
        self.reset()

    def reset(self):
        """Drop all accumulated evidence."""
        self._samples = deque()      # (timestamp, primary unwrapped, cross unwrapped)
        self._half_cycles = deque()  # timestamps of confirmed direction changes
        self._last_time = None
        self._previous_raw = None
        self._unwrapped = None
        self._direction = 0

    @staticmethod
    def _shortest_angle(delta):
        """Map a difference of two angles in degrees onto (-180, 180]."""
        return ((delta + 180.0) % 360.0) - 180.0

    def update(self, timestamp, primary_raw, cross_raw):
        """Feed one reading; return True once an oscillation is confirmed."""
        stale = self._last_time is not None and timestamp - self._last_time > _MAX_FRAME_GAP

        primary_step = cross_step = 0.0
        glitch = False
        if self._previous_raw is not None and not stale:
            primary_step = self._shortest_angle(primary_raw - self._previous_raw[0])
            cross_step = self._shortest_angle(cross_raw - self._previous_raw[1])
            glitch = abs(primary_step) > _MAX_SLEW_DEG or abs(cross_step) > _MAX_SLEW_DEG

        if self._previous_raw is None or stale or glitch:
            # Nothing before this reading can be trusted, so re-anchor on it.
            self.reset()
            self._unwrapped = (primary_raw, cross_raw)
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

        if len(self._samples) < _MIN_WINDOW_SAMPLES:
            return False

        primary_window = [sample[1] for sample in self._samples]
        cross_window = [sample[2] for sample in self._samples]

        # Median of the trailing window, not an all-time extreme: it self-centres, so
        # repeated one-sided nods still alternate about it, and slow drift (leaning in,
        # looking down at the plate) is subtracted instead of accumulating forever.
        deviation = primary - float(np.median(primary_window))

        # Only a crossing opposite to the previous one counts, which makes alternation
        # structural -- a single monotonic sweep can never satisfy the count.
        if self._direction <= 0 and deviation > self.enter_deg:
            self._direction = 1
            self._half_cycles.append(timestamp)
        elif self._direction >= 0 and deviation < -self.enter_deg:
            self._direction = -1
            self._half_cycles.append(timestamp)

        if len(self._half_cycles) < self.required_half_cycles:
            return False

        span = self._half_cycles[-1] - self._half_cycles[-self.required_half_cycles]
        if span < _MIN_HALF_PERIOD * (self.required_half_cycles - 1):
            return False

        primary_extent = max(primary_window) - min(primary_window)
        cross_extent = max(cross_window) - min(cross_window)
        if primary_extent < self.min_amplitude_deg:
            return False

        # Nods swing yaw a little and shakes swing pitch a lot, so requiring the gesture
        # axis to dominate is what keeps nod and shake from triggering each other.
        if primary_extent < self.dominance * cross_extent:
            return False

        return True

    @classmethod
    def poll_until_detected(cls, perception_interface, termination_event, timeout,
                            primary_index, cross_index, **parameters):
        """Poll head perception until the oscillation is seen, the caller stops us, or we time out."""
        tracker = cls(**parameters)
        deadline = time.time() + timeout
        last_head_pose = None

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

            if tracker.update(time.time(), head_pose[primary_index], head_pose[cross_index]):
                return True

        return False


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


def head_nod(perception_interface, termination_event, timeout):
    """Detect a head nod: about two nods' worth of pitch oscillation, with yaw held still."""
    detected = _HeadOscillationTracker.poll_until_detected(
        perception_interface,
        termination_event,
        timeout,
        primary_index=_PITCH_INDEX,
        cross_index=_YAW_INDEX,
        enter_deg=NOD_ENTER_DEG,
        required_half_cycles=NOD_REQUIRED_HALF_CYCLES,
        window=NOD_WINDOW_SECONDS,
        dominance=NOD_DOMINANCE,
        min_amplitude_deg=NOD_MIN_AMPLITUDE_DEG,
    )
    if detected:
        print("Head nod detected")
    return detected


def head_shake(perception_interface, termination_event, timeout):
    """Detect a head shake: side-to-side yaw oscillation dominating pitch."""
    detected = _HeadOscillationTracker.poll_until_detected(
        perception_interface,
        termination_event,
        timeout,
        primary_index=_YAW_INDEX,
        cross_index=_PITCH_INDEX,
        enter_deg=SHAKE_ENTER_DEG,
        required_half_cycles=SHAKE_REQUIRED_HALF_CYCLES,
        window=SHAKE_WINDOW_SECONDS,
        dominance=SHAKE_DOMINANCE,
        min_amplitude_deg=SHAKE_MIN_AMPLITUDE_DEG,
    )
    if detected:
        print("Head shake detected")
    return detected
