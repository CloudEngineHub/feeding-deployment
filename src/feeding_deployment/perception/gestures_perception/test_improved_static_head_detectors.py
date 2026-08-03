"""Offline checks for `improved_static_head_detectors` against the recorded clips.

Replays the labelled clips in `gestures_examples/*.pkl` through the real tracker at the
recorded 10 Hz, plus a few synthetic cases covering the failure modes that motivated the
rewrite. The module under test only needs `time`/`numpy`/`collections`, so this runs
without ROS, DECA, or a robot:

    python3 test_improved_static_head_detectors.py     # prints a summary table
    pytest test_improved_static_head_detectors.py
"""

import contextlib
import importlib.util
import inspect
import io
import math
import pickle
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_EXAMPLES = _HERE / "gestures_examples"
_FRAME_PERIOD = 0.1  # the recorded clips are sampled at 10 Hz


def _load_module_under_test():
    """Load the detector module straight from disk, bypassing package side effects."""
    path = _HERE / "improved_static_head_detectors.py"
    spec = importlib.util.spec_from_file_location("improved_static_head_detectors", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODULE = _load_module_under_test()

NOD_SETTINGS = dict(
    primary_index=MODULE._PITCH_INDEX,
    cross_index=MODULE._YAW_INDEX,
    enter_deg=MODULE.NOD_ENTER_DEG,
    required_half_cycles=MODULE.NOD_REQUIRED_HALF_CYCLES,
    window=MODULE.NOD_WINDOW_SECONDS,
    dominance=MODULE.NOD_DOMINANCE,
    min_amplitude_deg=MODULE.NOD_MIN_AMPLITUDE_DEG,
)

SHAKE_SETTINGS = dict(
    primary_index=MODULE._YAW_INDEX,
    cross_index=MODULE._PITCH_INDEX,
    enter_deg=MODULE.SHAKE_ENTER_DEG,
    required_half_cycles=MODULE.SHAKE_REQUIRED_HALF_CYCLES,
    window=MODULE.SHAKE_WINDOW_SECONDS,
    dominance=MODULE.SHAKE_DOMINANCE,
    min_amplitude_deg=MODULE.SHAKE_MIN_AMPLITUDE_DEG,
)

QUIET_GESTURES = ("blinking", "eyebrows_raised", "open_mouth")


def _clips(dataset, kind):
    """Head-pose arrays for one dataset ('positive' or 'negative' examples)."""
    with open(_EXAMPLES / f"{dataset}.pkl", "rb") as handle:
        data = pickle.load(handle)
    clips = []
    for example in data[f"{kind}_examples"]:
        poses = np.asarray(example["head_pose"], dtype=float)
        if poses.ndim == 2 and poses.shape[1] >= 6:
            clips.append(poses)
    return clips


def _replay(poses, primary_index, cross_index, **parameters):
    """Drive the tracker over one clip; return the detection time, or None."""
    tracker = MODULE._HeadOscillationTracker(**parameters)
    for frame, pose in enumerate(poses):
        timestamp = frame * _FRAME_PERIOD
        if tracker.update(timestamp, float(pose[primary_index]), float(pose[cross_index])):
            return timestamp
    return None


def _detections(dataset, kind, settings):
    settings = dict(settings)
    primary_index = settings.pop("primary_index")
    cross_index = settings.pop("cross_index")
    return [
        _replay(poses, primary_index, cross_index, **settings) is not None
        for poses in _clips(dataset, kind)
    ]


# --------------------------------------------------------------------------------------
# Recorded clips
# --------------------------------------------------------------------------------------

def test_nod_detects_every_recorded_nod():
    detected = _detections("head_nod", "positive", NOD_SETTINGS)
    assert all(detected), f"missed recorded nods: {detected}"


def test_shake_detects_every_recorded_shake():
    detected = _detections("shake_my_head_from_left_to_right", "positive", SHAKE_SETTINGS)
    assert all(detected), f"missed recorded shakes: {detected}"


def test_nod_does_not_fire_on_shakes():
    """A shake swings pitch 14-32 deg, so only cross-axis dominance separates the two."""
    detected = _detections("shake_my_head_from_left_to_right", "positive", NOD_SETTINGS)
    assert not any(detected), f"nod fired on shake clips: {detected}"


def test_shake_does_not_fire_on_nods():
    detected = _detections("head_nod", "positive", SHAKE_SETTINGS)
    assert not any(detected), f"shake fired on nod clips: {detected}"


def test_neither_detector_fires_on_other_facial_gestures():
    """Blinking, raised eyebrows and an open mouth are not head gestures."""
    for dataset in QUIET_GESTURES:
        for name, settings in (("nod", NOD_SETTINGS), ("shake", SHAKE_SETTINGS)):
            detected = _detections(dataset, "positive", settings)
            assert not any(detected), f"{name} fired on {dataset}: {detected}"


# --------------------------------------------------------------------------------------
# Synthetic cases: the failure modes that motivated the rewrite
# --------------------------------------------------------------------------------------

def _wrap_into_range(angle):
    """Put an angle on (-180, 180], the way the euler decomposition reports it."""
    return ((angle + 180.0) % 360.0) - 180.0


def _nodding_pitch(timestamp, centre=178.0, amplitude=8.0, hz=1.2):
    """A 16 deg peak-to-peak nod centred near the +-180 seam, so readings wrap mid-gesture."""
    return _wrap_into_range(centre + amplitude * math.sin(2.0 * math.pi * hz * timestamp))


def test_branch_flip_alone_is_not_a_nod():
    """The root cause: a still head whose pitch flips across +-180 emits ~358 deg steps."""
    poses = []
    for frame in range(200):
        pitch = 179.6 if frame % 2 else -179.6  # ~0.8 deg of real motion
        poses.append([0.0, 0.0, 0.0, 0.0, pitch, 1.0])
    assert _replay(np.asarray(poses), **_positional(NOD_SETTINGS)) is None


def test_monotonic_sweep_is_not_a_nod():
    """Leaning in or looking down is one-way motion, however large."""
    poses = [[0.0, 0.0, 0.0, 0.0, -40.0 + 0.8 * frame, 1.0] for frame in range(100)]
    assert _replay(np.asarray(poses), **_positional(NOD_SETTINGS)) is None


def test_synthetic_nod_is_detected_across_the_seam():
    poses = [
        [0.0, 0.0, 0.0, 0.0, _nodding_pitch(frame * _FRAME_PERIOD), 1.0]
        for frame in range(100)
    ]
    detected_at = _replay(np.asarray(poses), **_positional(NOD_SETTINGS))
    assert detected_at is not None
    assert detected_at < 3.0, f"nod took {detected_at:.1f}s to confirm"


def test_evidence_does_not_span_a_long_dropout():
    """Half a nod, a 5 s gap, then the other half must not add up to a gesture."""
    tracker = MODULE._HeadOscillationTracker(
        **{k: v for k, v in NOD_SETTINGS.items() if k not in ("primary_index", "cross_index")}
    )
    fired = False
    for frame in range(10):
        fired |= tracker.update(frame * _FRAME_PERIOD, _nodding_pitch(frame * _FRAME_PERIOD), 1.0)
    for frame in range(10):
        timestamp = 5.0 + frame * _FRAME_PERIOD
        fired |= tracker.update(timestamp, _nodding_pitch(timestamp), 1.0)
    assert not fired


def _positional(settings):
    """Turn a settings dict into `_replay` kwargs."""
    settings = dict(settings)
    return dict(
        primary_index=settings.pop("primary_index"),
        cross_index=settings.pop("cross_index"),
        **settings,
    )


# --------------------------------------------------------------------------------------
# The polling loop itself
# --------------------------------------------------------------------------------------

class _FakeClock:
    """Stand-in for `time`, so the polling loop runs at fake-time speed."""

    def __init__(self):
        self.now = 0.0

    def time(self):
        return self.now

    def sleep(self, duration):
        self.now += duration


class _FakePerception:
    """Replays a scripted list of head-perception payloads, one per poll."""

    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.polls = 0

    def get_head_perception_data(self):
        if self.polls >= len(self.payloads):
            return None
        payload = self.payloads[self.polls]
        self.polls += 1
        return payload


def _scripted_nod(frames=120):
    """A nod, peppered with dropouts and repeated frames the way DECA delivers them."""
    payloads = []
    for frame in range(frames):
        if frame % 17 == 0:
            payloads.append(None)                          # transient face loss
            continue
        source = frame - 1 if frame % 11 == 0 else frame    # DECA reusing its last frame
        pitch = _nodding_pitch(source * _FRAME_PERIOD)
        payloads.append({"head_pose": (0.0, 0.0, 0.0, 0.0, pitch, 1.0)})
    return payloads


def _run_loop(payloads, detector, timeout=20.0, perception=None, **kwargs):
    real_time = MODULE.time
    MODULE.time = _FakeClock()
    try:
        return detector(perception or _FakePerception(payloads), None, timeout, **kwargs)
    finally:
        MODULE.time = real_time


def test_polling_loop_detects_a_nod_through_dropouts_and_repeats():
    assert _run_loop(_scripted_nod(), MODULE.head_nod) is True


def test_polling_loop_times_out_on_a_still_head():
    payloads = [
        {"head_pose": (0.0, 0.0, 0.0, 0.0, 179.6 if frame % 2 else -179.6, 1.0)}
        for frame in range(400)
    ]
    assert _run_loop(payloads, MODULE.head_nod) is False


def test_polling_loop_honours_the_termination_event():
    class _Stopped:
        def is_set(self):
            return True

    real_time = MODULE.time
    MODULE.time = _FakeClock()
    try:
        perception = _FakePerception(_scripted_nod())
        assert MODULE.head_nod(perception, _Stopped(), 20.0) is False
        assert perception.polls == 0, "returned only after polling perception"
    finally:
        MODULE.time = real_time


# --------------------------------------------------------------------------------------
# Debug streaming and continuous mode
# --------------------------------------------------------------------------------------

def test_detectors_stay_callable_with_three_positional_arguments():
    """Gesture discovery calls detectors as fn(perception, event, timeout) and nothing more."""
    for detector in (MODULE.head_nod, MODULE.head_shake, MODULE.mouth_open):
        parameters = list(inspect.signature(detector).parameters.values())
        assert [p.name for p in parameters[:3]] == ["perception_interface", "termination_event", "timeout"]
        assert all(p.default is not inspect.Parameter.empty for p in parameters[3:]), \
            f"{detector.__name__} grew a required argument"


def test_continuous_mode_keeps_watching_after_a_detection():
    payloads = _scripted_nod(frames=400)
    perception = _FakePerception(payloads)
    assert _run_loop(payloads, MODULE.head_nod, timeout=60.0,
                     perception=perception, continuous=True) is True
    assert perception.polls == len(payloads), \
        "continuous mode stopped early instead of watching the whole stream"


def test_default_mode_returns_on_the_first_detection():
    payloads = _scripted_nod(frames=400)
    perception = _FakePerception(payloads)
    assert _run_loop(payloads, MODULE.head_nod, timeout=60.0, perception=perception) is True
    assert perception.polls < len(payloads), \
        "default mode should return at the first detection, not keep polling"


def test_debug_stream_prints_the_columns_and_the_blocking_gate():
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        assert _run_loop(_scripted_nod(), MODULE.head_nod, debug=True) is True
    output = stream.getvalue()
    for column in ("roll", "pitch", "yaw", "unwrap", "step", "deg/s", "base", "dev",
                   "p2p_pri", "p2p_crs", "ratio", "verdict"):
        assert column in output, f"debug stream is missing the {column!r} column"
    assert "half-cycles" in output, "debug stream never named the blocking gate"
    assert "*** DETECTED ***" in output
    assert "enter +-4.0 deg" in output, "debug stream did not report the thresholds in use"


def test_debug_stream_survives_dropouts_and_reports_gap_resets():
    """A long face loss must print a GAP RESET rather than blow up on missing fields."""
    payloads = ([{"head_pose": (0.0, 0.0, 0.0, 5.0, _nodding_pitch(i * _FRAME_PERIOD), 1.0)}
                 for i in range(5)]
                + [None] * 30
                + [{"head_pose": (0.0, 0.0, 0.0, 5.0, _nodding_pitch(i * _FRAME_PERIOD), 1.0)}
                   for i in range(5, 60)])
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        _run_loop(payloads, MODULE.head_nod, debug=True, continuous=True)
    assert "GAP RESET" in stream.getvalue()


def test_slew_gate_is_frame_rate_independent():
    """The velocity limit is deg/s, so slower frames must not turn real motion into a glitch."""
    settings = {key: value for key, value in NOD_SETTINGS.items()
                if key not in ("primary_index", "cross_index")}

    # 60 deg between frames 0.1 s apart is 600 deg/s -- beyond any neck, so a glitch.
    fast = MODULE._HeadOscillationTracker(**settings)
    for timestamp, pitch in ((0.0, 0.0), (0.1, 5.0), (0.2, 65.0)):
        fast.update(timestamp, pitch, 0.0)
    assert fast.last_reading["blocked_by"].startswith("SLEW RESET"), fast.last_reading

    # The same 60 deg across 0.2 s is 300 deg/s -- a fast shake, and must be kept.
    slow = MODULE._HeadOscillationTracker(**settings)
    for timestamp, pitch in ((0.0, 0.0), (0.2, 5.0), (0.4, 65.0)):
        slow.update(timestamp, pitch, 0.0)
    assert not str(slow.last_reading.get("blocked_by")).startswith("SLEW RESET"), slow.last_reading


def test_anchoring_frame_never_reports_a_detection():
    """After a reset the window is empty, so no gate combination may read as a hit."""
    settings = {key: value for key, value in NOD_SETTINGS.items()
                if key not in ("primary_index", "cross_index")}
    tracker = MODULE._HeadOscillationTracker(**settings)
    assert tracker.update(0.0, 178.0, 1.0) is False
    assert tracker.update(9.0, 178.0, 1.0) is False  # gap reset
    assert tracker.last_reading["blocked_by"].startswith("GAP RESET")


# --------------------------------------------------------------------------------------

def _summarise():
    datasets = ["head_nod", "shake_my_head_from_left_to_right", "talking",
                "head_still_atleast_three_secs", "look_at_robot_atleast_three_secs",
                "blinking", "open_mouth", "eyebrows_raised"]
    print(f"{'dataset':34s} {'kind':4s} {'clips':>5s} {'nod fires':>10s} {'shake fires':>12s}")
    for dataset in datasets:
        for kind in ("positive", "negative"):
            nod = _detections(dataset, kind, NOD_SETTINGS)
            shake = _detections(dataset, kind, SHAKE_SETTINGS)
            print(f"{dataset:34s} {kind[:3]:4s} {len(nod):5d} "
                  f"{sum(nod):>4d}/{len(nod):<5d} {sum(shake):>6d}/{len(shake):<5d}")

    print("\nDetection latency on the target gesture:")
    for dataset, settings, label in (("head_nod", NOD_SETTINGS, "nod"),
                                     ("shake_my_head_from_left_to_right", SHAKE_SETTINGS, "shake")):
        kwargs = _positional(settings)
        times = [_replay(poses, **kwargs) for poses in _clips(dataset, "positive")]
        hit = [t for t in times if t is not None]
        print(f"  {label:6s} recall {len(hit)}/{len(times)}  "
              f"median {np.median(hit):.1f}s  max {max(hit):.1f}s")


if __name__ == "__main__":
    _summarise()
    print()
    failures = 0
    for name, test in sorted(globals().items()):
        if name.startswith("test_") and callable(test):
            try:
                test()
                print(f"  PASS  {name}")
            except AssertionError as error:
                failures += 1
                print(f"  FAIL  {name}: {error}")
    print(f"\n{failures} failure(s)")
