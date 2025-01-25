import time
import numpy as np
import numpy as np
from gymnasium.spaces import Box

def continuous_mouth_open(perception_interface, termination_event, timeout):
    mouth_open_threshold = 0.4
    required_continuous_seconds = 2.0

    def euclidean_distance(p1, p2):
        """Calculate Euclidean distance between two points."""
        return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5
    sampling_rate = 0.1
    num_frames = 0
    continuous_open_frames = 0
    required_frames = int(required_continuous_seconds / sampling_rate)
    while num_frames * sampling_rate < timeout and (termination_event is None or not termination_event.is_set()):
        num_frames += 1
        head_perception_data = perception_interface.get_head_perception_data()
        if head_perception_data is None:
            continue
        else:
            time.sleep(0.1) # Maintain 10 Hz rate
        face_keypoints = head_perception_data['face_keypoints']
        mouth_points = face_keypoints[48:68]
        A = euclidean_distance(mouth_points[2], mouth_points[10])
        B = euclidean_distance(mouth_points[4], mouth_points[8])
        C = euclidean_distance(mouth_points[0], mouth_points[6])
        mar = (A + B) / (2.0 * C)
        if mar > mouth_open_threshold:
            continuous_open_frames += 1
        else:
            continuous_open_frames = 0
        if continuous_open_frames >= required_frames:
            return True
    return False
