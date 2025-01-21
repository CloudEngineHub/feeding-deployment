import time
import numpy as np
import numpy as np
from gymnasium.spaces import Box

def headshake(perception_interface, termination_event, timeout):
    head_shake_threshold = 9.0
    required_direction_changes = 1.0
    direction_changes = 0
    last_direction = None
    min_head_yaw = float('inf')
    max_head_yaw = -float('inf')
    sampling_rate = 0.1
    num_frames = 0
    while num_frames * sampling_rate < timeout and (termination_event is None or not termination_event.is_set()):
        num_frames += 1
        head_perception_data = perception_interface.get_head_perception_data()
        if head_perception_data is None:
            continue
        else:
            time.sleep(0.1) # Maintain 10 Hz rate
        head_pose = head_perception_data['head_pose']
        (_, _, _, _, _, head_yaw) = head_pose
        if head_yaw - min_head_yaw > head_shake_threshold:
            if last_direction != 'right':
                direction_changes += 1
                last_direction = 'right'
            min_head_yaw = float('inf')
        if max_head_yaw - head_yaw > head_shake_threshold:
            if last_direction != 'left':
                direction_changes += 1
                last_direction = 'left'
            max_head_yaw = -float('inf')
        min_head_yaw = min(head_yaw, min_head_yaw)
        max_head_yaw = max(head_yaw, max_head_yaw)
        if direction_changes >= required_direction_changes:
            return True
    return False

import numpy as np
from gymnasium.spaces import Box

def head_still(perception_interface, termination_event, timeout):
    stillness_threshold = 0.5
    stable_duration = 0
    last_head_pose = None
    sampling_rate = 0.1
    num_frames = 0
    while num_frames * sampling_rate < timeout and (termination_event is None or not termination_event.is_set()):
        num_frames += 1
        head_perception_data = perception_interface.get_head_perception_data()
        if head_perception_data is None:
            continue
        else:
            time.sleep(0.1) # Maintain 10 Hz rate
        head_pose = head_perception_data['head_pose']
        if last_head_pose is not None:
            pose_difference = sum((abs(a - b) for (a, b) in zip(head_pose, last_head_pose)))
            if pose_difference < stillness_threshold:
                stable_duration += sampling_rate
            else:
                stable_duration = 0
        last_head_pose = head_pose
        if stable_duration >= 5.0:
            return True
    return False

import numpy as np
from gymnasium.spaces import Box

def mouth_open_for_three_seconds(perception_interface, termination_event, timeout):
    mouth_open_threshold = 0.5

    def euclidean_distance(p1, p2):
        """Calculate Euclidean distance between two points."""
        return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5
    sampling_rate = 0.1
    num_frames = 0
    open_frames = 0
    required_open_frames = 3 / sampling_rate
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
            open_frames += 1
        else:
            open_frames = 0
        if open_frames >= required_open_frames:
            return True
    return False

import numpy as np
from gymnasium.spaces import Box

def i_am_saying_something(perception_interface, termination_event, timeout):
    mouth_movement_threshold = 0.0
    required_mouth_movements = 0.0

    def euclidean_distance(p1, p2):
        """Calculate Euclidean distance between two points."""
        return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5
    mouth_movements = 0
    last_mouth_aspect_ratio = None
    sampling_rate = 0.1
    num_frames = 0
    while num_frames * sampling_rate < timeout and (termination_event is None or not termination_event.is_set()):
        num_frames += 1
        head_perception_data = perception_interface.get_head_perception_data()
        if head_perception_data is None:
            continue
        else:
            time.sleep(0.1) # Maintain 10 Hz rate
        face_keypoints = head_perception_data['face_keypoints']
        if face_keypoints is None:
            continue
        mouth_points = face_keypoints[48:68]
        A = euclidean_distance(mouth_points[2], mouth_points[10])
        B = euclidean_distance(mouth_points[4], mouth_points[8])
        C = euclidean_distance(mouth_points[0], mouth_points[6])
        mar = (A + B) / (2.0 * C)
        if last_mouth_aspect_ratio is not None:
            if abs(mar - last_mouth_aspect_ratio) > mouth_movement_threshold:
                mouth_movements += 1
        last_mouth_aspect_ratio = mar
        if mouth_movements >= required_mouth_movements:
            return True
    return False
