import time
import numpy as np
import numpy as np
from gymnasium.spaces import Box

def head_nod_1(perception_interface, termination_event, timeout):
    head_nod_threshold = 0.0
    required_direction_changes = 4.0
    direction_changes = 0
    min_head_pitch = float('inf')
    max_head_pitch = -float('inf')
    last_direction = None
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
        (_, _, _, _, head_pitch, _) = head_pose
        if head_pitch - min_head_pitch > head_nod_threshold:
            if last_direction != 'up':
                direction_changes += 1
                last_direction = 'up'
            min_head_pitch = head_pitch
            max_head_pitch = head_pitch
        elif max_head_pitch - head_pitch > head_nod_threshold:
            if last_direction != 'down':
                direction_changes += 1
                last_direction = 'down'
            max_head_pitch = head_pitch
            min_head_pitch = head_pitch
        min_head_pitch = min(head_pitch, min_head_pitch)
        max_head_pitch = max(head_pitch, max_head_pitch)
        if direction_changes >= required_direction_changes:
            return True
    return False

import numpy as np
from gymnasium.spaces import Box

def headshake(perception_interface, termination_event, timeout):
    head_shake_threshold = 0.0
    required_direction_changes = 3.0
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
        if last_direction is None:
            last_direction = head_yaw
            continue
        if head_yaw - min_head_yaw > head_shake_threshold:
            direction_changes += 1
            min_head_yaw = head_yaw
            max_head_yaw = head_yaw
        if max_head_yaw - head_yaw > head_shake_threshold:
            direction_changes += 1
            max_head_yaw = head_yaw
            min_head_yaw = head_yaw
        min_head_yaw = min(head_yaw, min_head_yaw)
        max_head_yaw = max(head_yaw, max_head_yaw)
        if direction_changes >= required_direction_changes:
            return True
    return False

import numpy as np
from gymnasium.spaces import Box

def head_tilt(perception_interface, termination_event, timeout):
    head_tilt_threshold = 0.0
    required_direction_changes = 6.0
    direction_changes = 0
    last_direction = None
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
        (_, _, _, head_roll, _, _) = head_pose
        if last_direction is None:
            last_direction = head_roll
            continue
        if head_roll - last_direction > head_tilt_threshold:
            direction_changes += 1
            last_direction = head_roll
        elif last_direction - head_roll > head_tilt_threshold:
            direction_changes += 1
            last_direction = head_roll
        if direction_changes >= required_direction_changes:
            return True
    return False

import numpy as np
from gymnasium.spaces import Box

def head_still(perception_interface, termination_event, timeout):
    stillness_threshold = 0.010000000149011612
    sampling_rate = 0.1
    num_frames = 0
    still_frames = 0
    last_head_pose = None
    while num_frames * sampling_rate < timeout and (termination_event is None or not termination_event.is_set()):
        num_frames += 1
        head_perception_data = perception_interface.get_head_perception_data()
        if head_perception_data is None:
            continue
        else:
            time.sleep(0.1) # Maintain 10 Hz rate
        head_pose = head_perception_data['head_pose']
        if last_head_pose is not None:
            pose_diff = [abs(head_pose[i] - last_head_pose[i]) for i in range(6)]
            if all((diff < stillness_threshold for diff in pose_diff)):
                still_frames += 1
            else:
                still_frames = 0
        else:
            still_frames = 0
        last_head_pose = head_pose
        if still_frames * sampling_rate >= 5:
            return True
    return False

import numpy as np
from gymnasium.spaces import Box

def mouth_open_1(perception_interface, termination_event, timeout):
    mouth_open_threshold = 0.5

    def euclidean_distance(p1, p2):
        """Calculate Euclidean distance between two points."""
        return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5
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
        mouth_points = face_keypoints[48:68]
        A = euclidean_distance(mouth_points[2], mouth_points[10])
        B = euclidean_distance(mouth_points[4], mouth_points[8])
        C = euclidean_distance(mouth_points[0], mouth_points[6])
        mar = (A + B) / (2.0 * C)
        if mar > mouth_open_threshold:
            return True
    return False

import numpy as np
from gymnasium.spaces import Box

def mouth_open_continuous(perception_interface, termination_event, timeout):
    mouth_open_threshold = 0.5
    required_continuous_seconds = 3.0

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

import numpy as np
from gymnasium.spaces import Box

def eye_blink(perception_interface, termination_event, timeout):
    eye_aspect_ratio_threshold = 0.1
    blink_duration_threshold = 0.30000000000000004
    required_blinks = 1.0

    def euclidean_distance(p1, p2):
        """Calculate Euclidean distance between two points."""
        return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

    def calculate_ear(eye_points):
        """Calculate the Eye Aspect Ratio (EAR) for an eye."""
        A = euclidean_distance(eye_points[1], eye_points[5])
        B = euclidean_distance(eye_points[2], eye_points[4])
        C = euclidean_distance(eye_points[0], eye_points[3])
        ear = (A + B) / (2.0 * C)
        return ear
    blink_count = 0
    blink_start_time = None
    blink_detected = False
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
        left_eye_points = face_keypoints[36:42]
        right_eye_points = face_keypoints[42:48]
        left_ear = calculate_ear(left_eye_points)
        right_ear = calculate_ear(right_eye_points)
        ear = (left_ear + right_ear) / 2.0
        if ear < eye_aspect_ratio_threshold:
            if not blink_detected:
                blink_start_time = num_frames * sampling_rate
                blink_detected = True
        elif blink_detected:
            blink_duration = num_frames * sampling_rate - blink_start_time
            if blink_duration < blink_duration_threshold:
                blink_count += 1
            blink_detected = False
        if blink_count >= required_blinks:
            return True
    return False

import numpy as np
from gymnasium.spaces import Box

def eye_brows_raised(perception_interface, termination_event, timeout):
    eyebrow_raise_threshold = 2.0
    required_raises = 3.0

    def euclidean_distance(p1, p2):
        """Calculate Euclidean distance between two points."""
        return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5
    raise_count = 0
    direction_changes = 0
    min_eyebrow_distance = float('inf')
    max_eyebrow_distance = -float('inf')
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
            break
        left_eyebrow_points = face_keypoints[17:22]
        right_eyebrow_points = face_keypoints[22:27]
        left_eye_points = face_keypoints[36:42]
        right_eye_points = face_keypoints[42:48]
        left_eyebrow_distance = sum((euclidean_distance(left_eyebrow_points[i], left_eye_points[i]) for i in range(5))) / 5
        right_eyebrow_distance = sum((euclidean_distance(right_eyebrow_points[i - 5], right_eye_points[i - 5]) for i in range(5))) / 5
        average_eyebrow_distance = (left_eyebrow_distance + right_eyebrow_distance) / 2
        if average_eyebrow_distance - min_eyebrow_distance > eyebrow_raise_threshold:
            direction_changes += 1
            min_eyebrow_distance = float('inf')
        if max_eyebrow_distance - average_eyebrow_distance > eyebrow_raise_threshold:
            direction_changes += 1
            max_eyebrow_distance = -float('inf')
        min_eyebrow_distance = min(average_eyebrow_distance, min_eyebrow_distance)
        max_eyebrow_distance = max(average_eyebrow_distance, max_eyebrow_distance)
        if direction_changes >= required_raises * 2:
            return True
    return False
