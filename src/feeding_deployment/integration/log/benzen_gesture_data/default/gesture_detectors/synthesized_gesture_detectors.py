import time
import numpy as np
import numpy as np
from gymnasium.spaces import Box

def headnod(perception_interface, termination_event, timeout):
    head_nod_threshold = 18.0
    required_direction_changes = 2.0
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
        if max_head_pitch - head_pitch > head_nod_threshold:
            if last_direction != 'down':
                direction_changes += 1
                last_direction = 'down'
            max_head_pitch = head_pitch
        min_head_pitch = min(head_pitch, min_head_pitch)
        max_head_pitch = max(head_pitch, max_head_pitch)
        if direction_changes >= required_direction_changes:
            return True
    return False

import numpy as np
from gymnasium.spaces import Box

def headshake(perception_interface, termination_event, timeout):
    head_shake_threshold = 9.0
    required_direction_changes = 2.0
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
    required_direction_changes = 7.0
    direction_changes = 0
    min_head_roll = float('inf')
    max_head_roll = -float('inf')
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
        if head_roll - min_head_roll > head_tilt_threshold:
            if last_direction != 'right':
                direction_changes += 1
                last_direction = 'right'
            min_head_roll = head_roll
        if max_head_roll - head_roll > head_tilt_threshold:
            if last_direction != 'left':
                direction_changes += 1
                last_direction = 'left'
            max_head_roll = head_roll
        min_head_roll = min(head_roll, min_head_roll)
        max_head_roll = max(head_roll, max_head_roll)
        if direction_changes >= required_direction_changes:
            return True
    return False

import numpy as np
from gymnasium.spaces import Box

def continuous_head_still(perception_interface, termination_event, timeout):
    stillness_threshold = 0.5
    required_still_duration = 1.0
    still_duration = 0
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
            (_, _, _, roll, pitch, yaw) = head_pose
            (_, _, _, last_roll, last_pitch, last_yaw) = last_head_pose
            roll_change = abs(roll - last_roll)
            pitch_change = abs(pitch - last_pitch)
            yaw_change = abs(yaw - last_yaw)
            if roll_change < stillness_threshold and pitch_change < stillness_threshold and (yaw_change < stillness_threshold):
                still_duration += sampling_rate
            else:
                still_duration = 0
            if still_duration >= required_still_duration:
                return True
        last_head_pose = head_pose
    return False

import numpy as np
from gymnasium.spaces import Box

def mouth_open_1(perception_interface, termination_event, timeout):
    mouth_open_threshold = 0.6000000000000001

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
    required_continuous_frames = 9.0

    def euclidean_distance(p1, p2):
        """Calculate Euclidean distance between two points."""
        return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5
    sampling_rate = 0.1
    num_frames = 0
    continuous_open_frames = 0
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
        if continuous_open_frames >= required_continuous_frames:
            return True
    return False

import numpy as np
from gymnasium.spaces import Box

def eyes_blinking(perception_interface, termination_event, timeout):
    eye_aspect_ratio_threshold = 0.2
    blink_duration_threshold = 0.6000000000000001
    required_blinks = 1.5

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
    blink_in_progress = False
    blink_start_time = None
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
            if not blink_in_progress:
                blink_in_progress = True
                blink_start_time = num_frames * sampling_rate
        elif blink_in_progress:
            blink_duration = num_frames * sampling_rate - blink_start_time
            if blink_duration < blink_duration_threshold:
                blink_count += 1
            blink_in_progress = False
        if blink_count >= required_blinks:
            return True
    return False

import numpy as np
from gymnasium.spaces import Box

def eyebrows_raise(perception_interface, termination_event, timeout):
    eyebrow_raise_threshold = 3.0
    required_raises = 1.5
    raise_count = 0
    min_eyebrow_height = float('inf')
    max_eyebrow_height = -float('inf')
    direction_changes = 0

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
        if face_keypoints is None:
            break
        left_eyebrow_points = face_keypoints[17:22]
        right_eyebrow_points = face_keypoints[22:27]
        left_eye_points = face_keypoints[36:42]
        right_eye_points = face_keypoints[42:48]
        left_eyebrow_height = sum((euclidean_distance(left_eyebrow_points[i], left_eye_points[i]) for i in range(5))) / 5
        right_eyebrow_height = sum((euclidean_distance(right_eyebrow_points[i - 5], right_eye_points[i - 5]) for i in range(5))) / 5
        average_eyebrow_height = (left_eyebrow_height + right_eyebrow_height) / 2
        if average_eyebrow_height - min_eyebrow_height > eyebrow_raise_threshold:
            direction_changes += 1
            min_eyebrow_height = float('inf')
        if max_eyebrow_height - average_eyebrow_height > eyebrow_raise_threshold:
            direction_changes += 1
            max_eyebrow_height = -float('inf')
        min_eyebrow_height = min(average_eyebrow_height, min_eyebrow_height)
        max_eyebrow_height = max(average_eyebrow_height, max_eyebrow_height)
        if direction_changes >= required_raises * 2:
            return True
    return False

import numpy as np
from gymnasium.spaces import Box

def eyes_blinking_no_glasses(perception_interface, termination_event, timeout):
    eye_aspect_ratio_threshold = 0.1
    max_blink_duration = 0.6000000000000001
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
        left_eye_points = face_keypoints[36:42]
        right_eye_points = face_keypoints[42:48]
        left_ear = calculate_ear(left_eye_points)
        right_ear = calculate_ear(right_eye_points)
        ear = (left_ear + right_ear) / 2.0
        if ear < eye_aspect_ratio_threshold:
            if blink_start_time is None:
                blink_start_time = num_frames * sampling_rate
        elif blink_start_time is not None:
            blink_duration = num_frames * sampling_rate - blink_start_time
            if blink_duration <= max_blink_duration:
                blink_count += 1
            blink_start_time = None
        if blink_count >= required_blinks:
            return True
    return False

import numpy as np
from gymnasium.spaces import Box

def eyebrows_raised_no_glasses(perception_interface, termination_event, timeout):
    eyebrow_raise_threshold = 2.0
    required_raises = 1.7999999999999998
    raise_count = 0
    direction_changes = 0
    min_eyebrow_distance = float('inf')
    max_eyebrow_distance = -float('inf')

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
        if face_keypoints is None:
            break
        left_eyebrow_points = face_keypoints[17:22]
        right_eyebrow_points = face_keypoints[22:27]
        left_eye_points = face_keypoints[36:42]
        right_eye_points = face_keypoints[42:48]
        left_eyebrow_distance = sum((euclidean_distance(left_eyebrow_points[i], left_eye_points[i % len(left_eye_points)]) for i in range(len(left_eyebrow_points)))) / len(left_eyebrow_points)
        right_eyebrow_distance = sum((euclidean_distance(right_eyebrow_points[i], right_eye_points[i % len(right_eye_points)]) for i in range(len(right_eyebrow_points)))) / len(right_eyebrow_points)
        average_eyebrow_distance = (left_eyebrow_distance + right_eyebrow_distance) / 2
        if average_eyebrow_distance - min_eyebrow_distance > eyebrow_raise_threshold:
            direction_changes += 1
            min_eyebrow_distance = float('inf')
        if max_eyebrow_distance - average_eyebrow_distance > eyebrow_raise_threshold:
            direction_changes += 1
            max_eyebrow_distance = -float('inf')
        min_eyebrow_distance = min(average_eyebrow_distance, min_eyebrow_distance)
        max_eyebrow_distance = max(average_eyebrow_distance, max_eyebrow_distance)
        if direction_changes >= 2:
            raise_count += 1
            direction_changes = 0
            min_eyebrow_distance = float('inf')
            max_eyebrow_distance = -float('inf')
        if raise_count >= required_raises:
            return True
    return False
