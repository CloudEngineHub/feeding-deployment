import time
import numpy as np
import numpy as np
from gymnasium.spaces import Box

def headshake(perception_interface, termination_event, timeout):
    head_shake_threshold = 9.0
    required_direction_changes = 1.5
    direction_changes = 0
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
            direction_changes += 1
            min_head_yaw = head_yaw
        if max_head_yaw - head_yaw > head_shake_threshold:
            direction_changes += 1
            max_head_yaw = head_yaw
        min_head_yaw = min(head_yaw, min_head_yaw)
        max_head_yaw = max(head_yaw, max_head_yaw)
        if direction_changes >= required_direction_changes:
            return True
    return False

import numpy as np
from gymnasium.spaces import Box

def twerking(perception_interface, termination_event, timeout):
    hip_roll_threshold = 9.0
    required_direction_changes = 1.0
    direction_changes = 0
    min_hip_roll = float('inf')
    max_hip_roll = -float('inf')
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
        (_, _, _, hip_roll, _, _) = head_pose
        if hip_roll - min_hip_roll > hip_roll_threshold:
            if last_direction != 'up':
                direction_changes += 1
                last_direction = 'up'
            min_hip_roll = hip_roll
        if max_hip_roll - hip_roll > hip_roll_threshold:
            if last_direction != 'down':
                direction_changes += 1
                last_direction = 'down'
            max_hip_roll = hip_roll
        min_hip_roll = min(hip_roll, min_hip_roll)
        max_hip_roll = max(hip_roll, max_hip_roll)
        if direction_changes >= required_direction_changes:
            return True
    return False
