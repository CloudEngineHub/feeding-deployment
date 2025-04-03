import time
import numpy as np
import numpy as np
from gymnasium.spaces import Box

def head_nod_1(perception_interface, termination_event, timeout):
    pitch_change_threshold = 0.0
    nod_count_required = 5.5
    sampling_rate = 0.1
    num_frames = 0
    nod_count = 0
    last_pitch = None
    direction = None
    accumulated_movement = 0
    while num_frames * sampling_rate < timeout and (termination_event is None or not termination_event.is_set()):
        num_frames += 1
        head_perception_data = perception_interface.get_head_perception_data()
        if head_perception_data is None:
            continue
        else:
            time.sleep(0.1) # Maintain 10 Hz rate
        head_pose = head_perception_data['head_pose']
        if head_pose is None:
            break
        (_, _, _, _, pitch, _) = head_pose
        if last_pitch is None:
            last_pitch = pitch
            continue
        pitch_diff = pitch - last_pitch
        if direction is None:
            if pitch_diff > pitch_change_threshold:
                direction = 'down'
                accumulated_movement = pitch_diff
            elif pitch_diff < -pitch_change_threshold:
                direction = 'up'
                accumulated_movement = pitch_diff
        else:
            accumulated_movement += pitch_diff
            if direction == 'down':
                if accumulated_movement < -pitch_change_threshold:
                    nod_count += 1
                    direction = 'up'
                    accumulated_movement = 0
                elif accumulated_movement > pitch_change_threshold:
                    accumulated_movement = pitch_change_threshold
            elif direction == 'up':
                if accumulated_movement > pitch_change_threshold:
                    nod_count += 1
                    direction = 'down'
                    accumulated_movement = 0
                elif accumulated_movement < -pitch_change_threshold:
                    accumulated_movement = -pitch_change_threshold
        last_pitch = pitch
    return nod_count >= nod_count_required
