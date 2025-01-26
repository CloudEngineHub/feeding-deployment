import time
import numpy as np
import numpy as np
from gymnasium.spaces import Box

def head_still(perception_interface, termination_event, timeout):
    stillness_threshold = 0.5
    required_stillness_duration = 2.5
    stillness_duration = 0
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
        if last_head_pose is None:
            last_head_pose = head_pose
            continue
        change_in_pose = sum((abs(a - b) for (a, b) in zip(head_pose, last_head_pose)))
        if change_in_pose < stillness_threshold:
            stillness_duration += sampling_rate
        else:
            stillness_duration = 0
        last_head_pose = head_pose
        if stillness_duration >= required_stillness_duration:
            return True
    return False

{"head_still": "Head still"}

