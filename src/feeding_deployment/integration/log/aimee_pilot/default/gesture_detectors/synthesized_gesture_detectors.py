import time
import numpy as np
import numpy as np
from gymnasium.spaces import Box

def head_still(perception_interface, termination_event, timeout):
    stillness_threshold = 0.0
    sampling_rate = 0.1
    num_frames = 0
    still_frames = 0
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
        (_, _, _, roll, pitch, yaw) = head_pose
        if abs(roll) < stillness_threshold and abs(pitch) < stillness_threshold and (abs(yaw) < stillness_threshold):
            still_frames += 1
        else:
            still_frames = 0
        if still_frames * sampling_rate >= 5:
            return True
    return False
