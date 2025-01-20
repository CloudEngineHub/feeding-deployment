import time
import numpy as np
import time
import numpy as np
from gymnasium.spaces import Box

def headshake_user_generated(perception_interface, termination_event, timeout):
    head_shake_threshold = 0.0
    max_yaw_data_size = 800.0
    start_time = time.time()
    yaw_data = []
    direction_changes = 0
    while time.time() - start_time < timeout and (termination_event is None or not termination_event.is_set()):
        head_perception_data = perception_interface.get_head_perception_data()
        if head_perception_data is None:
            continue
        else:
            time.sleep(0.1) # Maintain 10 Hz rate
        head_pose = head_perception_data['head_pose']
        (head_x, head_y, head_z, head_roll, head_pitch, head_yaw) = head_pose
        yaw_data.append(head_yaw)
        if len(yaw_data) >= 3:
            if yaw_data[-2] - yaw_data[-3] > head_shake_threshold and yaw_data[-1] - yaw_data[-2] < -head_shake_threshold or (yaw_data[-3] - yaw_data[-2] < -head_shake_threshold and yaw_data[-2] - yaw_data[-1] > head_shake_threshold):
                direction_changes += 1
            if direction_changes >= 2:
                return True
        if len(yaw_data) > max_yaw_data_size:
            yaw_data.pop(0)
    return False
