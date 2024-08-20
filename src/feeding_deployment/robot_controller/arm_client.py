# This RPC server allows other processes to communicate with the Kinova arm
# low-level controller, which runs in its own, dedicated real-time process.
#
# Note: Operations that are not time-sensitive should be run in a separate,
# non-real-time process to avoid interfering with the real-time low-level
# control and causing latency spikes.

import queue
import time
from multiprocessing.managers import BaseManager as MPBaseManager
import numpy as np
# from arm_controller import JointCompliantController
# from constants import RPC_AUTHKEY, ARM_RPC_PORT
RPC_AUTHKEY = b'secret-key'
ARM_RPC_PORT = 5000
# from ik_solver import IKSolver
from kinova import KinovaArm

class Arm:
    def __init__(self):
        self.arm = KinovaArm()
        # self.arm.set_joint_limits(speed_limits=(7 * (30,)), acceleration_limits=(7 * (80,)))
        self.command_queue = queue.Queue(1)
        self.controller = None

    def reset(self):
        # Go to home position
        self.arm.home()

        # switch to joint compliant mode
        self.arm.switch_to_joint_compliant_mode(self.command_queue)

    def get_state(self):
        arm_pos, arm_quat = self.arm.get_tool_pose()
        if arm_quat[3] < 0.0:  # Enforce quaternion uniqueness
            np.negative(arm_quat, out=arm_quat)
        state = {
            'arm_pos': arm_pos,
            'arm_quat': arm_quat,
            'gripper_pos': np.array([self.arm.gripper_pos]),
        }
        return state
    
    def execute_action(self, command_pos):
        print(f"Received command: {command_pos}")
        gripper_pos = 0
        self.command_queue.put((command_pos, gripper_pos))

    def close(self):
        self.arm.disconnect()

class ArmManager(MPBaseManager):
    pass

ArmManager.register('Arm', Arm)

if __name__ == '__main__':
    hostname = 'localhost'
    # manager = ArmManager(address=(hostname, ARM_RPC_PORT), authkey=RPC_AUTHKEY)
    # server = manager.get_server()
    # print(f'Arm manager server started at {hostname}:{ARM_RPC_PORT}')
    # server.serve_forever()
    import numpy as np
    # from constants import POLICY_CONTROL_PERIOD
    POLICY_CONTROL_PERIOD = 0.1
    manager = ArmManager(address=(hostname, ARM_RPC_PORT), authkey=RPC_AUTHKEY)
    manager.connect()
    arm = manager.Arm()
    try:
        arm.reset()
        print("Current Arm State:", arm.get_state())
        input("Press Enter to set arm pos...")
        next_pos = [0.0, 0.26179939, 3.14159265, -2.26892803, 0.0, 0.95993109, 1.8]
        arm.execute_action(next_pos)
        input("Press Enter to exit...")
        # while True:
            # time.sleep(1)   
        # for i in range(50):
        #     arm.execute_action({
        #         'arm_pos': np.array([0.135, 0.002, 0.211]),
        #         'arm_quat': np.array([0.706, 0.707, 0.029, 0.029]),
        #         'gripper_pos': np.zeros(1),
        #     })
        #     print(arm.get_state())
        #     state = arm.get_state()
        #     state['arm_quat'][0] = 3
        #     time.sleep(POLICY_CONTROL_PERIOD)
    finally:
        arm.close()
