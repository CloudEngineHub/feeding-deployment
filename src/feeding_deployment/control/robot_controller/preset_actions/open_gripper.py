'''
Entrypoint for controlling the robot arm on compute machine. Additionally runs two important threads:
1. A thread that checks no safety anomalies have occurred using the watchdog
2. A thread that publishes joint states to ROS
'''

import sys
import threading
import time
import numpy as np

try:
    import rospy
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Bool
    from geometry_msgs.msg import Pose
    # from netft_rdt_driver.srv import String_cmd
    ROSPY_IMPORTED = True
except ModuleNotFoundError as e:
    # print(f"ROS not imported: {e}")
    ROSPY_IMPORTED = False

from feeding_deployment.control.robot_controller.arm_interface import ArmInterface, ArmManager, NUC_HOSTNAME, ARM_RPC_PORT, RPC_AUTHKEY
from feeding_deployment.control.robot_controller.command_interface import KinovaCommand, JointTrajectoryCommand, JointCommand, CartesianCommand, OpenGripperCommand, CloseGripperCommand
# from feeding_deployment.safety.watchdog import WATCHDOG_MONITOR_FREQUENCY, PeekableQueue


if __name__ == "__main__":

    # Optional calibration arg: how far to open, 0.0 -> 1.0.
    #   1.0 = fully open (default, same as no arg)
    #   lower values open less, so the fingertips fit the utensil slot snugly
    #   instead of jamming against the extreme and stressing the fingers.
    # The Kinova gripper uses the opposite convention (0 = open, 1 = closed),
    # so we send gripper_pos = 1.0 - open_amount.
    open_amount = None
    if len(sys.argv) > 1:
        try:
            open_amount = float(sys.argv[1])
        except ValueError:
            print(f"Invalid open amount '{sys.argv[1]}': expected a number in [0.0, 1.0]")
            sys.exit(1)
        if not (0.0 <= open_amount <= 1.0):
            print(f"Open amount {open_amount} out of range: expected [0.0, 1.0]")
            sys.exit(1)

    assert ROSPY_IMPORTED, "ROS is required to run on the real robot"
    rospy.init_node("open_gripper_action")

    # make sure watchdog is running
    print("Waiting for Watchdog status...")
    rospy.wait_for_message("/watchdog_status", Bool)
    if open_amount is None:
        print("Watchdog is running, opening gripper (full)...")
    else:
        print(f"Watchdog is running, opening gripper to {open_amount:.3f} of full...")

    # Register ArmInterface (no lambda needed on the client-side)
    ArmManager.register("ArmInterface")

    # Client setup
    manager = ArmManager(address=(NUC_HOSTNAME, ARM_RPC_PORT), authkey=RPC_AUTHKEY)
    manager.connect()

    # This will now use the single, shared instance of ArmInterface
    arm_interface = manager.ArmInterface()
    if open_amount is None:
        arm_interface.open_gripper()
    else:
        arm_interface.set_gripper(1.0 - open_amount)