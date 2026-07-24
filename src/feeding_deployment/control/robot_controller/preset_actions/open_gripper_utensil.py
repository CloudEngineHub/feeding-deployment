'''
Preset: open the gripper to the calibrated utensil-grasp amount (the openness that
engages the utensil-mount slot). Matches what grasp_tool("utensil") commands in the
deployment. Like the other preset gripper actions, it waits on the watchdog and talks
to the arm over the RPC manager.
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
from feeding_deployment.control.robot_controller.command_interface import KinovaCommand, JointTrajectoryCommand, JointCommand, CartesianCommand, OpenGripperCommand, CloseGripperCommand, GRIPPER_GRASP_OPEN_AMOUNT, open_amount_to_gripper_pos


if __name__ == "__main__":

    # Defaults to the calibrated utensil-grasp amount so this stays in sync with the
    # deployment. Optional arg overrides it (0.0 -> 1.0) for re-calibration.
    open_amount = GRIPPER_GRASP_OPEN_AMOUNT
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
    rospy.init_node("open_gripper_utensil_action")

    # make sure watchdog is running
    print("Waiting for Watchdog status...")
    rospy.wait_for_message("/watchdog_status", Bool)
    print(f"Watchdog is running, opening gripper to utensil-grasp amount {open_amount:.3f}...")

    # Register ArmInterface (no lambda needed on the client-side)
    ArmManager.register("ArmInterface")

    # Client setup
    manager = ArmManager(address=(NUC_HOSTNAME, ARM_RPC_PORT), authkey=RPC_AUTHKEY)
    manager.connect()

    # This will now use the single, shared instance of ArmInterface
    arm_interface = manager.ArmInterface()
    arm_interface.set_gripper(open_amount_to_gripper_pos(open_amount))
