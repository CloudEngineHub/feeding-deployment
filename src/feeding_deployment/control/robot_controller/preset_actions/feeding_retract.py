'''
Preset: move the arm to scene_description.retract_pos — the deployment's rest
configuration. This is a DIFFERENT pose than retract.py (which uses its own
hardcoded joint vector).

retract_pos is read from the same scene config the executive uses (run.py default
scene_config="vention"). It is a plain "joint_positions" field, so the YAML "values"
are exactly what create_scene_description_from_config would put on
scene_description.retract_pos — read straight from the YAML here to avoid importing
the (pybullet-heavy) simulation package. Like the other preset actions, it waits on
the watchdog and talks to the arm over the RPC manager.
'''

import threading
import time
from pathlib import Path

import numpy as np
import yaml

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

# .../feeding_deployment/control/robot_controller/preset_actions/ -> .../feeding_deployment/simulation/configs/
SCENE_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent.parent / "simulation" / "configs" / "vention.yaml"


if __name__ == "__main__":

    assert ROSPY_IMPORTED, "ROS is required to run on the real robot"
    rospy.init_node("feeding_retract_action")

    with open(SCENE_CONFIG_PATH, "r") as f:
        scene_config = yaml.safe_load(f)
    retract_pos = scene_config["retract_pos"]["values"]

    # make sure watchdog is running
    print("Waiting for Watchdog status...")
    rospy.wait_for_message("/watchdog_status", Bool)
    print(f"Watchdog is running, moving to scene_description.retract_pos: {retract_pos}")

    # Register ArmInterface (no lambda needed on the client-side)
    ArmManager.register("ArmInterface")

    # Client setup
    manager = ArmManager(address=(NUC_HOSTNAME, ARM_RPC_PORT), authkey=RPC_AUTHKEY)
    manager.connect()

    # This will now use the single, shared instance of ArmInterface
    arm_interface = manager.ArmInterface()
    arm_interface.set_joint_position(retract_pos)
