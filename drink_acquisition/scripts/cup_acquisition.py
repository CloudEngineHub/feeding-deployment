import rospy

import sys

sys.path.append("../../../FLAIR/bite_acquisition/scripts")
sys.path.append("../../pybullet-cup-manipulation")

from robot_controller.kinova_controller import KinovaRobotController
# from cup_manipulation import generate_trajectory
from scene import (
    create_cup_manipulation_scene,
    CupManipulationSceneDescription,
)
import pybullet as p
from pybullet_helpers.gui import create_gui_connection
from geometry_msgs.msg import Pose, Point, Quaternion
from pybullet_helpers.geometry import Pose as PHPose
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import JointState
import numpy as np
import utils


if __name__ == "__main__":
    rospy.init_node("robot_controller", anonymous=True)
    robot_controller = KinovaRobotController()
    tf_utils = utils.TFUtils()

    # Move to a default start pose.
    x, y, z = 0.6, 0.05, 0.35
    default_tool_rot = Rotation.from_euler("xyz", [90, 0, 90], degrees=True)

    # Set by looking at the robot.
    quat = tuple(Rotation.from_euler("xyz", [0, 0, 90], degrees=True).as_quat())
    robot_base_pose = PHPose((0.0, 0.0, 0.0), quat)

    start_pose = Pose(
        position=Point(x, y, z), orientation=Quaternion(*default_tool_rot.as_quat())
    )
    robot_controller.move_to_pose(start_pose)

    # Get the initial joints.
    initial_joint_msg = rospy.wait_for_message("/robot_joint_states", JointState)
    assert initial_joint_msg.name == [
        "joint_1",
        "joint_2",
        "joint_3",
        "joint_4",
        "joint_5",
        "joint_6",
        "joint_7",
        "finger_joint",
    ]
    finger_val = initial_joint_msg.position[-1]
    initial_joints = tuple(initial_joint_msg.position[:7]) + (finger_val, finger_val)

    # Create the scene description.
    scene_description = CupManipulationSceneDescription(initial_joints=initial_joints)

    # Visualize the scene.
    physics_client_id = create_gui_connection(camera_yaw=180)
    scene = create_cup_manipulation_scene(physics_client_id, scene_description)
    while True:
        p.stepSimulation(physics_client_id)
        joint_msg = rospy.wait_for_message("/robot_joint_states", JointState)
        joint_positions = tuple(joint_msg.position[:7]) + (joint_msg.position[-1], joint_msg.position[-1])
        print("Joint positions:", joint_positions)
        scene.robot.set_joints(joint_positions)
