"""An interface for perception (robot joints, human head poses, etc.)."""

import threading
import time

import numpy as np
from pybullet_helpers.geometry import Pose
from pybullet_helpers.joint import JointPositions
from scipy.spatial.transform import Rotation as R
import json


try:
    import rospy
    from sensor_msgs.msg import JointState, CompressedImage
    from std_msgs.msg import String, Bool
    from visualization_msgs.msg import MarkerArray
    import tf2_ros
    from geometry_msgs.msg import TransformStamped
    from cv_bridge import CvBridge


    from feeding_deployment.head_perception.ros_wrapper import HeadPerceptionROSWrapper
except ModuleNotFoundError:
    pass

from feeding_deployment.robot_controller.arm_client import ArmInterfaceClient

class PerceptionInterface:
    """An interface for perception (robot joints, human head poses, etc.)."""

    def __init__(self, robot_interface: ArmInterfaceClient | None, record_goal_pose: bool = False, simulate_head_perception: bool = False) -> None:
        self._robot_interface = robot_interface

        # run head perception
        if robot_interface is None or simulate_head_perception:
            self._head_perception = None
        else:
            # self._head_perception = None
            self._head_perception = HeadPerceptionROSWrapper(record_goal_pose)
            
            # warm start head perception
            self._head_perception.set_tool("fork")
            for _ in range(10):
                self._head_perception.run_head_perception()

    def get_robot_joints(self) -> "JointState":
        """Get the current robot joint state."""
        joint_state_msg = rospy.wait_for_message("/robot_joint_states", JointState)
        q = np.array(joint_state_msg.position[:7])
        gripper_position = joint_state_msg.position[7]
        
        joint_state = q.tolist() + [
            gripper_position,
            gripper_position,
            gripper_position,
            gripper_position,
            -gripper_position,
            -gripper_position,
        ]
        return joint_state

    def get_camera_data(self):  # Rajat ToDo: Add return type
        return self._head_perception.get_camera_data()
    
    def set_head_perception_tool(self, tool: str) -> None:
        """Set the tool for head perception."""
        if self._head_perception is not None:
            self._head_perception.set_tool(tool)

    # Rajat ToDo: Change return type to Pose
    def get_head_perception_tool_tip_target_pose(self) -> np.ndarray:
        """Get a target of the forque from head perception."""
        if self._head_perception is not None:
            forque_target_transform = self._head_perception.run_head_perception()
        else:
            # forque_target_pose = Pose((-0.282, 0.540, 0.619), (-0.490, 0.510, 0.511, -0.489))

            forque_target_pose = np.eye(4)
            forque_target_pose[:3, 3] = [-0.282, 0.540, 0.619]
            forque_target_pose[:3, :3] = R.from_quat([-0.490, 0.510, 0.511, -0.489]).as_matrix()

            return forque_target_pose
        
    def get_tool_tip_pose(self) -> np.ndarray:
        raise NotImplementedError
    
    def get_tool_tip_pose_at_staging(self) -> np.ndarray:
        raise NotImplementedError
    
    def wait_for_user_continue_button(self) -> None:
        print("Waiting for transfer complete button press / ft sensor trigger ...")
        msg = rospy.wait_for_message("/transfer_complete", Bool)
        assert msg.data
        print("Received message, continuing ...")

