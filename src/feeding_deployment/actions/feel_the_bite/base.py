
import abc
import time
import numpy as np

from feeding_deployment.simulation.simulator import FeedingDeploymentPyBulletSimulator
from feeding_deployment.control.robot_controller.arm_client import ArmInterfaceClient
from feeding_deployment.interfaces.perception_interface import PerceptionInterface
from feeding_deployment.interfaces.rviz_interface import RVizInterface
from feeding_deployment.control.robot_controller.command_interface import CartesianCommand
from feeding_deployment.interfaces.web_interface import WebInterfaceTakeoverInterrupt

try:
    import rospy
    from std_msgs.msg import Bool
except ModuleNotFoundError:
    ROSPY_IMPORTED = False

class Transfer(abc.ABC):
    """ Base class for transfer actions. """

    def __init__(self, sim : FeedingDeploymentPyBulletSimulator, robot_interface: ArmInterfaceClient, perception_interface: PerceptionInterface, rviz_interface: RVizInterface, no_waits=False, web_interface=None):

        self.sim = sim
        self.robot_interface = robot_interface
        self.perception_interface = perception_interface
        self.rviz_interface = rviz_interface
        self.no_waits = no_waits
        self.web_interface = web_interface

        if self.robot_interface is not None:
            self.set_filter_noisy_readings_pub = rospy.Publisher('/head_perception/set_filter_noisy_readings', Bool, queue_size=1)

    def set_tool(self, tool):
        self.tool = tool

    def wait_for_head_perception_data(self, poll_hz: float = 10.0) -> dict:
        """Block until head perception returns a valid reading, then return it.

        A single DECA frame can come back None (no face detected, fewer than 4
        landmarks with valid depth, or -- while noisy-reading filtering is on --
        a reading rejected as noisy). The old code dereferenced that None
        (head_perception_data["tool_tip_target_pose"]); the resulting TypeError
        propagated uncaught out of the synchronous behavior tree and killed the
        executive, freezing the system until Ctrl-C. Instead, poll until a valid
        reading arrives.

        The wait is intentionally unbounded: there is no autonomous recovery from
        "no face is visible", so the operator resolves it either by getting a face
        back in view or by pressing arm-control (Take Over). We peek the takeover
        event each iteration and raise WebInterfaceTakeoverInterrupt so that button
        works during this wait; execute_action (base.py) owns the consume + teleop
        recovery, matching the rest of the codebase.
        """
        period = 1.0 / poll_hz
        waited = False
        while True:
            if self.web_interface is not None and self.web_interface.takeover_event.is_set():
                raise WebInterfaceTakeoverInterrupt()
            head_perception_data = self.perception_interface.get_head_perception_data()
            if head_perception_data is not None:
                if waited:
                    print("Head perception recovered; continuing transfer.")
                return head_perception_data
            if not waited:
                print("Waiting for a valid head-perception reading (no face detected yet) ...")
                waited = True
            time.sleep(period)

    def get_tip_wrist_transform(self):

        if self.tool == "fork":
            wrist_to_tip = self.sim.scene_description.tool_frame_to_utensil_tip
        elif self.tool == "drink":
            wrist_to_tip = self.sim.scene_description.tool_frame_to_drink_tip
        elif self.tool == "wipe":
            wrist_to_tip = self.sim.scene_description.tool_frame_to_wipe_tip
        else:
            raise ValueError("Tool not recognized")
        
        tip_to_wrist = np.linalg.inv(wrist_to_tip.to_matrix())
        return tip_to_wrist

    def move_to_ee_pose(self, pose):

        if self.robot_interface is None:
            plan = self.sim.plan_to_ee_pose(pose)
            self.sim.visualize_plan(plan)
        else:
            self.robot_interface.execute_command(CartesianCommand(pos=pose.position, quat=pose.orientation))

    @abc.abstractmethod
    def move_to_transfer_state(self, outside_mouth_distance, maintain_position_at_goal = False):
        """Move robot to the transfer state."""

    @abc.abstractmethod
    def move_to_before_transfer_state(self):
        """Move robot to the state before transfer."""

    