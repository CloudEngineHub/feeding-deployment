'''
Entrypoint for controlling the robot arm on compute machine. Additionally runs two important threads:
1. A thread that checks no safety anomalies have occurred using the watchdog
2. A thread that publishes joint states to ROS
'''

import threading
import time
import types
import numpy as np
import yaml
from pathlib import Path

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

def load_robot_config(config_path: str) -> types.SimpleNamespace:
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)
    parsed = {}
    for key, entry in raw.items():
        if isinstance(entry, dict):
            parsed[key] = entry["values"]
    return types.SimpleNamespace(**parsed)

from feeding_deployment.control.robot_controller.arm_interface import ArmInterface, ArmManager, NUC_HOSTNAME, ARM_RPC_PORT, RPC_AUTHKEY
from feeding_deployment.control.robot_controller.command_interface import KinovaCommand, JointTrajectoryCommand, CartesianTrajectoryCommand, JointCommand, CartesianCommand, OpenGripperCommand, CloseGripperCommand, SetGripperCommand
# from feeding_deployment.safety.watchdog import WATCHDOG_MONITOR_FREQUENCY, PeekableQueue
from feeding_deployment.safety.collision_threshold import collision_threshold

class ArmInterfaceClient:
    def __init__(self):

        assert ROSPY_IMPORTED, "ROS is required to run on the real robot"

        # make sure watchdog is running
        print("Waiting for Watchdog status...")
        rospy.wait_for_message("/watchdog_status", Bool)
        print("Watchdog is running, continuing...")

        # Register ArmInterface (no lambda needed on the client-side)
        ArmManager.register("ArmInterface")

        # Client setup
        self.manager = ArmManager(address=(NUC_HOSTNAME, ARM_RPC_PORT), authkey=RPC_AUTHKEY)
        self.manager.connect()

        # This will now use the single, shared instance of ArmInterface
        self._arm_interface = self.manager.ArmInterface()
        self.in_compliant_mode = False

    def switch_to_task_compliant_mode(self):
        assert not self.in_compliant_mode, "Already in compliant mode"
        self._arm_interface.switch_to_task_compliant_mode()
        self.in_compliant_mode = True

    def switch_to_joint_compliant_mode(self):
        assert not self.in_compliant_mode, "Already in compliant mode"
        self._arm_interface.switch_to_joint_compliant_mode()
        self.in_compliant_mode = True

    def switch_out_of_compliant_mode(self):
        assert self.in_compliant_mode, "Not in compliant mode"
        # time.sleep(2.0) # Wait for the arm to settle
        self._arm_interface.switch_out_of_compliant_mode()
        self.in_compliant_mode = False

    def get_state(self):
        return self._arm_interface.get_state()

    def stop_action(self):
        """Abort the current arm action without latching emergency stop.

        NOTE: this is issued on the same RPC connection as execute_command. If a
        blocking move is in flight on this connection, the manager may serialize
        this call behind it. See TELEOP_INTEGRATION.md (Default -> Stop).
        """
        return self._arm_interface.stop_action()

    def get_speed(self):
        return self._arm_interface.get_speed()
    
    def set_speed(self, speed: str):
        assert not self.in_compliant_mode, "Cannot set speed in compliant mode"
        assert speed in ["low", "medium", "high"], "Speed must be one of 'low', 'medium', 'high'"
        self._arm_interface.set_speed(speed)
        time.sleep(1.0) # Make sure the arm has time to change speed

    def set_tool(self, tool: str):
        assert not self.in_compliant_mode, "Cannot set tool in compliant mode"
        self._arm_interface.set_tool(tool)

    def execute_command(self, cmd: KinovaCommand) -> None:

        # if not self.in_compliant_mode:
            # input("Press enter to execute command...")

        if cmd.__class__.__name__ == "JointTrajectoryCommand":
            return self._arm_interface.set_joint_trajectory(cmd.traj)
        
        if cmd.__class__.__name__ == "CartesianTrajectoryCommand":
            return self._arm_interface.set_cartesian_trajectory(cmd.traj)

        if cmd.__class__.__name__ == "JointCommand":
            if self.in_compliant_mode:
                return self._arm_interface.compliant_set_joint_position(cmd.pos)
            else:
                joint_command_pos = cmd.pos
                if isinstance(joint_command_pos, np.ndarray):
                    joint_command_pos = joint_command_pos.tolist()  # Convert to a list if it's a NumPy array
                return self._arm_interface.set_joint_position(joint_command_pos)

        if cmd.__class__.__name__ == "CartesianCommand":
            if self.in_compliant_mode:
                return self._arm_interface.compliant_set_ee_pose(cmd.pos, cmd.quat)
            else:
                return self._arm_interface.set_ee_pose(cmd.pos, cmd.quat, cmd.soft_stop)

        if cmd.__class__.__name__ == "OpenGripperCommand":
            return self._arm_interface.open_gripper()

        if cmd.__class__.__name__ == "CloseGripperCommand":
            return self._arm_interface.close_gripper()

        if cmd.__class__.__name__ == "SetGripperCommand":
            return self._arm_interface.set_gripper(cmd.pos)

        raise NotImplementedError(f"Unrecognized command: {cmd}")

if __name__ == "__main__":

    rospy.init_node("arm_interface_client", anonymous=True)
    arm_client_interface = ArmInterfaceClient()

    _config_path = Path(__file__).parent.parent.parent / "simulation" / "configs" / "vention.yaml"
    config = load_robot_config(str(_config_path))

    run_commands = input("Press 'y' to run commands")

    if run_commands != "y":
        exit()

    # print current state
    state = arm_client_interface.get_state()

    def get_state():
        state = arm_client_interface.get_state()
        print("Current joint positions:", ", ".join([str(x) for x in state["position"]]))
        print("Current end-effector pose:", ", ".join([str(x) for x in state["ee_pos"]]))
        return state["ee_pos"], state["position"]
    
    ee_pose, joint_positions = get_state()

    # arm_client_interface.execute_command(JointCommand(config.above_plate_pos))

    # inside_pose = [0.07883421331644058, -0.26393580436706543, 0.0408918559551239, -0.5102654597504978, 0.4783280635647934, 0.5001110873986087, -0.5106077990522174]
    # above_pose = [0.07883421331644058, -0.26393580436706543, 0.2408918559551239, -0.5102654597504978, 0.4783280635647934, 0.5001110873986087, -0.5106077990522174]

    # inside_pose = [0.08389504253864288, -0.2624289393424988, 0.03942343592643738, -0.5102503363699538, 0.4782626801095757, 0.5001531665865582, -0.5106429408130433]
    # above_pose = [0.08389504253864288, -0.2624289393424988, 0.23942343592643738, -0.5102503363699538, 0.4782626801095757, 0.5001531665865582, -0.5106429408130433]

    # arm_client_interface.execute_command(CartesianCommand(above_pose[:3], above_pose[3:]))
    # arm_client_interface.execute_command(CartesianCommand(inside_pose[:3], inside_pose[3:]))

    # for i in range(5):
    #     arm_client_interface.execute_command(CartesianCommand(above_pose[:3], above_pose[3:]))
    #     arm_client_interface.execute_command(CartesianCommand(inside_pose[:3], inside_pose[3:]))
    #     arm_client_interface.execute_command(CloseGripperCommand())
    #     arm_client_interface.execute_command(CartesianCommand(above_pose[:3], above_pose[3:]))
    #     arm_client_interface.execute_command(CartesianCommand(inside_pose[:3], inside_pose[3:]))
    #     arm_client_interface.execute_command(OpenGripperCommand())

    # arm_client_interface.execute_command(CartesianCommand(inside_pose[:3], inside_pose[3:]))

    # arm_client_interface.execute_command(JointCommand([3.077154275222018, -1.8775424169435313, 1.185893516029325, -1.6186986053221517, -0.29601621411670553, -1.4471659974068016, -0.3531266384290186]))
    # arm_client_interface.execute_command(JointCommand(config.above_plate_holder_pos))
    # arm_client_interface.execute_command(CartesianCommand(config.inside_plate_holder_pose[:3], config.inside_plate_holder_pose[3:]))

    # arm_client_interface.execute_command(JointCommand(config.before_transfer_pos))
    # arm_client_interface.execute_command(JointCommand(config.above_drink_holder_pos))
    # arm_client_interface.execute_command(JointCommand(config.before_transfer_pos))
    # arm_client_interface.execute_command(JointCommand(config.outside_drink_holder_pos))
    # arm_client_interface.execute_command(JointCommand(config.intermediate_drink_holder_pos))
    # arm_client_interface.execute_command(JointCommand(config.retract_pos))
    # arm_client_interface.execute_command(JointCommand(config.retract_pos))
    # arm_client_interface.execute_command(JointCommand(config.outside_drink_holder_pos))
    # arm_client_interface.execute_command(JointCommand(config.above_drink_holder_pos))
    # arm_client_interface.execute_command(JointCommand(config.intermediate_drink_holder_pos))
    # arm_client_interface.execute_command(JointCommand(config.before_transfer_pos))

    print("Moving to left back retract position")
    arm_client_interface.execute_command(JointCommand(config.left_back_retract_pos))
    arm_client_interface.execute_command(JointCommand(config.back_retract_pos))
    arm_client_interface.execute_command(JointCommand(config.right_back_retract_pos))
    arm_client_interface.execute_command(JointCommand(config.behind_back_retract_pos))
    arm_client_interface.execute_command(JointCommand(config.left_back_retract_pos))

    # arm_client_interface.execute_command(JointCommand(config.above_drink_holder_pos))
    # arm_client_interface.execute_command(JointCommand(config.before_transfer_pos))
    # arm_client_interface.execute_command(JointCommand(config.home_pos))

    # arm_client_interface.execute_command(CartesianCommand(config.above_drink_holder_pose[:3], config.above_drink_holder_pose[3:]))
    # arm_client_interface.execute_command(CartesianCommand(config.inside_drink_holder_pose[:3], config.inside_drink_holder_pose[3:]))
    # arm_client_interface.execute_command(CloseGripperCommand())
    # arm_client_interface.execute_command(CartesianCommand(config.below_drink_holder_pose[:3], config.below_drink_holder_pose[3:]))
    # arm_client_interface.execute_command(CartesianCommand(config.outside_drink_holder_pose[:3], config.outside_drink_holder_pose[3:]))
    # get_state()
    # arm_client_interface.execute_command(CartesianCommand(config.below_drink_holder_pose[:3], config.below_drink_holder_pose[3:]))
    # arm_client_interface.execute_command(CartesianCommand(config.slightly_above_drink_holder_pose[:3], config.slightly_above_drink_holder_pose[3:]))
    # arm_client_interface.execute_command(OpenGripperCommand())
    # arm_client_interface.execute_command(CartesianCommand(config.above_drink_holder_pose[:3], config.above_drink_holder_pose[3:]))

    # arm_client_interface.execute_command(JointCommand(config.left_back_retract_pos))
    # arm_client_interface.execute_command(JointCommand(config.back_retract_pos))
    # arm_client_interface.execute_command(JointCommand(config.before_transfer_pos))
    # arm_client_interface.execute_command(JointCommand(config.above_plate_pos))
    # arm_client_interface.execute_command(JointCommand(config.retract_pos))
    # arm_client_interface.execute_command(JointCommand(config.above_plate_pos))
    # arm_client_interface.execute_command(JointCommand(config.before_transfer_pos))
    # arm_client_interface.execute_command(CartesianCommand(config.above_plate_pose[:3], config.above_plate_pose[3:]))

    # before_transfer_pos = [-3.0854968936528, -1.483771146589941, -2.4555141535696507, -1.3825566440970434, 1.3751202993852156, -0.8903261777143525, 1.7401439441938908]
    # acq_pos = [-3.0854968936528, -1.483771146589941, -2.4555141535696507, -1.3825566440970434, 1.3751202993852156, -0.8903261777143525, -3.008976818683771]
    # arm_client_interface.execute_command(JointCommand(config.retract_pos))
    # arm_client_interface.execute_command(JointCommand(before_transfer_pos))

    # print("Transfer pos:")
    # ee_pose, joint_positions = get_state()
    # arm_client_interface.execute_command(JointCommand(acq_pos))
    # arm_client_interface.execute_command(JointCommand(before_transfer_pos))
    # arm_client_interface.execute_command(JointCommand(config.retract_pos))