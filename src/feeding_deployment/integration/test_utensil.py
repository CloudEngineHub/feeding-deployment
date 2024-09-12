"""The main entry point for running the integrated system."""

from pathlib import Path
from typing import Any

import json

from relational_structs import (
    GroundAtom,
    LiftedAtom,
    Object,
    PDDLDomain,
    PDDLProblem,
    Predicate,
)
from relational_structs.utils import parse_pddl_plan
from tomsutils.pddl_planning import run_pyperplan_planning
from pybullet_helpers.geometry import Pose
from pybullet_helpers.link import get_link_pose, get_relative_link_pose
from pybullet_helpers.joint import JointPositions

from feeding_deployment.actions.high_level_actions import (
    TransferToolHLA,
    LookAtPlateHLA,
    AcquireBiteHLA,
    tool_type,
)
from feeding_deployment.interfaces.perception_interface import PerceptionInterface
from feeding_deployment.interfaces.web_interface import WebInterface
from feeding_deployment.interfaces.rviz_interface import RVizInterface
from feeding_deployment.robot_controller.arm_client import ArmInterfaceClient
from feeding_deployment.simulation.scene_description import (
    SceneDescription,
    create_scene_description_from_config,
)
from feeding_deployment.simulation.simulator import (
    FeedingDeploymentPyBulletSimulator,
    FeedingDeploymentSimulatorState,
)
from feeding_deployment.simulation.video import make_simulation_video

import pybullet as p
import time

def _main(use_flair_utensil: bool, make_videos: bool, max_motion_planning_time: float = 10
) -> None:
    """Testing components of the system."""

    print("use_flair_utensil:", use_flair_utensil)
    kwargs: dict[str, Any] = {}
    
    if use_flair_utensil:
        kwargs["utensil_urdf_path"] = Path(__file__).parent.parent / "assets" / "urdf" / "flair_feeding_utensil" / "feeding_utensil.urdf"
        utensil_from_end_effector = Pose.from_rpy(translation=(0.0, 0.02, 0.04), rpy=(0.0, -1.570796, -1.570796))

    # Initialize the simulator
    scene_description = SceneDescription(**kwargs)
    sim = FeedingDeploymentPyBulletSimulator(scene_description, use_gui=True)

    # Create skills for high-level planning.
    hla_hyperparams = {"max_motion_planning_time": max_motion_planning_time}

    if use_flair_utensil:
        sim.robot.set_finger_state(0.69)
        utensil_from_end_effector = Pose.from_rpy(translation=(0.0, 0.02, 0.04), rpy=(0.0, -1.570796, -1.570796))
    else:
        sim.robot.set_finger_state(sim.scene_description.tool_grasp_fingers_value)
        finger_frame_id = sim.robot.link_from_name("finger_tip")
        end_effector_link_id = sim.robot.link_from_name(sim.robot.tool_link_name)
        utensil_from_end_effector = get_relative_link_pose(
            sim.robot.robot_id, finger_frame_id, end_effector_link_id, sim.physics_client_id
        )
    
    # set to acquisition pose
    if use_flair_utensil:
        acquisition_robot_joint_state = JointPositions([0.005280353523030187, 6.157869196821472, 3.1416656242036614, 4.861354493854926, 1.7963305196145385e-05, 4.4379560890064225, 1.576155405856463]) \
            + sim.robot.get_joint_positions()[len(scene_description.above_plate_pos):]
        transfer_robot_joint_state = JointPositions([0.3333664491938215, 1.4858324332736625, 3.1415, 1.5856359930210362, 0.8180422599332581, 1.5794872866962613, 4.604932028296647]) \
            + sim.robot.get_joint_positions()[len(scene_description.before_transfer_pos):]
    else:
        acquisition_robot_joint_state = scene_description.above_plate_pos + sim.robot.get_joint_positions()[len(scene_description.above_plate_pos):]
        transfer_robot_joint_state = scene_description.before_transfer_pos + sim.robot.get_joint_positions()[len(scene_description.before_transfer_pos):]

    input("Press enter to set to acquisition (above plate) state")
    acquisition_state = FeedingDeploymentSimulatorState(
        robot_joints=acquisition_robot_joint_state,
        utensil_joints=sim.utensil.get_joint_positions(),
        drink_pose=scene_description.drink_pose,
        wipe_pose=scene_description.wipe_pose,
        utensil_pose=scene_description.utensil_pose,
        held_object="utensil",
        held_object_tf=utensil_from_end_effector,
    )
    sim.sync(acquisition_state)


    input("Press enter to set to transfer (staging) state")
    transfer_state = FeedingDeploymentSimulatorState(
        robot_joints=transfer_robot_joint_state,
        utensil_joints=sim.utensil.get_joint_positions(),
        drink_pose=scene_description.drink_pose,
        wipe_pose=scene_description.wipe_pose,
        utensil_pose=scene_description.utensil_pose,
        held_object="utensil",
        held_object_tf=utensil_from_end_effector,
    )
    sim.sync(transfer_state)

    input("Press enter to sample goal fork pose for transfer (within Benjamin's ROM)")
    # sample goal fork trajectories for transfer
    raise NotImplementedError


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--use_flair_utensil", action="store_true")
    parser.add_argument("--make_videos", action="store_true")
    parser.add_argument("--max_motion_planning_time", type=float, default=10.0)
    args = parser.parse_args()

    _main(args.use_flair_utensil, args.make_videos, args.max_motion_planning_time)