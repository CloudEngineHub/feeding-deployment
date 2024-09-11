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

    # Initialize the simulator
    scene_description = SceneDescription()
    sim = FeedingDeploymentPyBulletSimulator(scene_description, use_gui=True)

    # Create skills for high-level planning.
    hla_hyperparams = {"max_motion_planning_time": max_motion_planning_time}

    finger_frame_id = sim.robot.link_from_name("finger_tip")
    end_effector_link_id = sim.robot.link_from_name(sim.robot.tool_link_name)
    utensil_from_end_effector = get_relative_link_pose(
        sim.robot.robot_id, finger_frame_id, end_effector_link_id, sim.physics_client_id
    )

    sim.robot.set_finger_state(sim.scene_description.tool_grasp_fingers_value)
    sim.utensil.set_joints(joint_positions=JointPositions([0.5, 0.5]))  

    init_state = FeedingDeploymentSimulatorState(
        robot_joints=sim.robot.get_joint_positions(),
        utensil_joints=sim.utensil.get_joint_positions(),
        drink_pose=scene_description.drink_pose,
        wipe_pose=scene_description.wipe_pose,
        utensil_pose=scene_description.utensil_pose,
        held_object="utensil",
        held_object_tf=utensil_from_end_effector,
    )
    sim.sync(init_state)

    # step simulation until ctrl+c
    try:
        while True:
            # do nothing
            time.sleep(0.1)
            # p.stepSimulation(physicsClientId=sim.physics_client_id)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--use_flair_utensil", action="store_true")
    parser.add_argument("--make_videos", action="store_true")
    parser.add_argument("--max_motion_planning_time", type=float, default=10.0)
    args = parser.parse_args()

    _main(args.use_flair_utensil, args.make_videos, args.max_motion_planning_time)