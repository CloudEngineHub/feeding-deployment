"""The main entry point for running the integrated system."""

from pathlib import Path
from typing import Any
import json
import pybullet as p
import time
import numpy as np
import pandas as pd

from relational_structs.utils import parse_pddl_plan
from tomsutils.pddl_planning import run_pyperplan_planning
from pybullet_helpers.geometry import Pose
from pybullet_helpers.link import get_link_pose, get_relative_link_pose
from pybullet_helpers.joint import JointPositions
from pybullet_helpers.gui import visualize_pose

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


def _sample_food_pose(scene_description: SceneDescription, rng: np.random.Generator) -> Pose:
    # TODO
    return Pose.identity()


def _check_pose_reachability(pose: Pose, sim: FeedingDeploymentPyBulletSimulator) -> bool:
    # TODO
    return True


def _get_acquisition_to_transfer_plan(food_pose: Pose, target_pose: Pose, sim: FeedingDeploymentPyBulletSimulator, rng: np.random.Generator) -> list[JointPositions]:
    # TODO
    return []


def _measure_efficiency(plan: list[JointPositions], sim: FeedingDeploymentPyBulletSimulator) -> float:
    # TODO
    return 0.0


def _measure_comfort(plan: list[JointPositions], sim: FeedingDeploymentPyBulletSimulator, setting: str) -> float:
    # TODO
    return 0.0


def _main(use_flair_utensil: bool, make_videos: bool, max_motion_planning_time: float = 10,
          seed: int = 0, num_samples: int = 100,
) -> None:
    """Testing components of the system."""

    env_settings = ["Social", "TV", "Radio"]

    utensil = "flair" if use_flair_utensil else "new"
    rng = np.random.default_rng(seed)

    # Append to existing results or start new results file if none exists.
    outfile = Path(__file__).parent / "results.csv"
    if outfile.exists():
        df = pd.read_csv(outfile, index_col=False)
    else:
        headers = ["Head Reachable", "Food Reachable", "Efficiency", "Comfort", "Utensil", "Env", "Food x", "Food y", "Food z", "Food qx", "Food qy", "Food qz", "Food qw", "Head x", "Head y", "Head z", "Head qx", "Head qy", "Head qz", "Head qw"]
        df = pd.DataFrame(columns=headers)

    print("use_flair_utensil:", use_flair_utensil)
    kwargs: dict[str, Any] = {}
    
    if use_flair_utensil:
        kwargs["utensil_urdf_path"] = Path(__file__).parent.parent / "assets" / "urdf" / "flair_feeding_tool" / "feeding_tool.urdf"
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

    # read npy file
    tool_tip_target_transforms = np.load("benjamin_target_transforms.npy")

    for i in range(num_samples):
        # Sample food pose.
        food_pose = _sample_food_pose(scene_description, rng)

        # Check reachability of food pose.
        food_pose_reachable = _check_pose_reachability(food_pose, sim)

        # Sample head pose.
        idx = rng.integers(0, len(tool_tip_target_transforms))
        target_pose = Pose.from_matrix(tool_tip_target_transforms[idx])

        # Sample environment setting.
        idx = rng.integers(0, len(env_settings))
        env_setting = env_settings[idx]

        # Check reachability of target pose.
        target_pose_reachable = True  # TODO!

        # Generate a plan for acquisition -> transfer.
        if food_pose_reachable and target_pose_reachable:
            plan = _get_acquisition_to_transfer_plan(food_pose, target_pose, sim, rng)
            # Measure efficiency and comfort of plan.
            efficiency = _measure_efficiency(plan, sim)
            comfort = _measure_comfort(plan, sim, env_setting)
        else:
            efficiency = np.nan
            comfort = np.nan

        # Save result.
        datum = {
            "Head Reachable": target_pose_reachable,
            "Food Reachable": food_pose_reachable,
            "Efficiency": efficiency,
            "Comfort": comfort,
            "Utensil": utensil,
            "Env": env_setting,
            "Food x": food_pose.position[0],
            "Food y": food_pose.position[1],
            "Food z": food_pose.position[2],
            "Food qx": food_pose.orientation[0],
            "Food qy": food_pose.orientation[1],
            "Food qz": food_pose.orientation[2],
            "Food qw": food_pose.orientation[3],
            "Head x": target_pose.position[0],
            "Head y": target_pose.position[1],
            "Head z": target_pose.position[2],
            "Head qx": target_pose.orientation[0],
            "Head qy": target_pose.orientation[1],
            "Head qz": target_pose.orientation[2],
            "Head qw": target_pose.orientation[3],
        }
        df = pd.concat([df, pd.DataFrame([datum])], ignore_index=True)

    df.to_csv(outfile, index=False)
    print(f"Wrote out to {outfile}.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--use_flair_utensil", action="store_true")
    parser.add_argument("--make_videos", action="store_true")
    parser.add_argument("--max_motion_planning_time", type=float, default=10.0)
    args = parser.parse_args()

    _main(args.use_flair_utensil, args.make_videos, args.max_motion_planning_time)