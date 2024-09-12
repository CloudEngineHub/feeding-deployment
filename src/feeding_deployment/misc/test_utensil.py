"""The main entry point for running the integrated system."""

from pathlib import Path
from typing import Any
import pybullet as p
import time
import numpy as np
import pandas as pd

from relational_structs import Object
from pybullet_helpers.geometry import Pose, multiply_poses
from pybullet_helpers.link import get_link_pose, get_relative_link_pose
from pybullet_helpers.inverse_kinematics import inverse_kinematics, InverseKinematicsError, set_robot_joints_with_held_object
from pybullet_helpers.joint import JointPositions
from pybullet_helpers.gui import visualize_pose

from feeding_deployment.simulation.scene_description import (
    SceneDescription,
)
from feeding_deployment.actions.high_level_actions import PickToolHLA, tool_type, GroundHighLevelAction
from feeding_deployment.simulation.simulator import (
    FeedingDeploymentPyBulletSimulator,
    FeedingDeploymentSimulatorState,
)


def _sample_food_pose(sim: FeedingDeploymentPyBulletSimulator, rng: np.random.Generator) -> Pose:
    # Define workspace bounds.
    min_x, max_x = 0.35, 0.65
    min_y, max_y = 0.4, 0.75
    min_z, max_z = 0.2, 0.25
    
    # Show corners.
    # visualize_pose(Pose((min_x, min_y, min_z)), sim.physics_client_id)
    # visualize_pose(Pose((max_x, min_y, min_z)), sim.physics_client_id)
    # visualize_pose(Pose((min_x, max_y, min_z)), sim.physics_client_id)
    # visualize_pose(Pose((max_x, max_y, min_z)), sim.physics_client_id)
    # visualize_pose(Pose((min_x, min_y, max_z)), sim.physics_client_id)
    # visualize_pose(Pose((max_x, min_y, max_z)), sim.physics_client_id)
    # visualize_pose(Pose((min_x, max_y, max_z)), sim.physics_client_id)
    # visualize_pose(Pose((max_x, max_y, max_z)), sim.physics_client_id)

    # Sample.
    x = rng.uniform(min_x, max_x)
    y = rng.uniform(min_y, max_y)
    z = rng.uniform(min_z, max_z)

    return Pose((x, y, z), p.getQuaternionFromEuler((np.pi, 0, 0)))


def _check_pose_reachability(pose: Pose, utensil_from_end_effector: Pose,
                             acquisition_tf: Pose, sim: FeedingDeploymentPyBulletSimulator) -> bool:
    target_end_effector_pose = multiply_poses(pose, acquisition_tf, utensil_from_end_effector)

    try:
        joints = inverse_kinematics(sim.robot, target_end_effector_pose)
        set_robot_joints_with_held_object(sim.robot, sim.physics_client_id,
                                          sim.held_object_id,
                                          sim.held_object_tf,
                                          joints)

        reachable = True
    except InverseKinematicsError:
        reachable = False
    
    # Show target end effector pose.
    visualize_pose(multiply_poses(pose, acquisition_tf), sim.physics_client_id)
    while True:
        p.stepSimulation()

    return reachable


def _get_acquisition_to_transfer_plan(food_pose: Pose, target_pose: Pose, utensil_from_end_effector: Pose, sim: FeedingDeploymentPyBulletSimulator, rng: np.random.Generator) -> list[JointPositions]:
    # TODO
    return []


def _measure_efficiency(plan: list[JointPositions], sim: FeedingDeploymentPyBulletSimulator) -> float:
    # TODO
    return 0.0


def _measure_comfort(plan: list[JointPositions], sim: FeedingDeploymentPyBulletSimulator) -> float:
    # TODO
    return 0.0


def _main(use_flair_utensil: bool, max_motion_planning_time: float = 10,
          seed: int = 0, num_samples: int = 100,
) -> None:
    """Testing components of the system."""

    utensil = "flair" if use_flair_utensil else "new"
    rng = np.random.default_rng(seed)

    # Append to existing results or start new results file if none exists.
    outfile = Path(__file__).parent / "results.csv"
    if outfile.exists():
        df = pd.read_csv(outfile, index_col=False)
    else:
        headers = ["Head Reachable", "Food Reachable", "Efficiency", "Comfort", "Utensil", "Env", "Food x", "Food y", "Food z", "Food qx", "Food qy", "Food qz", "Food qw", "Head x", "Head y", "Head z", "Head qx", "Head qy", "Head qz", "Head qw"]
        df = pd.DataFrame(columns=headers)

    kwargs: dict[str, Any] = {
        "drink_pose": Pose((-100, -100, -100)),  # remove from scene
        "wipe_pose": Pose((-100, -100, -100)),  # remove from scene
    }
    
    if use_flair_utensil:
        kwargs["utensil_urdf_path"] = Path(__file__).parent.parent / "assets" / "urdf" / "flair_feeding_tool" / "feeding_tool.urdf"

    # Initialize the simulator.
    scene_description = SceneDescription(**kwargs)
    sim = FeedingDeploymentPyBulletSimulator(scene_description, use_gui=True)

    # Start by picking up the utensil.
    pick_tool_hla = PickToolHLA(sim, robot_interface=None, perception_interface=None,
                                rviz_interface=None, web_interface=None,
                                hla_hyperparams={"max_motion_planning_time": max_motion_planning_time},
                                run_on_robot=False, wrist_controller=None, flair=None)
    pick_utensil = GroundHighLevelAction(pick_tool_hla, (Object("utensil", tool_type),))
    pick_utensil.execute_action()

    if use_flair_utensil:
        sim.robot.set_finger_state(0.69)
        # utensil_from_end_effector = Pose.from_rpy(translation=(0.0, 0.02, 0.04), rpy=(0.0, -1.570796, -1.570796))
        utensil_from_end_effector = Pose.from_rpy(translation=(0.0, -0.02, -0.18), rpy=(0.0, -1.570796, -1.570796))
        acquisition_tf = Pose.from_rpy(translation=(0.0, 0.0, 0.0), rpy=(0.0, 0.0, np.pi))
    else:
        sim.robot.set_finger_state(sim.scene_description.tool_grasp_fingers_value)
        utensil_from_end_effector = Pose.from_rpy(translation=(0.0, -0.035, -0.22), rpy=(0, -1.570796, 1.570796))
        acquisition_tf = Pose.identity()

    # visualize_pose(multiply_poses(sim.robot.get_end_effector_pose(), utensil_from_end_effector.invert()), sim.physics_client_id)
    # while True:
    #     p.stepSimulation()

    # # set to acquisition pose
    # if use_flair_utensil:
    #     acquisition_robot_joint_state = JointPositions([0.005280353523030187, 6.157869196821472, 3.1416656242036614, 4.861354493854926, 1.7963305196145385e-05, 4.4379560890064225, 1.576155405856463]) \
    #         + sim.robot.get_joint_positions()[len(scene_description.above_plate_pos):]
    #     transfer_robot_joint_state = JointPositions([0.3333664491938215, 1.4858324332736625, 3.1415, 1.5856359930210362, 0.8180422599332581, 1.5794872866962613, 4.604932028296647]) \
    #         + sim.robot.get_joint_positions()[len(scene_description.before_transfer_pos):]
    # else:
    #     acquisition_robot_joint_state = scene_description.above_plate_pos + sim.robot.get_joint_positions()[len(scene_description.above_plate_pos):]
    #     transfer_robot_joint_state = scene_description.before_transfer_pos + sim.robot.get_joint_positions()[len(scene_description.before_transfer_pos):]

    # acquisition_state = FeedingDeploymentSimulatorState(
    #     robot_joints=acquisition_robot_joint_state,
    #     utensil_joints=sim.utensil.get_joint_positions(),
    #     drink_pose=scene_description.drink_pose,
    #     wipe_pose=scene_description.wipe_pose,
    #     utensil_pose=scene_description.utensil_pose,
    #     held_object="utensil",
    #     held_object_tf=utensil_from_end_effector,
    # )
    # sim.sync(acquisition_state)


    # transfer_state = FeedingDeploymentSimulatorState(
    #     robot_joints=transfer_robot_joint_state,
    #     utensil_joints=sim.utensil.get_joint_positions(),
    #     drink_pose=scene_description.drink_pose,
    #     wipe_pose=scene_description.wipe_pose,
    #     utensil_pose=scene_description.utensil_pose,
    #     held_object="utensil",
    #     held_object_tf=utensil_from_end_effector,
    # )
    # sim.sync(transfer_state)

    # read npy file
    tool_tip_target_transforms = np.load("benjamin_target_transforms.npy")

    for i in range(num_samples):
        # Sample food pose.
        food_pose = _sample_food_pose(sim, rng)

        # Check reachability of food pose.
        food_pose_reachable = _check_pose_reachability(food_pose, utensil_from_end_effector, acquisition_tf, sim)

        # Sample head pose.
        idx = rng.integers(0, len(tool_tip_target_transforms))
        target_pose = Pose.from_matrix(tool_tip_target_transforms[idx])

        # Check reachability of target pose.
        target_pose_reachable = True  # TODO!

        # Generate a plan for acquisition -> transfer.
        if food_pose_reachable and target_pose_reachable:
            plan = _get_acquisition_to_transfer_plan(food_pose, target_pose, utensil_from_end_effector, sim, rng)
            # Measure efficiency and comfort of plan.
            efficiency = _measure_efficiency(plan, sim)
            comfort = _measure_comfort(plan, sim)
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
    parser.add_argument("--max_motion_planning_time", type=float, default=10.0)
    args = parser.parse_args()

    _main(args.use_flair_utensil, args.max_motion_planning_time)