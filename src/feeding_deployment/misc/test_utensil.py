"""The main entry point for running the integrated system."""

from pathlib import Path
from typing import Any
import pybullet as p
import time
import numpy as np
import pandas as pd
from functools import partial
import tqdm

from pybullet_helpers.geometry import Pose, multiply_poses
from pybullet_helpers.link import get_link_pose, get_relative_link_pose, get_link_state
from pybullet_helpers.inverse_kinematics import inverse_kinematics, InverseKinematicsError, set_robot_joints_with_held_object, sample_collision_free_inverse_kinematics
from pybullet_helpers.joint import JointPositions, get_joint_infos, get_joints
from pybullet_helpers.gui import visualize_pose
from pybullet_helpers.utils import create_pybullet_cylinder
from pybullet_helpers.motion_planning import (
    get_joint_positions_distance,
    run_smooth_motion_planning_to_pose,
    get_motion_plan_distance,
)

from feeding_deployment.simulation.scene_description import (
    SceneDescription,
)
from feeding_deployment.actions.high_level_actions import (
    Object,
    GroundHighLevelAction,
    TransferToolHLA,
    LookAtPlateHLA,
    tool_type,
)
from feeding_deployment.simulation.simulator import (
    FeedingDeploymentPyBulletSimulator,
    FeedingDeploymentSimulatorState,
)
from feeding_deployment.simulation.planning import _plan_to_sim_state_trajectory, remap_trajectory_to_constant_distance


def _sample_food_pose(sim: FeedingDeploymentPyBulletSimulator, rng: np.random.Generator) -> Pose:
    # Define workspace bounds.
    min_x, max_x = 0.35, 0.65
    min_y, max_y = 0.4, 0.6
    min_z, max_z = 0.2, 0.21
    min_pitch, max_pitch = np.pi - np.pi / 6, np.pi + np.pi / 6
    
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
    pitch = rng.uniform(min_pitch, max_pitch)

    return Pose((x, y, z), p.getQuaternionFromEuler((0, pitch, 0)))


def _run_fork_tip_ik(pose: Pose, utensil_tip_from_end_effector: Pose,
                             sim: FeedingDeploymentPyBulletSimulator,
                             debug: bool = False,
                             check_collisions: bool = True) -> JointPositions | None:
    target_end_effector_pose = multiply_poses(pose, utensil_tip_from_end_effector.invert())

    joints = None
    try:
        if check_collisions:
            gen = sample_collision_free_inverse_kinematics(sim.robot, target_end_effector_pose,
                                                    sim.get_collision_ids(),
                                                    held_object=sim.held_object_id,
                                                    base_link_to_held_obj=sim.held_object_tf)
            joints = next(gen)
        else:
            joints = inverse_kinematics(sim.robot, target_end_effector_pose)
        set_robot_joints_with_held_object(sim.robot, sim.physics_client_id,
                                          sim.held_object_id,
                                          sim.held_object_tf,
                                          joints)

    except (InverseKinematicsError, StopIteration):
        pass

    # Show pose.
    if debug:
        ids = visualize_pose(target_end_effector_pose, sim.physics_client_id)
        time.sleep(0.1)
        print("Reachable?", joints is not None)
        for i in ids:
            p.removeUserDebugItem(i, sim.physics_client_id)

    return joints


def _get_acquisition_to_transfer_plan(init_joints: JointPositions, target_pose: Pose, utensil_tip_from_end_effector: Pose,
                                      sim: FeedingDeploymentPyBulletSimulator,
                                      max_motion_planning_time: float = 10) -> list[FeedingDeploymentSimulatorState]:
    
    set_robot_joints_with_held_object(sim.robot, sim.physics_client_id,
                                          sim.held_object_id,
                                          sim.held_object_tf,
                                          init_joints)
    
    look_at_plate_hla = LookAtPlateHLA(sim, None, None, None, None, {"max_motion_planning_time": max_motion_planning_time}, False, None, None)
    transfer_bite_hla = TransferToolHLA(sim, None, None, None, None, {"max_motion_planning_time": max_motion_planning_time}, False, None, None)

    utensil = Object("utensil", tool_type)
    look_at_plate = GroundHighLevelAction(look_at_plate_hla, (utensil, ))
    transfer_bite = GroundHighLevelAction(transfer_bite_hla, (utensil, ), params={"target_pose": target_pose,
                                                                                  "utensil_tip_from_end_effector": utensil_tip_from_end_effector})

    sim_states = []
    sim_states.extend(look_at_plate.execute_action())
    sim_states.extend(transfer_bite.execute_action())
    


    # set_robot_joints_with_held_object(sim.robot, sim.physics_client_id,
    #                                       sim.held_object_id,
    #                                       sim.held_object_tf,
    #                                       init_joints)
    
    # collision_ids = set()
    # seed = 0
    # end_effector_frame_to_plan_frame = utensil_tip_from_end_effector.invert()
    # plan = run_smooth_motion_planning_to_pose(target_pose, sim.robot, collision_ids, end_effector_frame_to_plan_frame,
    #                                    seed, sim.held_object_id,
    #                                    sim.held_object_tf,
    #                                    max_time=max_motion_planning_time)
    # assert plan is not None

    # for state in plan:
    #     set_robot_joints_with_held_object(sim.robot, sim.physics_client_id,
    #                                       sim.held_object_id,
    #                                       sim.held_object_tf,
    #                                       state)
    #     time.sleep(0.1)

    return sim_states


def _measure_plan_length(plan: list[JointPositions], sim: FeedingDeploymentPyBulletSimulator) -> float:

    def _score_motion_plan(plan: list[JointPositions]) -> float:
        weights = [1.0] * len(sim.robot.arm_joints)
        joint_infos = get_joint_infos(
            sim.robot.robot_id, sim.robot.arm_joints, sim.robot.physics_client_id
        )
        dist_fn = partial(
            get_joint_positions_distance,
            sim.robot,
            joint_infos,
            metric="weighted_joints",
            weights=weights,
        )
        return get_motion_plan_distance(plan, dist_fn)

    return _score_motion_plan(plan)


def _measure_plan_comfort(plan: list[JointPositions], head_pose: Pose, sim: FeedingDeploymentPyBulletSimulator) -> float:
    comfort_score = 0.0

    for state in plan:
        set_robot_joints_with_held_object(sim.robot, sim.physics_client_id,
                                          sim.held_object_id,
                                          sim.held_object_tf,
                                          state)
        comfort_score += _measure_state_comfort(head_pose, sim)

    return comfort_score


def _measure_state_comfort(feed_pose: Pose, sim: FeedingDeploymentPyBulletSimulator) -> float:
    # Rotate feed pose to get mouth pose.
    mouth_pose = multiply_poses(feed_pose, Pose.from_rpy((0.0, 0.0, 0.0), rpy=(np.pi, 0.0, 0.0)))

    # Snap food to fork. Super duper hacky.
    fork_pose = get_link_pose(sim.utensil_id, _get_fork_tip_link_id(sim), sim.physics_client_id)
    if not hasattr(sim, "_food_id"):
        sim._food_id = create_pybullet_cylinder((0.25, 0.1, 0.25, 1.0), radius=0.01,
                                                length=0.01, physics_client_id=sim.physics_client_id)
    p.resetBasePositionAndOrientation(
        sim._food_id,
        fork_pose.position,
        fork_pose.orientation,
        physicsClientId=sim.physics_client_id,
    )

    # The rays start in an array relative to the head position and point in
    # straight lines for a maximum distance.
    num_rows, num_cols = 5, 5
    max_ray_length = 1.0
    row_delta, col_delta = 0.01, 0.01
    assert (num_rows % 2 == 1) and (num_cols % 2 == 1)  # odd numbers

    ray_from_positions = []
    ray_to_positions = []

    visualize_pose(mouth_pose, sim.physics_client_id)

    for r in range(num_rows):
        row_val = (r - num_rows // 2) * row_delta
        for c in range(num_cols):
            col_val = (c - num_cols // 2) * col_delta
            # Transform to world pose frame.
            ray_from = Pose((row_val, col_val, 0.0))
            ray_to = Pose((row_val, col_val, max_ray_length))
            mouth_ray_from = multiply_poses(mouth_pose, ray_from)
            mouth_ray_to = multiply_poses(mouth_pose, ray_to)
            ray_from_positions.append(mouth_ray_from.position)
            ray_to_positions.append(mouth_ray_to.position)

    ray_outputs = p.rayTestBatch(
        rayFromPositions=ray_from_positions,
        rayToPositions=ray_to_positions,
        physicsClientId=sim.physics_client_id,
    )

    # See equation 11 in paper.
    alpha = 10.0
    sigma = np.eye(2)
    score = 0.0
    for i, output in enumerate(ray_outputs):
        if output[0] != -1:
            ray_from = ray_from_positions[i]
            world_hit_pose = Pose(output[3])
            # Transform the hit position back into the mouth frame.
            hit_pose = multiply_poses(mouth_pose.invert(), world_hit_pose)
            # assert multiply_poses(mouth_pose, hit_pose).allclose(world_hit_pose)
            # See equation 11 in paper.
            vec = np.array(hit_pose.position[:2])
            point_score = 1 - np.exp(-alpha * np.transpose(vec) @ sigma @ vec / (hit_pose.position[2] ** 2))
            score += point_score

            p.addUserDebugLine(ray_from, world_hit_pose.position, (point_score, point_score, 0.0),
                               physicsClientId=sim.physics_client_id)

    if ray_outputs:
        score /= len(ray_outputs)

    time.sleep(0.1)
    p.removeAllUserDebugItems(physicsClientId=sim.physics_client_id)
        
    # Put food out of view.
    p.resetBasePositionAndOrientation(
        sim._food_id,
        (-100, -100, -100),
        fork_pose.orientation,
        physicsClientId=sim.physics_client_id,
    )

    return score


def _get_fork_tip_link_id(sim: FeedingDeploymentPyBulletSimulator) -> int:
    joint_ids = get_joints(sim.utensil_id, sim.physics_client_id)
    joint_infos = get_joint_infos(sim.utensil_id, joint_ids, sim.physics_client_id)
    return [i for i, info in enumerate(joint_infos) if info.linkName == "fork_tip"][0]


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
        headers = ["Head Reachable", "Food Reachable", "Plan Length", "Comfort", "Utensil", "Food x", "Food y", "Food z", "Food qx", "Food qy", "Food qz", "Food qw", "Head x", "Head y", "Head z", "Head qx", "Head qy", "Head qz", "Head qw"]
        df = pd.DataFrame(columns=headers)

    kwargs: dict[str, Any] = {
        "drink_pose": Pose((-100, -100, -100)),  # remove from scene
        "wipe_pose": Pose((-100, -100, -100)),  # remove from scene
        "conservative_bb_pose": Pose((-100, -100, -100)),  # remove from scene
    }
    
    if use_flair_utensil:
        kwargs["utensil_urdf_path"] = Path(__file__).parent.parent / "assets" / "urdf" / "flair_feeding_tool" / "feeding_tool.urdf"

    # Initialize the simulator.
    scene_description = SceneDescription(**kwargs)
    sim = FeedingDeploymentPyBulletSimulator(scene_description, use_gui=True)

    # Set to acquisition pose, first without grasping.
    if use_flair_utensil:
        init_robot_joints = JointPositions([0.005280353523030187, 6.157869196821472, 3.1416656242036614, 4.861354493854926, 1.7963305196145385e-05, 4.4379560890064225, 1.576155405856463]) \
            + sim.robot.get_joint_positions()[len(scene_description.above_plate_pos):]
    else:
        init_robot_joints = scene_description.above_plate_pos + sim.robot.get_joint_positions()[len(scene_description.above_plate_pos):]

    acquisition_state = FeedingDeploymentSimulatorState(
        robot_joints=init_robot_joints,
        drink_pose=scene_description.drink_pose,
        wipe_pose=scene_description.wipe_pose,
        utensil_pose=scene_description.utensil_pose,
    )
    sim.sync(acquisition_state)
    
    # Recover the utensil-end-effector transform.
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

    # Simulate grasping.
    acquisition_state = FeedingDeploymentSimulatorState(
        robot_joints=init_robot_joints,
        drink_pose=scene_description.drink_pose,
        wipe_pose=scene_description.wipe_pose,
        held_object="utensil",
        held_object_tf=utensil_from_end_effector,
    )
    sim.sync(acquisition_state)

    # Get the transform from utensil base to utensil tip.
    fork_tip_idx = _get_fork_tip_link_id(sim)
    fork_tip_from_utensil = get_relative_link_pose(sim.utensil_id, fork_tip_idx, -1, sim.physics_client_id)
    fork_tip_from_end_effector = multiply_poses(utensil_from_end_effector, fork_tip_from_utensil)

    # utensil_pose = get_link_pose(sim.utensil_id, -1, sim.physics_client_id)
    # tip_pose = multiply_poses(utensil_pose, fork_tip_from_utensil)

    # tip_pose = multiply_poses(sim.robot.get_end_effector_pose(), fork_tip_from_end_effector)
    # visualize_pose(tip_pose, sim.physics_client_id)
    # while True:
    #     p.stepSimulation()

    # read npy file
    tool_tip_target_transforms = np.load("benjamin_target_transforms.npy")

    for _ in tqdm.tqdm(range(num_samples)):
        # Sample food pose.
        food_pose = _sample_food_pose(sim, rng)

        # Check reachability of food pose.
        acquisition_joints = _run_fork_tip_ik(food_pose, fork_tip_from_end_effector, sim)
        food_pose_reachable = acquisition_joints is not None

        # Sample head pose.
        idx = rng.integers(0, len(tool_tip_target_transforms))
        target_pose = Pose.from_matrix(tool_tip_target_transforms[idx])

        # Check reachability of target pose.
        transfer_joints = _run_fork_tip_ik(target_pose, fork_tip_from_end_effector, sim, check_collisions=False)
        target_pose_reachable = transfer_joints is not None

        # Generate a plan for acquisition -> transfer.
        if food_pose_reachable and target_pose_reachable:
            sim_states = _get_acquisition_to_transfer_plan(init_robot_joints, target_pose, fork_tip_from_end_effector, sim,
                                                     max_motion_planning_time=max_motion_planning_time)
            plan = [s.robot_joints for s in sim_states]
            # Measure efficiency and comfort of plan.
            plan_length = _measure_plan_length(plan, sim)
            comfort = _measure_plan_comfort(plan, target_pose, sim)
        else:
            plan_length = np.nan
            comfort = np.nan

        # Save result.
        datum = {
            "Head Reachable": target_pose_reachable,
            "Food Reachable": food_pose_reachable,
            "Plan Length": plan_length,
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
        # print(f"Wrote out to {outfile}.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--use_flair_utensil", action="store_true")
    parser.add_argument("--max_motion_planning_time", type=float, default=10.0)
    args = parser.parse_args()

    _main(args.use_flair_utensil, args.max_motion_planning_time)