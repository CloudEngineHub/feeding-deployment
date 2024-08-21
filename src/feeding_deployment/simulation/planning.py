"""Functions for planning using the feeding deployment simulator."""

from __future__ import annotations

from typing import Callable
from feeding_deployment.simulation.simulator import FeedingDeploymentPyBulletSimulator
from feeding_deployment.simulation.state import FeedingDeploymentSimulatorState
from pybullet_helpers.joint import JointPositions, get_joint_infos
from pybullet_helpers.geometry import Pose, multiply_poses, interpolate_poses, get_pose
from dataclasses import dataclass, field
from pybullet_helpers.link import get_link_pose, get_relative_link_pose
from pybullet_helpers.motion_planning import (
    get_joint_positions_distance,
    run_smooth_motion_planning_to_pose,
    smoothly_follow_end_effector_path,
)
from pybullet_helpers.inverse_kinematics import (
    end_effector_transform_to_joints,
    set_robot_joints_with_held_object,
)
from pybullet_helpers.math_utils import geometric_sequence


def _get_plan_to_pregrasp_cup(
    sim: FeedingDeploymentPyBulletSimulator,
    seed: int,
    max_motion_plan_time: float,
) -> list[FeedingDeploymentSimulatorState]:

    robot = sim.robot
    physics_client_id = sim.physics_client_id
    assert physics_client_id == robot.physics_client_id
    collision_ids = sim.get_collision_ids()

    # Commands will be in end effector space, but grasp planning will be in
    # finger frame space.
    finger_frame_id = robot.link_from_name("finger_tip")
    end_effector_link_id = robot.link_from_name(robot.tool_link_name)
    finger_from_end_effector = get_relative_link_pose(
        robot.robot_id, end_effector_link_id, finger_frame_id, physics_client_id
    )

    # Find target finger frame pose relative to the cup handle.
    cup_handle_pose = get_link_pose(
        sim.cup_id, sim.cup_handle_link_id, physics_client_id
    )
    cup_grasp = multiply_poses(cup_handle_pose, sim.scene_description.cup_grasp_transform)

    plan = run_smooth_motion_planning_to_pose(
        cup_grasp,
        robot,
        collision_ids,
        finger_from_end_effector,
        seed,
        max_time=max_motion_plan_time,
    )
    assert plan is not None
    cup_pose = get_pose(sim.cup_id, physics_client_id)
    return [FeedingDeploymentSimulatorState(joints, cup_pose) for joints in plan]


def _get_plan_to_move_grasp_cup_from_pregrasp_position(
    sim: FeedingDeploymentPyBulletSimulator,
    _joint_distance_fn: Callable[[JointPositions, JointPositions], float],
    num_grasp_waypoints: int,
    max_motion_plan_time: float,
) -> list[FeedingDeploymentSimulatorState]:

    # Assumes that _get_plan_to_pregrasp_cup() was just called.
    robot = sim.robot
    physics_client_id = sim.physics_client_id
    collision_ids = sim.get_collision_ids(include_cup=False)
    cup_pose = get_pose(sim.cup_id, physics_client_id)

    tf = Pose(
        (0.0, 0.0, sim.scene_description.cup_grasp_distance),
        (0.0, 0.0, 0.0, 1.0),
    )
    current_end_effector_pose = robot.get_end_effector_pose()
    new_end_effector_pose = multiply_poses(current_end_effector_pose, tf)
    interpolated_poses = list(
        interpolate_poses(
            current_end_effector_pose,
            new_end_effector_pose,
            num_interp=num_grasp_waypoints,
        )
    )
    plan = smoothly_follow_end_effector_path(
        robot,
        interpolated_poses,
        robot.get_joint_positions(),
        collision_ids,
        _joint_distance_fn,
        max_time=max_motion_plan_time,
    )
    sim_states = [FeedingDeploymentSimulatorState(joints, cup_pose) for joints in plan]

    # Open the fingers.
    robot.set_joints(plan[-1])
    robot.open_fingers()
    sim_states.append(FeedingDeploymentSimulatorState(robot.get_joint_positions(), cup_pose))

    # Simulate grasping by faking a constraint with the held object.
    robot.set_joints(plan[-1])
    world_from_end_effector = get_link_pose(
        robot.robot_id, robot.end_effector_id, physics_client_id
    )
    world_from_held_object = get_pose(sim.cup_id, physics_client_id)
    base_link_to_held_obj = multiply_poses(
        world_from_end_effector.invert(), world_from_held_object
    )

    # Move off the table so that the cup is no longer in collision with the table.
    tf = Pose((0.0, -0.1, 0.0), (0.0, 0.0, 0.0, 1.0))
    joints = end_effector_transform_to_joints(robot, tf)
    set_robot_joints_with_held_object(
        robot, physics_client_id, sim.cup_id, base_link_to_held_obj, joints
    )
    sim_state = FeedingDeploymentSimulatorState(joints, cup_pose=None, held_object="cup", held_object_tf=base_link_to_held_obj)
    sim_states.append(sim_state)

    return sim_states


def get_plan_to_grasp_cup(
    sim: FeedingDeploymentPyBulletSimulator,
    seed: int = 0,
    max_motion_plan_time: float = 10.,
    num_grasp_waypoints: int = 5,
) -> list[FeedingDeploymentSimulatorState]:
    """Make a plan to grasp the cup from the current simulator state."""

    # Create joint distance function.
    weights = geometric_sequence(0.9, len(sim.robot.arm_joint_names))
    joint_infos = get_joint_infos(
        sim.robot.robot_id, sim.robot.arm_joints, sim.physics_client_id
    )

    def _joint_distance_fn(pt1: JointPositions, pt2: JointPositions) -> float:
        return get_joint_positions_distance(
            sim.robot,
            joint_infos,
            pt1,
            pt2,
            metric="weighted_joints",
            weights=weights,
        )

    plan = _get_plan_to_pregrasp_cup(sim, seed, max_motion_plan_time)
    sim.robot.set_joints(plan[-1].robot_joints)

    grasp_cup_plan = _get_plan_to_move_grasp_cup_from_pregrasp_position(
        sim,
        _joint_distance_fn,
        num_grasp_waypoints,
        max_motion_plan_time,
    )
    plan.extend(grasp_cup_plan)
    sim.robot.set_joints(plan[-1].robot_joints)

    return plan
