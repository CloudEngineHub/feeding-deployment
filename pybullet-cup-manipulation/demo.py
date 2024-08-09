"""Script to develop cup manipulation skills in simulation."""

from pybullet_helpers.robots import create_pybullet_robot
from pybullet_helpers.robots.single_arm import SingleArmPyBulletRobot
from pybullet_helpers.geometry import Pose, get_pose, multiply_poses, Quaternion
from pybullet_helpers.joint import JointPositions
from pybullet_helpers.ikfast.utils import ikfast_inverse_kinematics
from pybullet_helpers.camera import create_gui_connection
from pybullet_helpers.utils import create_pybullet_cylinder, create_pybullet_block
from pybullet_helpers.motion_planning import run_motion_planning
from pybullet_helpers.gui import visualize_pose
from itertools import islice
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R

import pybullet as p


def _initialize_scene() -> tuple[SingleArmPyBulletRobot, Pose, int, int, set[int]]:
    """Returns robot, cup ID, table ID, other collision IDs."""

    robot_base_pose = Pose((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    robot_home_joints = [
        np.pi / 2,
        -np.pi / 4,
        -np.pi / 2,
        0.,
        np.pi / 2,
        -np.pi / 2,
        np.pi / 2,
        0.,
        0.,
    ]

    robot_holder_rgba = (0.5, 0.5, 0.5, 1.0)
    robot_holder_half_extents = (0.25, 0.25, 0.5)
    robot_holder_position = (0.0, 0.0, -0.5)
    robot_holder_orientation = (0.0, 0.0, 0.0, 1.0)

    collision_region1_orientation = (0.0, 0.0, 0.0, 1.0)
    collision_region1_position = (-0.75, -0.3, 0.0)
    collision_region1_half_extents = (0.5, 0.75, 1.0)
    collision_region1_rgba = (1.0, 0.0, 0.0, 0.25)

    collision_region2_orientation = (0.0, 0.0, 0.0, 1.0)
    collision_region2_position = (-0.075, 0.55, 0.05)
    collision_region2_half_extents = (0.05, 0.05, 0.05)
    collision_region2_rgba = (1.0, 0.0, 0.0, 0.25)

    wheelchair_position = (-0.5, 0.0, -0.25)
    wheelchair_orientation = (0.0, 0.0, 1.0, 0.0)

    table_rgba = (0.5, 0.5, 0.5, 1.0)
    table_half_extents = (0.75, 0.25, 0.5)
    table_position = (-0.5, 0.75, -0.5)
    table_orientation = (0.0, 0.0, 0.0, 1.0)

    cup_rgba = (0.0, 0.0, 1.0, 1.0)
    cup_radius = 0.02
    cup_length = 0.12
    cup_position = (0.0, 0.75, cup_length / 2)
    cup_orientation = (0.0, 0.0, 0.0, 1.0)

    physics_client_id = create_gui_connection(camera_pitch=180)
    p.setGravity(0.0, 0.0, 0.0, physicsClientId=physics_client_id)

    # Create robot.
    robot = create_pybullet_robot("kinova-gen3", physics_client_id,
                                  base_pose=robot_base_pose,
                                  control_mode="reset",
                                  home_joint_positions=robot_home_joints)
    
    # Create a base for visualization purposes.
    robot_holder_id = create_pybullet_block(
        robot_holder_rgba,
        half_extents=robot_holder_half_extents,
        physics_client_id=physics_client_id,
    )
    p.resetBasePositionAndOrientation(
        robot_holder_id,
        robot_holder_position,
        robot_holder_orientation,
        physicsClientId=physics_client_id,
    )

    # Create wheelchair for visualization purposes only.
    wheelchair_urdf_path = (
        Path(__file__).parent.parent
        / "assets"
        / "urdf"
        / "wheelchair"
        / "wheelchair.urdf"
    )
    wheelchair_id = p.loadURDF(
        str(wheelchair_urdf_path), useFixedBase=True, physicsClientId=physics_client_id
    )
    p.resetBasePositionAndOrientation(
        wheelchair_id,
        wheelchair_position,
        wheelchair_orientation,
        physicsClientId=physics_client_id,
    )

    # Create cup.
    cup_id = create_pybullet_cylinder(
        cup_rgba,
        radius=cup_radius,
        length=cup_length,
        physics_client_id=physics_client_id,
    )
    p.resetBasePositionAndOrientation(
        cup_id,
        cup_position,
        cup_orientation,
        physicsClientId=physics_client_id,
    )

    table_id = create_pybullet_block(
        table_rgba,
        half_extents=table_half_extents,
        physics_client_id=physics_client_id,
    )
    p.resetBasePositionAndOrientation(
        table_id,
        table_position,
        table_orientation,
        physicsClientId=physics_client_id,
    )

    # Create collision areas.
    collision_region_ids = set()
    
    collision_region_id1 = create_pybullet_block(
        collision_region1_rgba,
        half_extents=collision_region1_half_extents,
        physics_client_id=physics_client_id,
    )
    p.resetBasePositionAndOrientation(
        collision_region_id1,
        collision_region1_position,
        collision_region1_orientation,
        physicsClientId=physics_client_id,
    )
    collision_region_ids.add(collision_region_id1)

    collision_region_id2 = create_pybullet_block(
        collision_region2_rgba,
        half_extents=collision_region2_half_extents,
        physics_client_id=physics_client_id,
    )
    p.resetBasePositionAndOrientation(
        collision_region_id2,
        collision_region2_position,
        collision_region2_orientation,
        physicsClientId=physics_client_id,
    )
    collision_region_ids.add(collision_region_id2)

    return robot, cup_id, table_id, collision_region_ids


def collision_free_ik(robot: SingleArmPyBulletRobot, end_effector_pose: Pose, collision_ids: set[int]) -> JointPositions | None:
    """Find an IK solution that results in no collisions between the robot
    and the collision_ids."""
    
    # TODO: move this function into pybullet-helpers.
    # TODO move out these hyperparameters
    # TODO handle no solutions possible
    max_time: float = 0.05
    max_attempts: int = 1000000000
    max_candidates: int = 10
    max_distance: float = np.inf
    norm: float = np.inf
    rng = np.random.default_rng(0)

    physics_client_id = robot.physics_client_id

    generator = ikfast_inverse_kinematics(robot, end_effector_pose, max_time, max_distance, max_attempts, norm, rng)
    generator = islice(generator, max_candidates)

    for candidate in generator:
        # TODO refactor to avoid this...
        first_finger_idx, second_finger_idx = sorted(
            [robot.left_finger_joint_idx, robot.right_finger_joint_idx]
        )
        candidate.insert(first_finger_idx, robot.open_fingers)
        candidate.insert(second_finger_idx, robot.open_fingers)

        robot.set_joints(candidate)

        has_collision = False
        p.performCollisionDetection(physicsClientId=physics_client_id)
        for body in collision_ids:
            if p.getContactPoints(
                robot.robot_id, body, physicsClientId=physics_client_id
            ):
                has_collision = True
                break
        if not has_collision:
            return candidate
        
    return None


def _sample_grasp(cup_pose: Pose, rng: np.random.Generator) -> Pose:
    # Pose of grasping the center of the cup.
    default_cup_pregrasp_transform = Pose((0.0, 0.0, 0.0), (0, np.sqrt(2) / 2, np.sqrt(2) / 2, 0))
    grasp = multiply_poses(cup_pose, default_cup_pregrasp_transform)

    # Randomly rotate around the cup.
    angle = rng.uniform(-np.pi / 4, np.pi / 4)
    rotation_quat = p.getQuaternionFromEuler((0.0, angle, 0.0))
    tf = Pose((0.0, 0.0, 0.0), rotation_quat)
    grasp = multiply_poses(grasp, tf)

    # Move the grasp away from the cup in the opposite direction of angle.
    translation_distance = -0.2
    translation_vector = (0.0, 0.0, translation_distance * np.cos(angle))
    tf = Pose(translation_vector, (0.0, 0.0, 0.0, 1.0))
    grasp = multiply_poses(grasp, tf)

    return grasp


def _main():

    robot, cup_id, table_id, other_collision_ids = _initialize_scene()
    collision_ids = {cup_id, table_id} | other_collision_ids
    physics_client_id = robot.physics_client_id
    
    # Use the table pose as a frame of reference.
    table_frame = get_pose(table_id, physics_client_id)
    # Move the frame to the bottom right hand corner of the table so we can see it.
    dims = p.getVisualShapeData(table_id, physicsClientId=physics_client_id)[0][3]
    offset = Pose((dims[0] / 2, -dims[1] / 2, dims[2] / 2))
    table_frame = multiply_poses(offset, table_frame)
    
    visualize_pose(table_frame, physics_client_id)

    # Find target end effector pose relative to the cup.
    cup_pose = get_pose(cup_id, physics_client_id)
    # p.removeBody(cup_id, physicsClientId=physics_client_id)  # TODO

    # Find target joint positions using inverse kinematics.
    max_grasp_candidates = 1000
    rng = np.random.default_rng(0)
    for _ in range(max_grasp_candidates):
        import time; time.sleep(0.1)
        candidate = _sample_grasp(cup_pose, rng)
        target_joint_positions = collision_free_ik(robot, candidate, collision_ids)
        if target_joint_positions is not None:
            robot.set_joints(target_joint_positions)
            print("Succeeded!!")
            break
    else:
        print("Failed :(")

    # TODO: run motion planning.

    while True:
        p.stepSimulation(physicsClientId=physics_client_id)


if __name__ == "__main__":
    _main()
