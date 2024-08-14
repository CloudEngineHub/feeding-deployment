"""Utilities for cup manipulation."""

from pybullet_helpers.inverse_kinematics import (
    set_robot_joints_with_held_object,
)
from pybullet_helpers.geometry import Pose
from pybullet_helpers.joint import (
    JointPositions,
)
from pybullet_helpers.camera import capture_superimposed_image
import imageio.v2 as iio
from pathlib import Path

from scene import (
    CupManipulationSceneDescription,
    CupManipulationSceneIDs,
)

import pybullet as p


def make_cup_manipulation_video(
    scene: CupManipulationSceneIDs,
    scene_description: CupManipulationSceneDescription,
    joint_states: list[JointPositions],
    held_object_info: list[tuple[int, Pose] | None],
    outfile: Path,
    fps: int = 20,
) -> None:
    """Make a video for a simulated cup manipulation plan."""
    scene.reset(scene_description)
    imgs = []
    for joint_state, info in zip(joint_states, held_object_info, strict=True):
        if info is None:
            scene.robot.set_joints(joint_state)
        else:
            held_obj_id, base_link_to_held_obj = info
            set_robot_joints_with_held_object(
                scene.robot,
                scene.physics_client_id,
                held_obj_id,
                base_link_to_held_obj,
                joint_state,
            )
        img = capture_superimposed_image(
            scene.physics_client_id, **scene_description.camera_kwargs
        )
        imgs.append(img)
    iio.mimsave(outfile, imgs, fps=fps)
    print(f"Wrote out to {outfile}")
