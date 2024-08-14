"""Utilities for cup manipulation."""

from pybullet_helpers.inverse_kinematics import (
    set_robot_joints_with_held_object,
)
from pybullet_helpers.geometry import Pose
from pybullet_helpers.joint import (
    JointPositions,
)
from dataclasses import dataclass
from pybullet_helpers.camera import capture_superimposed_image
import imageio.v2 as iio
from pathlib import Path

from scene import (
    CupManipulationSceneDescription,
    CupManipulationSceneIDs,
)


@dataclass
class CupManipulationTrajectory:
    """A trajectory for cup manipulation."""

    joint_states: list[JointPositions]
    held_cup_transforms: list[Pose | None]

    def __post_init__(self) -> None:
        assert len(self.joint_states) == len(self.held_cup_transforms)


def make_cup_manipulation_video(
    scene: CupManipulationSceneIDs,
    scene_description: CupManipulationSceneDescription,
    traj: CupManipulationTrajectory,
    outfile: Path,
    fps: int = 20,
) -> None:
    """Make a video for a simulated cup manipulation plan."""
    scene.reset(scene_description)
    imgs = []
    for joint_state, base_link_to_held_obj in zip(
        traj.joint_states, traj.held_cup_transforms, strict=True
    ):
        if base_link_to_held_obj is None:
            scene.robot.set_joints(joint_state)
        else:
            set_robot_joints_with_held_object(
                scene.robot,
                scene.physics_client_id,
                scene.cup_id,
                base_link_to_held_obj,
                joint_state,
            )
        img = capture_superimposed_image(
            scene.physics_client_id, **scene_description.camera_kwargs
        )
        imgs.append(img)
    iio.mimsave(outfile, imgs, fps=fps)
    print(f"Wrote out to {outfile}")
