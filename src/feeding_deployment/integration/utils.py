"""Utilities for integration."""

import numpy as np

from feeding_deployment.drinking.utils import CupManipulationTrajectory
from feeding_deployment.robot_controller.kinova import (
    CloseGripperCommand,
    JointTrajectoryCommand,
    KinovaCommand,
    OpenGripperCommand,
)


def cup_manipulation_trajectory_to_kinova_commands(
    traj: CupManipulationTrajectory,
) -> list[KinovaCommand]:
    """The Kinova controller expects arm joints and gripper values."""
    cmds = []
    last_gripper: float | None = None
    current_trajectory = []
    for joint_state in traj.joint_states:
        assert len(joint_state) == 9  # making assumptions about Kinova
        arm = np.array(joint_state[:7])
        assert np.isclose(joint_state[7], joint_state[8])
        gripper = joint_state[8]
        if last_gripper is None or not np.isclose(gripper, last_gripper):
            if current_trajectory:
                cmds.append(JointTrajectoryCommand(current_trajectory))
                current_trajectory = []
            if last_gripper is None or last_gripper >= 0:
                cmds.append(CloseGripperCommand())
            else:
                cmds.append(OpenGripperCommand())
        current_trajectory.append(arm)
        last_gripper = gripper
    if current_trajectory:
        cmds.append(JointTrajectoryCommand(current_trajectory))
    return cmds
