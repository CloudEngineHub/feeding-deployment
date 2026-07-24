from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


# Gripper openness expressed as "open amount": 1.0 = fully open, 0.0 = fully closed.
# The Kinova driver uses the opposite raw convention (0.0 = open, 1.0 = closed),
# so raw gripper position = 1.0 - open_amount (see open_amount_to_gripper_pos).
GRIPPER_OPEN_AMOUNT = 1.0        # generic open / release into free space
GRIPPER_GRASP_OPEN_AMOUNT = 0.3  # engage the utensil-mount slot to hold the utensil
GRIPPER_CLOSE_OPEN_AMOUNT = 0.04  # generic close — compact, not driven to the hard stop

# Per-tool grasp openness. grasp_tool holds a tool by opening INTO its mount slot;
# tools not listed here keep the full-open grasp (current behavior).
GRASP_OPEN_AMOUNTS = {
    "utensil": GRIPPER_GRASP_OPEN_AMOUNT,
}


def open_amount_to_gripper_pos(open_amount):
    """Convert an 'open amount' (1.0 = open, 0.0 = closed) to the Kinova raw
    gripper position (0.0 = open, 1.0 = closed)."""
    return 1.0 - open_amount


class KinovaCommand:
    """Establish an interface for commands that can be sent to the robot."""


@dataclass(frozen=True)
class JointTrajectoryCommand(KinovaCommand):
    """Command to follow an joint trajectory."""

    traj: list[NDArray]

    # Rajat ToDo: Ask Tom if this is bad practice
    def __init__(self, traj):
        object.__setattr__(self, "traj", [np.array(x) for x in traj])
        num_dof = 7
        assert all(x.shape == (num_dof,) for x in self.traj)

@dataclass(frozen=True)
class CartesianTrajectoryCommand(KinovaCommand):
    """Command to follow a cartesian trajectory."""

    traj: list[tuple[NDArray, NDArray]]

    def __init__(self, traj):
        object.__setattr__(self, "traj", [(np.array(pos), np.array(quat)) for pos, quat in traj])
        assert all(pos.shape == (3,) and quat.shape == (4,) for pos, quat in self.traj)


@dataclass(frozen=True)
class JointCommand(KinovaCommand):
    """Command to set the joint positions."""

    pos: NDArray

    def __init__(self, pos):
        object.__setattr__(self, "pos", np.array(pos))  # convert list to numpy array
        num_dof = 7
        assert self.pos.shape == (num_dof,)


@dataclass(frozen=True)
class CartesianCommand(KinovaCommand):
    """Command to set the cartesian pose."""

    pos: NDArray
    quat: NDArray
    soft_stop: bool = False

    def __init__(self, pos, quat, soft_stop=False):
        object.__setattr__(self, "pos", np.array(pos))
        object.__setattr__(self, "quat", np.array(quat))
        object.__setattr__(self, "soft_stop", bool(soft_stop))
        assert self.pos.shape == (3,)
        assert self.quat.shape == (4,)


class OpenGripperCommand(KinovaCommand):
    """Command to open the gripper."""


class CloseGripperCommand(KinovaCommand):
    """Command to close the gripper."""


@dataclass(frozen=True)
class SetGripperCommand(KinovaCommand):
    """Command to set the gripper to a specific raw position (0.0=open, 1.0=closed)."""

    pos: float

    def __init__(self, pos):
        object.__setattr__(self, "pos", float(pos))
        assert 0.0 <= self.pos <= 1.0, f"gripper pos out of range [0,1]: {self.pos}"
