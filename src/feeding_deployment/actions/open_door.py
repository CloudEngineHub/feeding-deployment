from typing import Any

import time

from relational_structs import (
    GroundAtom,
    GroundOperator,
    LiftedAtom,
    LiftedOperator,
    Object,
    Predicate,
    Type,
    Variable,
)
from feeding_deployment.actions.base import (
    HighLevelAction,
    appliance_type,
    GripperFree,
    InFrontOf,
    DoorOpen,
    DoorClosed,
)

class OpenDoorHLA(HighLevelAction):
    """Open a door (fridge or microwave)."""

    def get_name(self) -> str:
        return "OpenDoor"

    def get_operator(self) -> LiftedOperator:
        obj = Variable("?obj", appliance_type)
        return LiftedOperator(
            self.get_name(),
            parameters=[obj],
            preconditions={
                LiftedAtom(GripperFree, []),
                LiftedAtom(InFrontOf, [obj]),
                LiftedAtom(DoorClosed, [obj]),
            },
            add_effects={LiftedAtom(DoorOpen, [obj])},
            delete_effects={LiftedAtom(DoorClosed, [obj])},
        )
    
    def get_behavior_tree_filename(
        self,
        objects: tuple[Object, ...],
        params: dict[str, Any],
    ) -> str:
        del params  # not used right now
        assert len(objects) == 1
        appliance_type = objects[0]
        assert self.sim.scene_description.scene_label == "vention"
        assert appliance_type.name in ["fridge", "microwave"]
        return f"open_{appliance_type.name}.yaml"

    def open_fridge(self, speed: str) -> None:

        # set speed of the robot to highest
        self.robot_interface.set_speed("high")
        assert self.sim.held_object_name is None
        print("Opening fridge door ...")
        self.move_to_joint_positions(self.sim.scene_description.retract_pos)
        self.move_to_joint_positions(self.sim.scene_description.fridge_door_gaze_pos)

        handle_opening_poses = self.perception_interface.perceive_handle_opening_poses()

        # visualize on rviz
        poses = []
        poses.append(handle_opening_poses["pre_grasp_pose"])
        poses.append(handle_opening_poses["grasp_pose"])
        poses.extend(handle_opening_poses["opening_waypoints"])
        print(f"Visualizing {len(poses)} handle opening poses in RViz ...")
        self.rviz_interface.visualize_poses(poses, frame_id="base_link", ns="handle_opening_poses")

        print(f"Perceived handle opening poses: {handle_opening_poses}")

        self.move_to_joint_positions(self.sim.scene_description.home_pos)

        # print current end-effector pose for debugging
        current_state = self.robot_interface.get_state()
        ee_pose = current_state["ee_pos"]
        print(f"Current end-effector pose: {ee_pose}")
        print(f"Moving to pre-grasp pose: {handle_opening_poses['pre_grasp_pose']}")

        self.move_to_ee_pose(handle_opening_poses["pre_grasp_pose"])
        self.open_gripper()
        self.move_to_ee_pose(handle_opening_poses["grasp_pose"])
        self.close_gripper()
        self.move_to_ee_pose(handle_opening_poses["post_grasp_pose"])
        
    def open_microwave(self, speed: str) -> None:
        assert self.sim.held_object_name is None
        print("Opening microwave door ...")