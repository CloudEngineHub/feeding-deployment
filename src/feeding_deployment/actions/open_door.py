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

        handle_opening_poses = self.perception_interface.perceive_handle_opening_poses("white fridge door")

        # visualize on rviz
        poses = []
        poses.append(handle_opening_poses["pre_grasp_pose"])
        poses.append(handle_opening_poses["grasp_pose"])
        poses.extend(handle_opening_poses["opening_waypoints"])
        poses.append(handle_opening_poses["post_release_pose"])
        poses.append(handle_opening_poses["pre_push_pose"])
        poses.append(handle_opening_poses["push_pose"])
        poses.extend(handle_opening_poses["push_waypoints"])
        print(f"Visualizing {len(poses)} handle opening poses in RViz ...")
        self.rviz_interface.visualize_poses(poses, frame_id="base_link", ns="handle_opening_poses")

        # self.move_to_joint_positions(self.sim.scene_description.home_pos)
        self.move_to_joint_positions(self.sim.scene_description.fridge_door_staging_pos)

        self.move_to_ee_pose(handle_opening_poses["pre_grasp_pose"])
        self.open_gripper()
        self.move_to_ee_pose(handle_opening_poses["grasp_pose"])
        self.close_gripper()
        # self.move_to_ee_pose(handle_opening_poses["post_grasp_pose"])
        self.move_to_ee_pose_trajectory(handle_opening_poses["opening_waypoints"])
        self.open_gripper()
        self.move_to_ee_pose(handle_opening_poses["post_release_pose"])
        
        # self.move_to_joint_positions(self.sim.scene_description.fridge_door_intermediate_restract_pos)

        self.move_to_ee_pose(handle_opening_poses["pre_push_pose"])
        self.move_to_ee_pose(handle_opening_poses["push_pose"])
        self.move_to_ee_pose_trajectory(handle_opening_poses["push_waypoints"])
        
    def open_microwave(self, speed: str) -> None:
        assert self.sim.held_object_name is None
        print("Opening microwave door ...")
        self.move_to_joint_positions(self.sim.scene_description.retract_pos)
        self.move_to_joint_positions(self.sim.scene_description.fridge_door_gaze_pos)

        handle_opening_poses = self.perception_interface.perceive_handle_opening_poses("microwave")

        # visualize on rviz
        poses = []
        poses.append(handle_opening_poses["pre_grasp_pose"])
        poses.append(handle_opening_poses["grasp_pose"])
        poses.extend(handle_opening_poses["opening_waypoints"])
        poses.append(handle_opening_poses["post_release_pose"])
        poses.append(handle_opening_poses["pre_push_pose"])
        poses.append(handle_opening_poses["push_pose"])
        poses.extend(handle_opening_poses["push_waypoints"])
        poses.append(handle_opening_poses["before_above_closing_waypoint"])
        poses.append(handle_opening_poses["above_closing_waypoint"])
        poses.append(handle_opening_poses["closing_waypoint"])
        poses.extend(handle_opening_poses["closing_waypoints"])
        print(f"Visualizing {len(poses)} handle opening poses in RViz ...")
        self.rviz_interface.visualize_poses(poses, frame_id="base_link", ns="handle_opening_poses")

        # self.move_to_joint_positions(self.sim.scene_description.home_pos)
        self.move_to_joint_positions(self.sim.scene_description.fridge_door_staging_pos)

        self.move_to_ee_pose(handle_opening_poses["pre_grasp_pose"])
        self.open_gripper()
        self.move_to_ee_pose(handle_opening_poses["grasp_pose"])
        self.close_gripper()
        # self.move_to_ee_pose(handle_opening_poses["post_grasp_pose"])
        self.move_to_ee_pose_trajectory(handle_opening_poses["opening_waypoints"])
        self.open_gripper()
        self.move_to_ee_pose(handle_opening_poses["post_release_pose"])
        
        # self.move_to_joint_positions(self.sim.scene_description.fridge_door_intermediate_restract_pos)

        self.move_to_ee_pose(handle_opening_poses["pre_push_pose"])
        self.move_to_ee_pose(handle_opening_poses["push_pose"])
        self.move_to_ee_pose_trajectory(handle_opening_poses["push_waypoints"])

        self.move_to_ee_pose(handle_opening_poses["before_above_closing_waypoint"])
        self.move_to_ee_pose(handle_opening_poses["above_closing_waypoint"])
        self.move_to_ee_pose(handle_opening_poses["closing_waypoint"])
        self.move_to_ee_pose_trajectory(handle_opening_poses["closing_waypoints"])

        self.close_gripper()
        self.move_to_ee_pose(handle_opening_poses["offset_closing_waypoints"][0])
        self.move_to_ee_pose_trajectory(handle_opening_poses["offset_closing_waypoints"])