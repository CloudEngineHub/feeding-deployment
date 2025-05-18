from typing import Any, Callable

import numpy as np
import time
import pickle
from scipy.spatial.transform import Rotation
import inspect
from pathlib import Path
import threading
import types
import json
import imageio

from pybullet_helpers.geometry import Pose

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
    tool_type,
    GripperFree,
    Holding,
    ToolPrepared,
    EmulateTransferDone,
)
from feeding_deployment.actions.feel_the_bite.outside_mouth_transfer import OutsideMouthTransfer
from feeding_deployment.perception.gestures_perception.synthesizer import PersonalizedGestureDetectorSynthesizer
from feeding_deployment.perception.gestures_perception.static_gesture_detectors import mouth_open, head_nod

class EmulateTransferHLA(HighLevelAction):
    """Emulate transfer by bringing the empty gripper in front of the user's mouth."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.transfer = OutsideMouthTransfer(self.sim, self.robot_interface, self.perception_interface, self.rviz_interface, self.no_waits)

        self.gesture_examples_path = self.gesture_detectors_dir / "gesture_examples"
        self.synthesized_gestures_dict_path = self.gesture_detectors_dir / "synthesized_gestures_dict.json"
        if not self.gesture_examples_path.exists():
            self.gesture_examples_path.mkdir(parents=True)
        if not self.synthesized_gestures_dict_path.exists():
            with open(self.synthesized_gestures_dict_path, "w") as f:
                f.write("{}")
        self.detector_synthesizer = PersonalizedGestureDetectorSynthesizer(self.log_dir)

        self.test_mode = False

    def emulate_transfer(self, speed: str):

        if self.robot_interface is not None:
            self.robot_interface.set_speed(speed)

        self.move_to_joint_positions(self.sim.scene_description.before_transfer_pos)

        if self.web_interface is not None:
            self.web_interface.fix_explanation("Moving to infront of mouth")

        self.perception_interface.set_head_perception_tool("fork")
        self.perception_interface.start_head_perception_thread()
        if self.robot_interface is not None:
            time.sleep(5.0) # let head perception thread warmstart / robot to stabilize
            self.robot_interface.set_tool("fork")
            self.perception_interface.zero_ft_sensor()
        else:
            time.sleep(1.0) # let sim head perception thread warmstart

        if self.robot_interface is not None:
            self.perception_interface.speak("Please look towards the robot when ready")

        if self.robot_interface and not self.test_mode:
                input("Press enter if user is looking at the robot")
                self.perception_interface.log_looking_at_robot_head_perception()
                print("Logged looking at robot head perception data")
                input("Press enter to start the transfer")
        
        self.transfer.set_tool("fork")
        self.transfer.move_to_transfer_state(outside_mouth_distance=0.05)

        if self.robot_interface is not None:
            time.sleep(5.0)
            # self.perception_interface.detect_force_trigger()

        # shutdown the head perception thread and move to before transfer state
        self.perception_interface.stop_head_perception_thread()
        self.transfer.move_to_before_transfer_state()        

    def get_name(self) -> str:
        return "EmulateTransfer"
    
    def get_operator(self) -> LiftedOperator:
        return LiftedOperator(
            self.get_name(),
            parameters=[],
            preconditions={LiftedAtom(GripperFree, [])},
            add_effects={LiftedAtom(EmulateTransferDone, [])},
            delete_effects=set(),
        )
    
    def get_behavior_tree_filename(
        self,
        objects: tuple[Object, ...],
        params: dict[str, Any],
    ) -> str:
        return f"emulate_transfer.yaml"
    
    def execute_action(
        self,
        objects: tuple[Object, ...],
        params: dict[str, Any],
    ) -> None:
        print("Params: ", params)
        if params["test_mode"]:
            self.test_mode = True
        else:
            self.test_mode = False
        return super().execute_action(objects, params)
