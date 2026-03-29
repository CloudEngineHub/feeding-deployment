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
        assert self.sim.held_object_name is None
        print("Opening fridge door ...")
        
    def open_microwave(self, speed: str) -> None:
        assert self.sim.held_object_name is None
        print("Opening microwave door ...")