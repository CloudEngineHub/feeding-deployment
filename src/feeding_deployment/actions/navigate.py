from typing import Any

from relational_structs import (
    LiftedAtom,
    LiftedOperator,
    Object,
    Variable,
)
from feeding_deployment.actions.base import (
    HighLevelAction,
    nav_target_type,
    InFrontOf,
)

class NavigateHLA(HighLevelAction):
    """Navigate to a target location/object."""

    def get_name(self) -> str:
        return "Navigate"

    def get_operator(self) -> LiftedOperator:
        obj = Variable("?obj", nav_target_type)
        return LiftedOperator(
            self.get_name(),
            parameters=[obj],
            preconditions=set(),
            add_effects={LiftedAtom(InFrontOf, [obj])},
            delete_effects=set(),
        )
    
    def get_behavior_tree_filename(
        self,
        objects: tuple[Object, ...],
        params: dict[str, Any],
    ) -> str:
        del params
        assert len(objects) == 1
        obj = objects[0]
        assert self.sim.scene_description.scene_label == "vention"
        assert obj.name in ["fridge", "microwave", "sink", "feeding_table"]
        return f"navigate_to_{obj.name}.yaml"

    def navigate_to_fridge(self, speed: str) -> None:
        print("Navigating to fridge ...")

    def navigate_to_microwave(self, speed: str) -> None:
        print("Navigating to microwave ...")

    def navigate_to_sink(self, speed: str) -> None:
        print("Navigating to sink ...")

    def navigate_to_feeding_table(self, speed: str) -> None:
        print("Navigating to feeding table ...")