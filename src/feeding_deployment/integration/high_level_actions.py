"""High-level actions that we can simulate and execute."""

import abc

from relational_structs import LiftedOperator, Predicate, Type, GroundOperator, Object
from feeding_deployment.robot_controller.arm_client import KinovaCommand

# Define some predicates that can be used for sequencing the high-level actions.
tool_type = Type("tool")  # utensil, cup, or wiping tool
GripperFree = Predicate("GripperFree", [])  # not holding any tool
Holding = Predicate("Holding", [tool_type])  # holding tool
ToolTransferDone = Predicate("ToolTransferDone", [tool_type])  # wiped, drank, or ate
ToolPrepared = Predicate("ToolPrepared", [tool_type])  # e.g., bite acquired


# Define high-level actions.
class HighLevelAction(abc.ABC):
    """Base class for high-level action."""

    def __init__(self, objects: tuple[Object]):
        self._objects = objects

    @abc.abstractmethod
    def get_operator(self) -> GroundOperator:
        """Create a planning operator for this HLA."""

    @abc.abstractmethod
    def get_robot_commands(self, sim) -> list[KinovaCommand]:
        """Return a list of commands to send to the robot to execute this.
        
        TODO: add type for sim.
        """

    @abc.abstractmethod
    def update_simulator(self, sim) -> None:
        """Update the given simulator assuming that the action was executed.
        
        TODO: add typing for sim
        TODO; add argument for any sensory inputs
        """


class PickToolHLA(HighLevelAction):
    """Pick up a tool (utensil, drink, or wipe)."""

    def __init__(self, objects: tuple[Object]):
        assert len(objects) == 1
        super().__init__(objects)
        self._tool = objects[0]

    def get_operator(self) -> GroundOperator:
        tool = tool_type("?tool")
        lifted_operator = LiftedOperator("PickTool",
                                        parameters=[tool],
                                        preconditions={GripperFree([])},
                                        add_effects={Holding([tool])},
                                        delete_effects={GripperFree([])})
        return lifted_operator.ground(self._objects)
    
    def get_robot_commands(self, sim) -> list[KinovaCommand]:
        import ipdb; ipdb.set_trace()

    def update_simulator(self, sim) -> None:
        import ipdb; ipdb.set_trace()


    
class StowToolHLA(HighLevelAction):
    """Stow a tool (utensil, drink, or wipe)."""

    def __init__(self, objects: tuple[Object]):
        assert len(objects) == 1
        super().__init__(objects)
        self._tool = objects[0]

    def get_operator(self) -> GroundOperator:
        tool = tool_type("?tool")
        lifted_operator = LiftedOperator("StowTool",
                                parameters=[tool],
                                preconditions={Holding([tool])},
                                add_effects={GripperFree([])},
                                delete_effects={Holding([tool])})
        return lifted_operator.ground(self._objects)
    
    def get_robot_commands(self, sim) -> list[KinovaCommand]:
        import ipdb; ipdb.set_trace()

    def update_simulator(self, sim) -> None:
        import ipdb; ipdb.set_trace()


class TransferToolHLA(HighLevelAction):
    """Wipe, or transfer drink, or transfer bite."""
    
    def __init__(self, objects: tuple[Object]):
        assert len(objects) == 1
        super().__init__(objects)
        self._tool = objects[0]

    def get_operator(self) -> GroundOperator:
        tool = tool_type("?tool")
        lifted_operator = LiftedOperator("TransferTool",
                                parameters=[tool],
                                preconditions={Holding([tool]), ToolPrepared([tool])},
                                add_effects={GripperFree([])},
                                delete_effects={Holding([tool]), ToolPrepared([tool])})
        return lifted_operator.ground(self._objects)

    def get_robot_commands(self, sim) -> list[KinovaCommand]:
        import ipdb; ipdb.set_trace()

    def update_simulator(self, sim) -> None:
        import ipdb; ipdb.set_trace()


class PrepareToolHLA(HighLevelAction):
    """Bite acquisition; other tools are always prepared."""
    
    def __init__(self, objects: tuple[Object]):
        assert len(objects) == 1
        super().__init__(objects)
        self._tool = objects[0]

    def get_operator(self) -> GroundOperator:
        tool = tool_type("?tool")
        lifted_operator = LiftedOperator("PrepareTool",
                            parameters=[tool],
                            preconditions={Holding([tool])},
                            add_effects={ToolPrepared([tool])},
                            delete_effects=set())
        return lifted_operator.ground(self._objects)

    def get_robot_commands(self, sim) -> list[KinovaCommand]:
        import ipdb; ipdb.set_trace()

    def update_simulator(self, sim) -> None:
        import ipdb; ipdb.set_trace()
