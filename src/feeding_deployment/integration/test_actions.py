"""Testing pick and stow tool actions of the integrated system."""

from pathlib import Path
import shutil
import queue
import time

try:
    import rospy

    ROSPY_IMPORTED = True
except ModuleNotFoundError:
    ROSPY_IMPORTED = False

from relational_structs import Object
from pybullet_helpers.geometry import Pose
from pybullet_helpers.link import get_relative_link_pose

from feeding_deployment.actions.base import tool_type, table_type, plate_type, appliance_type, holder_type
from feeding_deployment.actions.acquisition import AcquireBiteHLA
from feeding_deployment.actions.flair.flair import FLAIR
from feeding_deployment.actions.pick_tool import PickToolHLA
from feeding_deployment.actions.pick_plate import PickPlateFromApplianceHLA, PickPlateFromHolderHLA, PickPlateFromTableHLA
from feeding_deployment.actions.place_plate import PlacePlateInApplianceHLA, PlacePlateOnHolderHLA, PlacePlateOnTableHLA
from feeding_deployment.actions.close_door import CloseDoorHLA
from feeding_deployment.actions.open_door import OpenDoorHLA
from feeding_deployment.actions.stow_tool import StowToolHLA
from feeding_deployment.actions.transfer_tool import TransferToolHLA
from feeding_deployment.preference_learning.config.mealtime_context import MEALS, food_items_for_flair
from feeding_deployment.preference_learning.config.preference_bundle import DEFAULT_BITE_ORDERING
from feeding_deployment.interfaces.perception_interface import PerceptionInterface
from feeding_deployment.interfaces.web_interface import WebInterface
from feeding_deployment.integration.data_logger import DataLogger
from feeding_deployment.interfaces.rviz_interface import RVizInterface
from feeding_deployment.control.robot_controller.arm_client import ArmInterfaceClient
from feeding_deployment.control.wrist_controller.wrist_controller import WristInterface
from feeding_deployment.simulation.scene_description import create_scene_description_from_config
from feeding_deployment.simulation.simulator import FeedingDeploymentPyBulletSimulator


# Standalone bite-acquisition test defaults: the strawberries + whipped cream
# meal exercises skewer-then-dip, and the ordering preference asks for the dip
# explicitly (with the deployment default, "no particular order", FLAIR's
# preference planner may return a bare bite and never dip). Pass
# --bite_ordering to test other orderings, e.g. DEFAULT_BITE_ORDERING.
DEFAULT_TEST_MEAL = "chicken nuggets and ketchup"
DEFAULT_TEST_BITE_ORDERING = "dip every chicken nugget in ketchup"

def _tool_id(sim, tool: str) -> int:
    return {"utensil": sim.utensil_id, "drink": sim.drink_id, "wipe": sim.wipe_id}[tool]


def _attach_tool_to_gripper(sim, tool: str) -> None:
    """Set the sim state so the robot is holding the given tool."""
    sim.held_object_name = tool
    sim.held_object_id = _tool_id(sim, tool)
    sim.robot.set_finger_state(sim.scene_description.tool_grasp_fingers_value)
    finger_frame_id = sim.robot.link_from_name("finger_tip")
    end_effector_link_id = sim.robot.link_from_name(sim.robot.tool_link_name)
    sim.held_object_tf = get_relative_link_pose(
        sim.robot.robot_id, finger_frame_id, end_effector_link_id, sim.physics_client_id
    )


def test_PickToolHLA(tool, sim, robot_interface, perception_interface, rviz_interface, web_interface, hla_hyperparams, wrist_interface, no_waits, log_dir, run_behavior_tree_dir, execution_log, gesture_detectors_dir):

    assert tool in ["utensil", "drink", "wipe"], f"Tool {tool} not recognized"

    high_level_action = PickToolHLA(sim, robot_interface, perception_interface, rviz_interface, web_interface, hla_hyperparams, wrist_interface, None, no_waits, log_dir, run_behavior_tree_dir, execution_log, gesture_detectors_dir)

    # PickTool requires an empty gripper.
    sim.held_object_name = None

    tool_obj = Object(tool, tool_type)
    table_obj = Object("table", table_type)
    high_level_action.execute_action(objects=[tool_obj, table_obj], params={})


def test_StowToolHLA(tool, sim, robot_interface, perception_interface, rviz_interface, web_interface, hla_hyperparams, wrist_interface, no_waits, log_dir, run_behavior_tree_dir, execution_log, gesture_detectors_dir):

    assert tool in ["utensil", "drink", "wipe"], f"Tool {tool} not recognized"

    high_level_action = StowToolHLA(sim, robot_interface, perception_interface, rviz_interface, web_interface, hla_hyperparams, wrist_interface, None, no_waits, log_dir, run_behavior_tree_dir, execution_log, gesture_detectors_dir)

    # StowTool requires the tool to be held.
    _attach_tool_to_gripper(sim, tool)
    if robot_interface is not None:
        rviz_interface.tool_update(True, sim.held_object_name, Pose((0, 0, 0), (0, 0, 0, 1)))

    tool_obj = Object(tool, tool_type)
    table_obj = Object("table", table_type)
    high_level_action.execute_action(objects=[tool_obj, table_obj], params={})


# Plate locations and the HLA + PDDL type that go with each, for both directions.
# The table entry covers whichever physical table the mealtime setting resolves to
# (the HLA picks pick_plate_from_{dining,movable}_table.yaml itself; with no
# preference context in this harness that is the movable table).
_PICK_PLATE_HLAS = {
    "table": (PickPlateFromTableHLA, table_type),
    "holder": (PickPlateFromHolderHLA, holder_type),
    "fridge": (PickPlateFromApplianceHLA, appliance_type),
    "microwave": (PickPlateFromApplianceHLA, appliance_type),
}
_PLACE_PLATE_HLAS = {
    "table": (PlacePlateOnTableHLA, table_type),
    "holder": (PlacePlateOnHolderHLA, holder_type),
    "fridge": (PlacePlateInApplianceHLA, appliance_type),
    "microwave": (PlacePlateInApplianceHLA, appliance_type),
}


def test_PickPlateHLA(location, sim, robot_interface, perception_interface, rviz_interface, web_interface, hla_hyperparams, wrist_interface, no_waits, log_dir, run_behavior_tree_dir, execution_log, gesture_detectors_dir):

    assert location in _PICK_PLATE_HLAS, f"Location {location} not recognized"

    hla_cls, location_type = _PICK_PLATE_HLAS[location]
    high_level_action = hla_cls(sim, robot_interface, perception_interface, rviz_interface, web_interface, hla_hyperparams, wrist_interface, None, no_waits, log_dir, run_behavior_tree_dir, execution_log, gesture_detectors_dir)

    # PickPlate requires an empty gripper (the pick skills assert this).
    sim.held_object_name = None

    plate_obj = Object("plate", plate_type)
    location_obj = Object(location, location_type)
    # Speed / PlateHandleColor / PlateHandleColorTolerance come from the behavior tree's parameter defaults.
    high_level_action.execute_action(objects=[plate_obj, location_obj], params={})


def test_PlacePlateHLA(location, sim, robot_interface, perception_interface, rviz_interface, web_interface, hla_hyperparams, wrist_interface, no_waits, log_dir, run_behavior_tree_dir, execution_log, gesture_detectors_dir):

    assert location in _PLACE_PLATE_HLAS, f"Location {location} not recognized"

    hla_cls, location_type = _PLACE_PLATE_HLAS[location]
    high_level_action = hla_cls(sim, robot_interface, perception_interface, rviz_interface, web_interface, hla_hyperparams, wrist_interface, None, no_waits, log_dir, run_behavior_tree_dir, execution_log, gesture_detectors_dir)

    # PlacePlate runs with the plate held. The place skills don't assert on it
    # (their held-object asserts are commented out), but keep the sim state
    # honest so a pick chained afterwards sees a full gripper.
    sim.held_object_name = "plate"

    plate_obj = Object("plate", plate_type)
    location_obj = Object(location, location_type)
    # Speed comes from the behavior tree's parameter defaults.
    high_level_action.execute_action(objects=[plate_obj, location_obj], params={})


def test_CloseDoorHLA(location, sim, robot_interface, perception_interface, rviz_interface, web_interface, hla_hyperparams, wrist_interface, no_waits, log_dir, run_behavior_tree_dir, execution_log, gesture_detectors_dir):

    assert location in ["fridge", "microwave"], f"Location {location} not recognized"

    high_level_action = CloseDoorHLA(sim, robot_interface, perception_interface, rviz_interface, web_interface, hla_hyperparams, wrist_interface, None, no_waits, log_dir, run_behavior_tree_dir, execution_log, gesture_detectors_dir)

    # CloseDoor requires an empty gripper.
    sim.held_object_name = None

    appliance_obj = Object(location, appliance_type)
    # Speed comes from the behavior tree's parameter defaults.
    high_level_action.execute_action(objects=[appliance_obj], params={})


def test_OpenDoorHLA(location, sim, robot_interface, perception_interface, rviz_interface, web_interface, hla_hyperparams, wrist_interface, no_waits, log_dir, run_behavior_tree_dir, execution_log, gesture_detectors_dir):

    assert location in ["fridge", "microwave"], f"Location {location} not recognized"

    high_level_action = OpenDoorHLA(sim, robot_interface, perception_interface, rviz_interface, web_interface, hla_hyperparams, wrist_interface, None, no_waits, log_dir, run_behavior_tree_dir, execution_log, gesture_detectors_dir)

    # OpenDoor requires an empty gripper.
    sim.held_object_name = None

    appliance_obj = Object(location, appliance_type)
    # Speed comes from the behavior tree's parameter defaults.
    high_level_action.execute_action(objects=[appliance_obj], params={})


def test_AcquireBiteHLA(sim, robot_interface, perception_interface, rviz_interface, web_interface, hla_hyperparams, wrist_interface, flair, no_waits, log_dir, run_behavior_tree_dir, execution_log, gesture_detectors_dir):

    high_level_action = AcquireBiteHLA(sim, robot_interface, perception_interface, rviz_interface, web_interface, hla_hyperparams, wrist_interface, flair, no_waits, log_dir, run_behavior_tree_dir, execution_log, gesture_detectors_dir)

    # AcquireBite runs with the utensil already held (normally after PickUtensil).
    _attach_tool_to_gripper(sim, "utensil")
    if robot_interface is not None:
        rviz_interface.tool_update(True, sim.held_object_name, Pose((0, 0, 0), (0, 0, 0, 1)))

    utensil_obj = Object("utensil", tool_type)
    table_obj = Object("table", table_type)
    # Speed / FoodDippingDepth / SkeweringDepth / SkeweringOrientation /
    # BiteSelectionAutocontinueSeconds / PickupConfirm come from the behavior
    # tree's parameter defaults.
    high_level_action.execute_action(objects=[utensil_obj, table_obj], params={})


def test_TransferToolHLA(tool, sim, robot_interface, perception_interface, rviz_interface, web_interface, hla_hyperparams, wrist_interface, no_waits, log_dir, run_behavior_tree_dir, execution_log, gesture_detectors_dir):

    assert tool in ["utensil", "drink", "wipe"], f"Tool {tool} not recognized"

    high_level_action = TransferToolHLA(sim, robot_interface, perception_interface, rviz_interface, web_interface, hla_hyperparams, wrist_interface, None, no_waits, log_dir, run_behavior_tree_dir, execution_log, gesture_detectors_dir)

    # Transfer runs with the tool held (normally after PickTool + AcquireBite).
    # Harmless to re-attach when AcquireBite already did it.
    _attach_tool_to_gripper(sim, tool)
    if robot_interface is not None:
        rviz_interface.tool_update(True, sim.held_object_name, Pose((0, 0, 0), (0, 0, 0, 1)))

    tool_obj = Object(tool, tool_type)
    table_obj = Object("table", table_type)
    # Speed / the transfer cues and signals / OutsideMouthDistance /
    # TaskReselectionAutocontinueSeconds / RetractAfterTransfer come from the
    # behavior tree's parameter defaults. Inside- vs outside-mouth transfer is
    # the scene description's transfer_type (--transfer_type, default outside).
    high_level_action.execute_action(objects=[tool_obj, table_obj], params={})


def _make_flair(log_dir: Path, perception_interface, meal: str, bite_ordering: str, allow_dip: bool, use_interface: bool):
    """Build a FLAIR instance set up the way the preference session sets it up
    before feeding (food items derived from the meal, bite-ordering preference,
    dip allowed), so AcquireBite can run standalone without a preference session.

    AcquireBite asserts flair.is_preference_set(); food items + preference are
    normally pushed by PreferenceSession._apply_food_items / apply_bite_ordering."""
    grounded_sam = getattr(perception_interface, "_grounded_sam", None)
    flair = FLAIR(log_dir, grounded_sam=grounded_sam, history_dir=log_dir)

    food_items = food_items_for_flair(meal)  # raises KeyError if not in the catalog
    flair.set_food_items(food_items)
    flair.set_preference(bite_ordering)
    flair.set_allow_dip(allow_dip)
    print(f"FLAIR set up for meal {meal!r}: solids={food_items['solid']}, dips={food_items['dip']}")
    print(f"  bite ordering: {bite_ordering!r}; dipping {'allowed' if allow_dip else 'suppressed'}")
    if food_items["dip"] and allow_dip and not use_interface:
        # acquire_bite's no-web-interface branch selects the bite autonomously
        # with dip_type hardcoded to "No dip", so the dip skill never runs. The
        # dip only happens via the web app's bite-selection page (it defaults to
        # the predicted dip and auto-continues after
        # BiteSelectionAutocontinueSeconds).
        print("  WARNING: without --use_interface the dip is skipped (skewer only) -- rerun with --use_interface to test dipping")
    return flair


def _seed_handle_opening_poses(log_dir: Path, handle_poses_pkl: str | None) -> None:
    """Copy a handle_opening_pos.pkl into this run's log dir so CloseDoorHLA can
    run without OpenDoor first: perceive_handle_closing_poses does no perception,
    it only loads that pickle (written by the last perceive_handle_opening_poses).

    NOTE: the poses are in the arm base frame -- they only match reality if the
    base is parked at the appliance where the opening was perceived."""
    if handle_poses_pkl is not None:
        src = Path(handle_poses_pkl)
        assert src.exists(), f"Handle poses pickle not found: {src}"
    else:
        candidates = sorted(
            (Path(__file__).parent / "log").glob("*/handle_opening_pos.pkl"),
            key=lambda p: p.stat().st_mtime,
        )
        assert candidates, (
            "No handle_opening_pos.pkl found under integration/log/*/ -- "
            "run OpenDoor once or pass --handle_poses_pkl"
        )
        src = candidates[-1]
    shutil.copy(src, log_dir / "handle_opening_pos.pkl")
    mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(src.stat().st_mtime))
    print(f"Seeded handle opening poses from {src} (written {mtime})")


def _main(
    scene_config: str, transfer_type: str, run_on_robot: bool, use_interface: bool, simulate_head_perception: bool, use_gui: bool, max_motion_planning_time: float = 10, tool: str = "utensil", no_waits: bool = False, action: str = "tool", location: str = "table", handle_poses_pkl: str = None, meal: str = DEFAULT_TEST_MEAL, bite_ordering: str = DEFAULT_TEST_BITE_ORDERING, no_dip: bool = False, place_location: str = "holder"
) -> None:
    """Testing pick and stow tool actions."""

    if ROSPY_IMPORTED:
        rospy.init_node("test_actions")
    else:
        assert not run_on_robot, "Need ROS to run on robot"

    # logs are saved in user/scenario directory
    log_dir = Path(__file__).parent / "log" / "test_actions"
    # if log_dir.exists():
    #     shutil.rmtree(log_dir)
    # log_dir.mkdir(parents=True, exist_ok=True)

    execution_log = Path(__file__).parent / "log" / "execution_log.txt" # in root log directory
    run_behavior_tree_dir = log_dir / "behavior_trees"
    gesture_detectors_dir = log_dir / "gesture_detectors"

    # # Copy the initial behavior trees into a directory for this run.
    # run_behavior_tree_dir.mkdir(exist_ok=True)
    # original_behavior_tree_dir = Path(__file__).parents[1] / "actions" / "behavior_trees"
    # assert original_behavior_tree_dir.exists()
    # for original_bt_filename in original_behavior_tree_dir.glob("*.yaml"):
    #     shutil.copy(original_bt_filename, run_behavior_tree_dir)

    # # Copy the initial gesture detection file into a directory for this run.
    # gesture_detectors_dir.mkdir(exist_ok=True)
    # original_gesture_detection_filepath = Path(__file__).parents[1] / "perception" / "gestures_perception" / "synthesized_gesture_detectors.py"
    # assert original_gesture_detection_filepath.exists()
    # shutil.copy(original_gesture_detection_filepath, gesture_detectors_dir)

    # Initialize the interface to the robot.
    if run_on_robot:
        robot_interface = ArmInterfaceClient()  # type: ignore  # pylint: disable=no-member
        wrist_interface = WristInterface()
    else:
        robot_interface = None
        wrist_interface = None

    data_logger = DataLogger(state_dir=log_dir)

    if use_interface:
        task_selection_queue = queue.Queue()
        web_interface = WebInterface(task_selection_queue=task_selection_queue, data_logger=data_logger)
    else:
        web_interface = None

    # Initialize the perceiver (e.g., get joint states or human head poses).
    perception_interface = PerceptionInterface(robot_interface=robot_interface, simulate_head_perception=simulate_head_perception, data_logger=data_logger)

    scene_config_path = Path(__file__).parent.parent / "simulation" / "configs" / f"{scene_config}.yaml"
    scene_description = create_scene_description_from_config(str(scene_config_path), transfer_type)
    sim = FeedingDeploymentPyBulletSimulator(scene_description, use_gui=use_gui)

    if robot_interface is not None:
        rviz_interface = RVizInterface(scene_description)
    else:
        rviz_interface = None

    hla_hyperparams = {"max_motion_planning_time": max_motion_planning_time}

    if action in ("acquire_bite", "acquire_transfer_bite"):
        # Acquire one bite with the utensil already held: real detection +
        # FLAIR's next-bite prediction, then skewer (and dip, if the planner
        # picks one). No preference session runs here, so FLAIR is set up from
        # --meal / --bite_ordering directly.
        assert run_on_robot, f"{action} needs --run_on_robot (real detection + FLAIR prediction)"
        flair = _make_flair(log_dir, perception_interface, meal, bite_ordering, allow_dip=not no_dip, use_interface=use_interface)
        test_AcquireBiteHLA(sim, robot_interface, perception_interface, rviz_interface, web_interface, hla_hyperparams, wrist_interface, flair, no_waits, log_dir, run_behavior_tree_dir, execution_log, gesture_detectors_dir)
        if action == "acquire_transfer_bite":
            # Straight into the transfer, the way the executive sequences them:
            # the bite is on the fork and the gripper state carries over.
            print("Bite acquired -- transferring it now")
            test_TransferToolHLA("utensil", sim, robot_interface, perception_interface, rviz_interface, web_interface, hla_hyperparams, wrist_interface, no_waits, log_dir, run_behavior_tree_dir, execution_log, gesture_detectors_dir)
    elif action == "transfer":
        # Bring the held tool to the user's mouth. Nothing is acquired first, so
        # for the utensil put a bite on the fork by hand to test with food.
        test_TransferToolHLA(tool, sim, robot_interface, perception_interface, rviz_interface, web_interface, hla_hyperparams, wrist_interface, no_waits, log_dir, run_behavior_tree_dir, execution_log, gesture_detectors_dir)
    elif action == "pick_plate":
        # Pick the plate from the given location (table / holder / fridge / microwave).
        test_PickPlateHLA(location, sim, robot_interface, perception_interface, rviz_interface, web_interface, hla_hyperparams, wrist_interface, no_waits, log_dir, run_behavior_tree_dir, execution_log, gesture_detectors_dir)
    elif action == "place_plate":
        # Place the held plate at the given location (table / holder / fridge /
        # microwave). Nothing picks it up first, so start with the plate in the
        # gripper.
        test_PlacePlateHLA(place_location, sim, robot_interface, perception_interface, rviz_interface, web_interface, hla_hyperparams, wrist_interface, no_waits, log_dir, run_behavior_tree_dir, execution_log, gesture_detectors_dir)
    elif action == "pick_place_plate":
        # Pick the plate up and set it down somewhere else, back to back --
        # default table -> holder, the sequence the executive runs at the end of
        # a meal. The pick leaves the plate in the gripper for the place.
        for i in range(10):
            print(f"Picking the plate from the {location}, then placing it on the {place_location}")
            test_PickPlateHLA(location, sim, robot_interface, perception_interface, rviz_interface, web_interface, hla_hyperparams, wrist_interface, no_waits, log_dir, run_behavior_tree_dir, execution_log, gesture_detectors_dir)
            print(f"Plate picked from the {location} -- placing it on the {place_location} now")
            test_PlacePlateHLA(place_location, sim, robot_interface, perception_interface, rviz_interface, web_interface, hla_hyperparams, wrist_interface, no_waits, log_dir, run_behavior_tree_dir, execution_log, gesture_detectors_dir)
    elif action == "close_door":
        # Close the door of the given appliance (fridge / microwave), reusing
        # opening poses perceived by an earlier OpenDoor run.
        _seed_handle_opening_poses(log_dir, handle_poses_pkl)
        test_CloseDoorHLA(location, sim, robot_interface, perception_interface, rviz_interface, web_interface, hla_hyperparams, wrist_interface, no_waits, log_dir, run_behavior_tree_dir, execution_log, gesture_detectors_dir)
    elif action in ("open_door", "open_close_door"):
        # Open the door of the given appliance (fridge / microwave) with real
        # perception (terminal confirmation when --use_interface is not set).
        test_OpenDoorHLA(location, sim, robot_interface, perception_interface, rviz_interface, web_interface, hla_hyperparams, wrist_interface, no_waits, log_dir, run_behavior_tree_dir, execution_log, gesture_detectors_dir)
        # Preserve the perceived opening poses outside this run's log dir (which
        # the next test run rmtree's), so a later close_door run picks them up
        # by default (newest handle_opening_pos.pkl under integration/log/*/).
        produced = log_dir / "handle_opening_pos.pkl"
        if produced.exists():
            keep_dir = log_dir.parent / "test_actions_open"
            keep_dir.mkdir(exist_ok=True)
            shutil.copy(produced, keep_dir / "handle_opening_pos.pkl")
            print(f"Preserved opening poses at {keep_dir / 'handle_opening_pos.pkl'} -- a later close_door run will use them by default")
        if action == "open_close_door":
            # Close right after: the opening poses just written to this run's
            # log dir are exactly what perceive_handle_closing_poses loads.
            test_CloseDoorHLA(location, sim, robot_interface, perception_interface, rviz_interface, web_interface, hla_hyperparams, wrist_interface, no_waits, log_dir, run_behavior_tree_dir, execution_log, gesture_detectors_dir)
    else:
        # Pick the tool, then stow it.
        test_PickToolHLA(tool, sim, robot_interface, perception_interface, rviz_interface, web_interface, hla_hyperparams, wrist_interface, no_waits, log_dir, run_behavior_tree_dir, execution_log, gesture_detectors_dir)
        test_StowToolHLA(tool, sim, robot_interface, perception_interface, rviz_interface, web_interface, hla_hyperparams, wrist_interface, no_waits, log_dir, run_behavior_tree_dir, execution_log, gesture_detectors_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_config", type=str, default="vention")
    parser.add_argument("--transfer_type", type=str, default="outside")
    parser.add_argument("--run_on_robot", action="store_true")
    parser.add_argument("--use_interface", action="store_true")
    parser.add_argument("--simulate_head_perception", action="store_true")
    parser.add_argument("--use_gui", action="store_true")
    parser.add_argument("--max_motion_planning_time", type=float, default=10.0)
    parser.add_argument("--tool", type=str, default="utensil")
    parser.add_argument("--no_waits", action="store_true")
    parser.add_argument("--action", type=str, default="tool", choices=["tool", "pick_plate", "open_door", "close_door", "open_close_door", "acquire_bite", "transfer", "acquire_transfer_bite", "place_plate", "pick_place_plate"])
    parser.add_argument("--location", type=str, default="table", choices=["table", "holder", "fridge", "microwave"],
                        help="where the plate is picked from (pick_plate / pick_place_plate); the appliance for the door actions")
    parser.add_argument("--place_location", type=str, default="holder", choices=["table", "holder", "fridge", "microwave"],
                        help="where the plate is put down (place_plate / pick_place_plate)")
    parser.add_argument("--handle_poses_pkl", type=str, default=None, help="handle_opening_pos.pkl to close from (default: newest under integration/log/*/)")
    parser.add_argument("--meal", type=str, default=DEFAULT_TEST_MEAL, choices=MEALS, help="acquire_bite: catalog meal whose items FLAIR detects (solids + sauces)")
    parser.add_argument("--bite_ordering", type=str, default=DEFAULT_TEST_BITE_ORDERING, help=f"acquire_bite: bite-ordering preference given to FLAIR's planner (deployment default is {DEFAULT_BITE_ORDERING!r})")
    parser.add_argument("--no_dip", action="store_true", help="acquire_bite: suppress dipping (the 'do not dip' preference)")
    args = parser.parse_args()

    _main(args.scene_config, args.transfer_type, args.run_on_robot, args.use_interface, args.simulate_head_perception, args.use_gui, args.max_motion_planning_time, args.tool, args.no_waits, args.action, args.location, args.handle_poses_pkl, args.meal, args.bite_ordering, args.no_dip, args.place_location)
