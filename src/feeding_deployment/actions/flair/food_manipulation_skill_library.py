import numpy as np
from scipy.spatial.transform import Rotation
import threading

# ros imports
try:
    import rospy
    import tf2_ros
    from geometry_msgs.msg import Point
    from sensor_msgs.msg import JointState
    from std_msgs.msg import String, Float64, Bool
    ROSPY_IMPORTED = True
except ModuleNotFoundError:
    ROSPY_IMPORTED = False

from feeding_deployment.utils.pixel_selector import PixelSelector
from feeding_deployment.utils.camera_utils import angle_between_pixels, pixel2World, world2Pixel
from feeding_deployment.utils.tf_utils import TFUtils
from feeding_deployment.simulation.simulator import FeedingDeploymentPyBulletSimulator

from feeding_deployment.interfaces.perception_interface import PerceptionInterface
from feeding_deployment.interfaces.rviz_interface import RVizInterface
from feeding_deployment.control.robot_controller.arm_client import ArmInterfaceClient
from feeding_deployment.control.wrist_controller.wrist_controller import WristInterface
from feeding_deployment.control.robot_controller.command_interface import (
    CartesianCommand,
    CloseGripperCommand,
    JointCommand,
    KinovaCommand,
    OpenGripperCommand,
)

from pybullet_helpers.geometry import Pose

# Height of the dipping sauce's surface above the surface the plate-depth sample
# sees. The camera looks down, so the sauce is that much CLOSER than the plate --
# hence plate_depth - this = sauce depth. The bowl measures 4 cm tall; 43 mm is
# what the whipped cream actually read in the Jul 24 test_actions frames (the
# valid depths inside the sauce mask cluster at 358-362 mm against a 404 mm plate,
# i.e. 42-46 mm above it). Re-measure if the bowl or the plate changes.
BOWL_RIM_ABOVE_PLATE_MM = 43.0


def plate_surface_depth_mm(depth_image, plate_bounds):
    """Median valid depth (mm) over the middle of the detected plate, or None.

    The plate is the most reliable depth target in the scene -- large, matte and
    always in frame -- which is what makes it a usable stand-in for a dipping
    sauce, whose white, glossy surface routinely returns nothing. Only the central
    half of the plate's bounding box is sampled, so the rim and the table just
    outside it stay out of the median."""
    if plate_bounds is None:
        return None
    x, y, w, h = (int(v) for v in plate_bounds)
    if w <= 0 or h <= 0:
        return None
    height, width = depth_image.shape[:2]
    x0, x1 = max(x + w // 4, 0), min(x + 3 * w // 4, width)
    y0, y1 = max(y + h // 4, 0), min(y + 3 * h // 4, height)
    if x1 <= x0 or y1 <= y0:
        return None
    region = depth_image[y0:y1, x0:x1]
    valid = region[np.isfinite(region) & (region > 50) & (region < 1000)]
    if valid.size == 0:
        return None
    return float(np.median(valid))


class FoodManipulationSkillLibrary:
    def __init__(self, sim : FeedingDeploymentPyBulletSimulator, robot_interface: ArmInterfaceClient, wrist_interface: WristInterface, perception_interface: PerceptionInterface, rviz_interface: RVizInterface, no_waits=False):
        
        self.sim = sim
        self.robot_interface = robot_interface
        self.wrist_interface = wrist_interface
        self.perception_interface = perception_interface
        self.rviz_interface = rviz_interface
        self.no_waits = no_waits

        if self.sim.scene_description.scene_label == "wheelchair":
            self.plate_height = 0.12
        elif self.sim.scene_description.scene_label == "vention":
            # self.plate_height = 0.155 # for silicone fork
            # self.plate_height = 0.158 # for metal fork
            # self.plate_height = 0.16 # for metal fork
            # self.plate_height = 0.185
            # self.plate_height = 0.197 # green table
            # self.plate_height = 0.221
            self.plate_height = 0.215 # movable table
            # self.plate_height = 0.225 # dining table (used in social meals)
        else:
            raise NotImplementedError("Scene label not recognized; plate height required for bite acquisition")

        self.pixel_selector = PixelSelector()
        if self.robot_interface is not None:
            self.tf_utils = TFUtils()

        self.cached_reset_tip_to_wrist =  np.array(
            [[ 4.97225726e-05, -1.53284719e-03, -9.99998824e-01, -1.79554958e-02],
            [-3.22083141e-02,  9.99480001e-01, -1.53365339e-03,  5.15432893e-04],
            [ 9.99481176e-01,  3.22083525e-02,  3.26293322e-07, -2.55042925e-01],
            [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]
        )

        print("Skill library initialized")

    def move_to_joint_positions(self, joint_positions):

        if self.robot_interface is None:
            plan = self.sim.plan_to_joint_positions(joint_positions)
            print("Plan has length", len(plan))
            self.sim.visualize_plan(plan)
        else:
            self.robot_interface.execute_command(JointCommand(pos=joint_positions))

    def move_to_ee_pose(self, pose, plan_override=False):

        if not plan_override and not self.no_waits:
            plan = self.sim.plan_to_ee_pose(pose)
        if self.robot_interface is None:
            self.sim.visualize_plan(plan)
        else:
            self.robot_interface.execute_command(CartesianCommand(pos=pose.position, quat=pose.orientation))

    def set_wrist_state(self, pitch_angle, roll_angle):
        if self.robot_interface is None:
            self.sim.set_wrist_state(pitch_angle, roll_angle)
        else:
            self.wrist_interface.set_wrist_state(pitch_angle, roll_angle)

    def robot_reset(self):
        self.move_to_joint_positions(self.sim.scene_description.above_plate_pos)

    def reset(self):

        print("Moving to above plate position: ", self.sim.scene_description.above_plate_pos)
        self.move_to_joint_positions(self.sim.scene_description.above_plate_pos)
        self.set_wrist_state(0, 0)

    def move_utensil_to_pose(self, tip_pose, tip_to_wrist = None):

        if self.robot_interface is not None:

            self.tf_utils.publishTransformationToTF('arm_base_link', 'fork_tip_target', tip_pose)

            if tip_to_wrist is None:
                tip_to_wrist = self.tf_utils.getTransformationFromTF('arm_fork_tip', 'arm_tool_frame')
            tool_frame_target = tip_pose @ tip_to_wrist

            self.rviz_interface.visualize_fork(tip_pose)
            self.tf_utils.publishTransformationToTF('arm_base_link', 'tool_frame_target', tool_frame_target)
            
            if not self.no_waits:
                input("Execute command?")

            pose = Pose.from_matrix(tool_frame_target)
            self.move_to_ee_pose(pose)
        else:
            if tip_to_wrist is None:
                raise ValueError("tip_to_wrist must be provided in simulation")
            
            tool_frame_target = tip_pose @ tip_to_wrist
            plan = self.sim.plan_to_ee_pose(Pose.from_matrix(tool_frame_target))
            self.sim.visualize_plan(plan)
    
    def get_transform(self, from_frame, to_frame):
        if self.robot_interface is not None:
            return self.tf_utils.getTransformationFromTF(from_frame, to_frame)
        else:
            if from_frame == "arm_fork_tip" and to_frame == "arm_tool_frame":
                tip_to_wrist = np.array([[0, 0, -1, -1.79500833e-02],
                                        [0, 1, 0, -2.66243553e-03],
                                        [1, 0, 0, -2.55099477e-01],
                                        [0, 0, 0, 1]])
                return tip_to_wrist

            pose_transform = self.sim.get_transform(from_frame, to_frame)
            return pose_transform.to_matrix()

    def skewering_skill(self, color_image, depth_image, camera_info, keypoint=None, major_axis=None, skewering_depth=0.015, action_index=0):
        if keypoint is not None:
            (center_x, center_y) = keypoint
        else:
            clicks = self.pixel_selector.run(color_image)
            (center_x, center_y) = clicks[0]
            major_axis = -np.pi/2
        
        print(f"Center x {center_x}, Center y {center_y}, Action index {action_index}")

        # get 3D point from depth image
        validity, point = pixel2World(camera_info, center_x, center_y, depth_image)
        # breakpoint()
        if not validity:
            print("Invalid point")
            return False

        print("Getting transformation from base_link to camera_color_optical_frame")
        base_to_camera_transform = self.get_transform('arm_base_link', 'camera_color_optical_frame')
        print("Base to camera transform: ", base_to_camera_transform)

        food_base = np.eye(4)
        food_base[:3,3] = point.reshape(1,3)
        food_base = base_to_camera_transform @ food_base
        print("Depth to skewer: ", food_base[2,3] - skewering_depth)
        print("Plate height: ", self.plate_height)
        # food_base[2,3] = max(food_base[2,3] - skewering_depth, self.plate_height) 
        food_base[2,3] = self.plate_height
        # magic number for skewering offset
        food_base[0,3] += 0.004 # positive moves away from the robot
        food_base[1,3] += 0.005 # positive moves left from the robot
        # keep the orientation of the food base fixed
        food_base[:3,:3] = Rotation.from_quat([-0.7071068, 0.7071068, 0, 0]).as_matrix()

        # print("Food base: ", food_base)

        if self.robot_interface is not None:
            self.tf_utils.publishTransformationToTF('arm_base_link', 'food_frame', food_base)
            self.rviz_interface.visualize_food(food_base)

        if major_axis < np.pi/2:
            major_axis = major_axis + np.pi/2

        # caching this so that the robot doesn't rotate the wrist again
        tip_to_wrist = self.get_transform('arm_fork_tip', 'arm_tool_frame')
        print("Tip to wrist: ", tip_to_wrist)
        
        # Action 0: Rotate twirl DoF to skewer angle
        self.set_wrist_state(0, -major_axis)

        # Action 1: Move to action start position
        waypoint_1_tip = np.copy(food_base)
        waypoint_1_tip[2,3] += 0.05
        # input('Press ENTER to continue to skewering action')
        self.move_utensil_to_pose(waypoint_1_tip, tip_to_wrist)

        # Action 2: Move inside food item
        waypoint_2_tip = np.copy(food_base)
        self.move_utensil_to_pose(waypoint_2_tip, tip_to_wrist)

        # Rajat ToDo: Switch to scooping pick up
        if self.robot_interface is not None:
            self.scooping_pickup()
        else:
            # Rajat ToDo: Implement scooping pick up for simulation
            self.move_utensil_to_pose(waypoint_1_tip, tip_to_wrist)

        return True

    def dipping_skill(self, color_image, depth_image, camera_info, keypoint=None, dipping_depth=0.02, plate_bounds=None):
        """ Dipping amount must be between 0.02 and 0.05"""

        if keypoint is not None:
            (center_x, center_y) = keypoint
            major_axis = -np.pi/2
        else:
            clicks = self.pixel_selector.run(color_image)
            (center_x, center_y) = clicks[0]
            major_axis = -np.pi/2

        # # visualize keypoint
        # import cv2
        # cv2.circle(color_image, (center_x, center_y), 5, (0, 0, 255), -1)
        # cv2.imshow("Color image", color_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # input("Press enter to continue...")

        # get 3D point from depth image, sampling an 11x11 patch centred on the dip
        # point instead of the default 5x5. Sauces are shiny and near-textureless,
        # so the depth right at the picked pixel is often a hole; the wider patch
        # finds a valid median nearby instead of failing the skill outright.
        validity, point = pixel2World(camera_info, center_x, center_y, depth_image, box_width=5)

        # Geometric fallback for when even the patch comes back empty (glare on
        # white sauce). The plate and the bowl stand on the same table and the
        # camera looks down at it, so the bowl rim sits at the plate's depth minus
        # the bowl height. Depth here is perpendicular distance along the optical
        # axis, so this transfers across the frame: the table is within a couple of
        # degrees of fronto-parallel, which costs ~2 mm laterally, while the rim
        # offset below is the term that actually matters.
        plate_depth_mm = plate_surface_depth_mm(depth_image, plate_bounds)
        fallback_depth_mm = None if plate_depth_mm is None else plate_depth_mm - BOWL_RIM_ABOVE_PLATE_MM

        measured_depth_mm = point[2] * 1000 if validity else None
        measured_str = f"{measured_depth_mm:.1f} mm" if validity else "INVALID (hole/glare)"
        if plate_depth_mm is None:
            fallback_str = "unavailable (no plate depth)"
        else:
            fallback_str = f"{fallback_depth_mm:.1f} mm (plate {plate_depth_mm:.1f} - bowl {BOWL_RIM_ABOVE_PLATE_MM:.0f})"

        if not validity and fallback_depth_mm is not None:
            validity, point = pixel2World(camera_info, center_x, center_y, depth_image,
                                          depth=fallback_depth_mm)
            using = "FALLBACK (plate depth - bowl height)" if validity else "NEITHER (fallback implausible)"
        elif validity:
            using = "measured patch"
        else:
            using = "NEITHER (no measurement, no plate depth)"

        print(f"[dip depth] measured: {measured_str} | fallback: {fallback_str} | using: {using}")
        if measured_depth_mm is not None and fallback_depth_mm is not None:
            # Both available: worth watching, since a large gap means either the
            # bowl height constant or the patch reading is off (a dark bowl
            # interior can read its bottom rather than the sauce surface).
            print(f"[dip depth] measured-vs-fallback disagreement: "
                  f"{abs(measured_depth_mm - fallback_depth_mm):.1f} mm")

        # breakpoint()
        if not validity:
            print("Invalid point")
            return False

        print("Getting transformation from base_link to camera_color_optical_frame")
        base_to_camera_transform = self.get_transform('arm_base_link', 'camera_color_optical_frame')
        print("Base to camera transform: ", base_to_camera_transform)

        food_base = np.eye(4)
        food_base[:3,3] = point.reshape(1,3)
        food_base = base_to_camera_transform @ food_base
        print("Food height detected: ", food_base[2,3])
        print("Plate height: ", self.plate_height)
        food_base[2,3] = self.plate_height + 0.06 - dipping_depth
        print("Food height after plate update: ", food_base[2,3])
        # food_base[2,3] = max(food_base[2,3] - dipping_depth, self.plate_height) 
        # magic number for skewering offset
        # food_base[0,3] += 0.012 # positive moves away from the robot
        # keep the orientation of the food base fixed
        food_base[:3,:3] = Rotation.from_quat([-0.7071068, 0.7071068, 0, 0]).as_matrix()

        if self.robot_interface is not None:
            self.tf_utils.publishTransformationToTF('arm_base_link', 'food_frame', food_base)
            self.rviz_interface.visualize_food(food_base)

        if major_axis < np.pi/2:
            major_axis = major_axis + np.pi/2

        # caching this so that the robot doesn't rotate the wrist again
        # tip_to_wrist = self.get_transform('arm_fork_tip', 'arm_tool_frame')
        # print("Tip to wrist: ", tip_to_wrist)
        
        # action 0: Rotate scooping DoF to dip angle
        self.wrist_interface.set_to_dip_pos()

        # Action 1: Move above food
        waypoint_1_tip = np.copy(food_base)
        waypoint_1_tip[2,3] -= 0.07
        waypoint_1_tip[2,3] += 0.13
        waypoint_1_tip[0,3] += 0.15
        self.move_utensil_to_pose(waypoint_1_tip, self.cached_reset_tip_to_wrist)

        # Action 2: Dip
        waypoint_2_tip = np.copy(food_base)
        waypoint_2_tip[2,3] -= 0.07
        waypoint_2_tip[0,3] += 0.15
        self.move_utensil_to_pose(waypoint_2_tip, self.cached_reset_tip_to_wrist)

        # Action 3: Move above food
        waypoint_3_tip = np.copy(food_base)
        waypoint_3_tip[2,3] -= 0.07
        waypoint_3_tip[2,3] += 0.13
        waypoint_3_tip[0,3] += 0.15
        self.move_utensil_to_pose(waypoint_3_tip, self.cached_reset_tip_to_wrist)

        # Action 4: Set scooping state
        self.wrist_interface.scoop_wrist()

        return True
        
    def scooping_pickup(self, hack = True):

        forkpitch_to_tip = self.get_transform('arm_forkpitch', 'arm_fork_tip')
        print("Forkpitch to tip: ", forkpitch_to_tip)
        distance = forkpitch_to_tip[0,3]

        print("Distance: ", distance)

        arm_tool_frame = self.get_transform('arm_base_link', 'arm_tool_frame')

        tool_frame_displacement = np.eye(4)
        tool_frame_displacement[0,3] = distance/10 # move down
        tool_frame_displacement[1,3] = -distance*3/4 # move back

        tool_frame_target = arm_tool_frame @ tool_frame_displacement

        if self.robot_interface is not None:
            self.tf_utils.publishTransformationToTF('arm_base_link', 'tool_frame_target', tool_frame_target)
        
        # input("Press enter to start scooping pickup")

        if self.robot_interface is not None:
            scoop_thread = threading.Thread(target=self.wrist_interface.scoop_wrist)
            scoop_thread.start()
        else:
            raise NotImplementedError("Scooping pickup not implemented for simulation")

        # input("Press enter to also move the robot...")
        self.move_to_ee_pose(Pose.from_matrix(tool_frame_target), plan_override=True) # Necessary so that robot doesn't spend time planning in simulation

        # wait for scoop thread to finish
        if self.robot_interface is not None:
            scoop_thread.join()