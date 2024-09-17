# Description: This script is used to detect ArUco markers and estimate their pose in the camera frame.

# python imports
import os, sys
import cv2
import numpy as np
import time
import math
from scipy.spatial.transform import Rotation

# ros imports
import rospy
import message_filters
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import tf2_ros
from geometry_msgs.msg import Point
from visualization_msgs.msg import MarkerArray, Marker

from feeding_deployment.robot_controller.arm_client import ArmInterfaceClient
from feeding_deployment.robot_controller.command_interface import CartesianCommand
from geometry_msgs.msg import TransformStamped
from collections import deque

# from feeding_deployment.head_perception.ros_wrapper import HeadPerceptionROSWrapper

from geometry_msgs.msg import Pose as pose_msg

class ArUcoPerception:
    def __init__(self):
        rospy.init_node('ArUcoPerception')
        self.AR_center_pose = None # get rid of this later

        self.robot_interface = ArmInterfaceClient()
        self.bridge = CvBridge()
        self.current_pose = None
        self.goal_pose_queue = deque(maxlen=10)

        self.cartesian_state_sub = message_filters.Subscriber('/robot_cartesian_state', pose_msg)
        self.cartesian_state_sub.registerCallback(self.read_pose)

        self.color_image_sub = message_filters.Subscriber('/camera/color/image_raw', Image)
        self.camera_info_sub = message_filters.Subscriber('/camera/color/camera_info', CameraInfo)
        self.depth_image_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw', Image)
        ts = message_filters.TimeSynchronizer([self.color_image_sub, self.camera_info_sub, self.depth_image_sub], 1)
        ts.registerCallback(self.rgbdCallback)

        self.voxel_publisher =  rospy.Publisher("/head_perception/voxels/marker_array", MarkerArray, queue_size=10)

        self.tfBuffer = tf2_ros.Buffer()  # Using default cache time of 10 secs
        self.listener = tf2_ros.TransformListener(self.tfBuffer)
        self.broadcaster = tf2_ros.TransformBroadcaster()
        time.sleep(1.0)



    def rgbdCallback(self, rgb_image_msg, camera_info_msg, depth_image_msg):

        try:
            # Convert your ROS Image message to OpenCV2
            rgb_image = self.bridge.imgmsg_to_cv2(rgb_image_msg, "bgr8")
            depth_image = self.bridge.imgmsg_to_cv2(depth_image_msg, "32FC1")
        except CvBridgeError as e:
            print(e)

        arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
        arucoParams = cv2.aruco.DetectorParameters_create()
        (corners, ids, rejected) = cv2.aruco.detectMarkers(rgb_image, arucoDict, parameters=arucoParams)

        if len(corners) > 0:
            ids = ids.flatten()
            for (markerCorner, markerID) in zip(corners, ids):
                corners = markerCorner.reshape((4, 2))
        else:
            return None  

        landmarks = corners
        landmarks_model = np.array([[-0.04,0.04,0],[0.04,0.04,0],[0.04,-0.04,0],[-0.04,-0.04,0]])

        # convert  2d landmarks to 3d world points 
        valid_landmarks_model = []
        valid_landmarks_world = []
        for i in range(landmarks.shape[0]):
            validity, point = self.pixel2World(camera_info_msg, landmarks[i,0].astype(int), landmarks[i,1].astype(int), depth_image)
            if validity:
                valid_landmarks_model.append(landmarks_model[i])
                valid_landmarks_world.append(point)

        if len(valid_landmarks_world) < 4:
            print("Not enough landmarks to fit model.")
            return

        valid_landmarks_model = np.array(valid_landmarks_model)
        valid_landmarks_world = np.array(valid_landmarks_world)

        scale_fixed = 1.0
        s, ret_R, ret_t = self.kabschUmeyama(valid_landmarks_world, valid_landmarks_model, scale_fixed)

        # print("landmarks_selected_model[:,:,np.newaxis].shape: ",landmarks_model[:,:,np.newaxis].shape)
        # print("ret_R.shape: ",ret_R.shape)
        landmarks_model_camera_frame = ret_t.reshape(3,1) + s * (ret_R @ landmarks_model[:,:,np.newaxis])
        landmarks_model_camera_frame = np.squeeze(landmarks_model_camera_frame)



        # get things into world frame

        tag_pos = np.append(np.mean(valid_landmarks_world, axis=0), 1) # pad with 1 for homogeneous coordinate

        transform = self.get_frame_to_frame_transform(camera_info_msg)

        if transform is not None:   
            base_to_camera = self.make_homogeneous_transform(transform)

            # cam to tag homogeneous transform
            camera_to_tag = np.zeros((4, 4))
            camera_to_tag[:3, :3] = ret_R
            camera_to_tag[:3, 3] = np.array([ tag_pos[0], tag_pos[1], tag_pos[2] ]).reshape(1, 3)
            camera_to_tag[3, 3] = 1 

            # base to tag homogeneous transform and update tf
            base_to_tag = np.dot(base_to_camera, camera_to_tag)
            self.updateTF("base_link", "AR_tag", base_to_tag)

            base_to_tool = self.get_frame_to_frame_transform(camera_info_msg, "base_link", "tool_frame")
            self.follow_AR_tag(self.make_homogeneous_transform(base_to_tool), base_to_tag)

        self.visualizeVoxels(landmarks_model_camera_frame)

    def visualizeVoxels(self, voxels):

        # print(voxels)

        markerArray = MarkerArray()

        marker = Marker()
        marker.header.seq = 0
        marker.header.stamp = rospy.Time.now()
        marker.header.frame_id = "camera_color_optical_frame"
        marker.ns = "visualize_voxels"
        marker.id =  1
        marker.type = 6; # CUBE LIST
        marker.action = 0; # ADD
        marker.lifetime = rospy.Duration()
        marker.scale.x = 0.01
        marker.scale.y = 0.01
        marker.scale.z = 0.01
        marker.color.r = 1
        marker.color.g = 1
        marker.color.b = 0
        marker.color.a = 1

        for i in range(voxels.shape[0]):

            point = Point()
            point.x = voxels[i,0]
            point.y = voxels[i,1]
            point.z = voxels[i,2]

            marker.points.append(point)


        # average of the voxels is the center of the AR tag
        center = np.mean(voxels, axis=0)
        point = Point()
        point.x = center[0]
        point.y = center[1]
        point.z = center[2]
        marker.points.append(point)
        # 
        markerArray.markers.append(marker)
        self.voxel_publisher.publish(markerArray)

        return center

    def pixel2World(self, camera_info, image_x, image_y, depth_image):

        # print("(image_y,image_x): ",image_y,image_x)
        # print("depth image: ", depth_image.shape[0], depth_image.shape[1])

        if image_y >= depth_image.shape[0] or image_x >= depth_image.shape[1]:
            return False, None

        depth = depth_image[image_y, image_x]

        if math.isnan(depth) or depth < 0.05 or depth > 1.0:

            depth = []
            for i in range(-2,2):
                for j in range(-2,2):
                    if image_y+i >= depth_image.shape[0] or image_x+j >= depth_image.shape[1]:
                        return False, None
                    pixel_depth = depth_image[image_y+i, image_x+j]
                    if not (math.isnan(pixel_depth) or pixel_depth < 50 or pixel_depth > 1000):
                        depth += [pixel_depth]

            if len(depth) == 0:
                return False, None

            depth = np.mean(np.array(depth))

        depth = depth/1000.0 # Convert from mm to m

        fx = camera_info.K[0]
        fy = camera_info.K[4]
        cx = camera_info.K[2]
        cy = camera_info.K[5]  

        # Convert to world space
        world_x = (depth / fx) * (image_x - cx)
        world_y = (depth / fy) * (image_y - cy)
        world_z = depth

        return True, (world_x, world_y, world_z)

    def kabschUmeyama(self, A, B, scale):

        assert A.shape == B.shape

        # print("A Shape:", A.shape)
        # print("B Shape:", B.shape)

        # Calculate scaled B
        scaled_B = B*scale

        # Calculate translation using centroids
        A_centered = A - np.mean(A, axis=0)
        B_centered = scaled_B - np.mean(scaled_B, axis=0)

        # Calculate rotation using scipy

        R, rmsd = Rotation.align_vectors(A_centered, B_centered)

        # print("R: ",R.as_matrix().shape)
        # print("Scaled B: ",np.mean(scaled_B, axis=0).shape)

        t = np.mean(A, axis=0) - R.as_matrix()@np.mean(scaled_B, axis=0)

        return scale, R.as_matrix(), t
    
    def get_frame_to_frame_transform(self, camera_info_data, frame_A = "base_link", target_frame = "camera_color_optical_frame"):
        stamp = camera_info_data.header.stamp
        try:
            transform = self.tfBuffer.lookup_transform(
                frame_A,
                target_frame,
                rospy.Time(secs=stamp.secs, nsecs=stamp.nsecs),
            )
            return transform
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ):
            print("Exexption finding transform between base_link and", target_frame)
            return None

    def make_homogeneous_transform(self, transform):
        A_to_B = np.zeros((4, 4))
        A_to_B[:3, :3] = Rotation.from_quat(
            [
                transform.transform.rotation.x,
                transform.transform.rotation.y,
                transform.transform.rotation.z,
                transform.transform.rotation.w,
            ]
        ).as_matrix()
        A_to_B[:3, 3] = np.array(
            [
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z,
            ]
        ).reshape(1, 3)
        A_to_B[3, 3] = 1

        return A_to_B

    def updateTF(self, source_frame, target_frame, pose):

        t = TransformStamped()

        t.header.stamp = rospy.Time.now()
        t.header.frame_id = source_frame
        t.child_frame_id = target_frame

        t.transform.translation.x = pose[0][3]
        t.transform.translation.y = pose[1][3]
        t.transform.translation.z = pose[2][3]

        R = Rotation.from_matrix(pose[:3, :3]).as_quat()
        t.transform.rotation.x = R[0]
        t.transform.rotation.y = R[1]
        t.transform.rotation.z = R[2]
        t.transform.rotation.w = R[3]

        self.broadcaster.sendTransform(t)


    def follow_AR_tag(self, base_to_tool, base_to_tag):

        # print(base_to_tool)

        static_transform = np.zeros((4, 4))



        static_transform[:3, :3] = np.array([  [0,  1,  0],[0,  0, -1],[-1,  0,  0] ])
        static_transform[:3, 3] = np.array([.02, 0, .4]).reshape(1, 3)
        static_transform[3, 3] = 1 

        goal_frame = np.dot(base_to_tag, static_transform)
        # self.updateTF("base_link", "goal_frame", goal_frame)

        target_position = (goal_frame[0, 3], goal_frame[1, 3], goal_frame[2, 3]) # list slice instead? does this accept non-tuple?
        target_orientation = Rotation.from_matrix(goal_frame[:3, :3]).as_quat()
        target_orientation = (target_orientation[0], target_orientation[1], target_orientation[2], target_orientation[3])

        self.goal_pose_queue.append((target_position, target_orientation))

        running_average_position = np.mean([pose[0] for pose in self.goal_pose_queue], axis=0)
        running_average_orientation = np.mean([pose[1] for pose in self.goal_pose_queue], axis=0)
        running_average_rotation = Rotation.from_quat(running_average_orientation).as_matrix()
        running_average_frame = np.zeros((4, 4))
        running_average_frame[:3, :3] = running_average_rotation
        running_average_frame[:3, 3] = running_average_position
        self.updateTF("base_link", "goal_frame", running_average_frame)

        # Move in the direction of the goal, but not too quickly.
        current_position = np.array(self.current_pose[0])
        current_orientation = np.array(self.current_pose[1])
        position_delta = running_average_position - current_position
        orientation_delta = running_average_orientation - current_orientation
        
        max_position_delta = 0.01
        magnitude = np.linalg.norm(position_delta)
        if magnitude > max_position_delta:
            scale = max_position_delta / magnitude
            position_delta = position_delta * scale
            orientation_delta = orientation_delta * scale

        move_position = current_position + position_delta
        move_orientation = current_orientation + orientation_delta 
        
        self.send_command(move_position, move_orientation)

        # print(target_position, target_orientation)

        # self.send_command(target_position, target_orientation)

        # tool_to_tag = np.dot(np.linalg.inv(base_to_tool),base_to_tag)

        # print(tool_to_tag[0,3], tool_to_tag[1,3], tool_to_tag[2,3])


    def send_command(self, target_position, target_orientation):
        cmd = CartesianCommand(target_position, target_orientation)
        self.robot_interface.execute_command(cmd)
        # robot_interface = ArmInterfaceClient()
        # robot_interface.execute_command(cmd)
        
    def read_pose(self, arm_pose_msg):
        self.current_pose = (
            (arm_pose_msg.position.x, arm_pose_msg.position.y, arm_pose_msg.position.z),
            (arm_pose_msg.orientation.x, arm_pose_msg.orientation.y, arm_pose_msg.orientation.z, arm_pose_msg.orientation.w)
        )

        """
        
(0.5736776157933938, -0.09580082382673927, 0.0766157263532676) [0.5627823  0.43596175 0.43279593 0.55308329]"""

        """
        position: 
            x: 0.5629775524139404
            y: -0.09015465527772903
            z: 0.07269313931465149
        orientation: 
            x: 0.5700255012489195
            y: 0.46046393122423585
            z: 0.4095031450955806
            w: 0.5434621147092666
        """

if __name__ == '__main__':



    # target_position = (0.61, -0.12, 0.6)
    # target_orientation = (0.56, 0.37, 0.49, 0.55)
    # cmd = CartesianCommand(target_position, target_orientation)
    # robot_interface = ArmInterfaceClient()
    # robot_interface.execute_command(cmd)

    aruco_perception = ArUcoPerception()
    rospy.spin()
