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


class ArUcoPerception:
    def __init__(self):
        rospy.init_node("ArUcoPerception")

        self.bridge = CvBridge()

        self.color_image_sub = message_filters.Subscriber(
            "/camera/color/image_raw", Image
        )
        self.camera_info_sub = message_filters.Subscriber(
            "/camera/color/camera_info", CameraInfo
        )
        self.depth_image_sub = message_filters.Subscriber(
            "/camera/aligned_depth_to_color/image_raw", Image
        )
        ts = message_filters.TimeSynchronizer(
            [self.color_image_sub, self.camera_info_sub, self.depth_image_sub], 1
        )
        ts.registerCallback(self.rgbdCallback)

        self.voxel_publisher = rospy.Publisher(
            "/head_perception/voxels/marker_array", MarkerArray, queue_size=10
        )
        self.center_publisher = rospy.Publisher("/aruco_center", Point, queue_size=10)

    def rgbdCallback(self, rgb_image_msg, camera_info_msg, depth_image_msg):

        try:
            # Convert your ROS Image message to OpenCV2
            rgb_image = self.bridge.imgmsg_to_cv2(rgb_image_msg, "bgr8")
            depth_image = self.bridge.imgmsg_to_cv2(depth_image_msg, "32FC1")
        except CvBridgeError as e:
            print(e)

        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        corners, ids, rejected = detector.detectMarkers(rgb_image)

        # (corners, ids, rejected) = cv2.aruco.detectMarkers(rgb_image, arucoDict, parameters=arucoParams)

        if len(corners) > 0:
            # print("Detected arUco")

            ids = ids.flatten()

            for markerCorner, markerID in zip(corners, ids):

                corners = markerCorner.reshape((4, 2))
                (topLeft, topRight, bottomRight, bottomLeft) = corners.astype(int)

                centerX = (topLeft[0] + bottomRight[0]) // 2
                centerY = (topLeft[1] + bottomRight[1]) // 2

                forwardX = (topLeft[0] + topRight[0]) // 2
                forwardY = (topLeft[1] + topRight[1]) // 2

                ## Visualization ####

                viz_image = rgb_image.copy()

                # draw the bounding box of the ArUCo detection
                cv2.line(viz_image, topLeft, topRight, (0, 255, 0), 2)
                cv2.line(viz_image, topRight, bottomRight, (0, 255, 0), 2)
                cv2.line(viz_image, bottomRight, bottomLeft, (0, 255, 0), 2)
                cv2.line(viz_image, bottomLeft, topLeft, (0, 255, 0), 2)

                # visualize center of ArUco marker
                cv2.circle(viz_image, (centerX, centerY), 4, (0, 0, 255), -1)

                # visualize top of ArUco marker
                cv2.circle(viz_image, (forwardX, forwardY), 4, (0, 0, 255), -1)

                # draw the ArUco marker ID on the image
                cv2.putText(
                    viz_image,
                    str(markerID),
                    (topLeft[0], topLeft[1] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

                print("[INFO] ArUco marker ID: {}".format(markerID))
        else:
            return None

        landmarks = corners
        landmarks_model = np.array(
            [[-0.04, 0.04, 0], [0.04, 0.04, 0], [0.04, -0.04, 0], [-0.04, -0.04, 0]]
        )

        # convert  2d landmarks to 3d world points
        valid_landmarks_model = []
        valid_landmarks_world = []
        for i in range(landmarks.shape[0]):
            validity, point = self.pixel2World(
                camera_info_msg,
                landmarks[i, 0].astype(int),
                landmarks[i, 1].astype(int),
                depth_image,
            )
            if validity:
                valid_landmarks_model.append(landmarks_model[i])
                valid_landmarks_world.append(point)

        if len(valid_landmarks_world) < 4:
            print("Not enough landmarks to fit model.")
            return

        valid_landmarks_model = np.array(valid_landmarks_model)
        valid_landmarks_world = np.array(valid_landmarks_world)

        scale_fixed = 1.0
        s, ret_R, ret_t = self.kabschUmeyama(
            valid_landmarks_world, valid_landmarks_model, scale_fixed
        )

        # print("landmarks_selected_model[:,:,np.newaxis].shape: ",landmarks_model[:,:,np.newaxis].shape)
        # print("ret_R.shape: ",ret_R.shape)
        landmarks_model_camera_frame = ret_t.reshape(3, 1) + s * (
            ret_R @ landmarks_model[:, :, np.newaxis]
        )
        landmarks_model_camera_frame = np.squeeze(landmarks_model_camera_frame)

        center_point = np.mean(landmarks_model_camera_frame, axis=0)
        self.center_publisher.publish(
            Point(center_point[0], center_point[1], center_point[2])
        )

        self.visualizeVoxels(landmarks_model_camera_frame)

    def visualizeVoxels(self, voxels):

        # print(voxels)

        markerArray = MarkerArray()

        marker = Marker()
        marker.header.seq = 0
        marker.header.stamp = rospy.Time.now()
        marker.header.frame_id = "camera_color_optical_frame"
        marker.ns = "visualize_voxels"
        marker.id = 1
        marker.type = 6
        # CUBE LIST
        marker.action = 0
        # ADD
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
            point.x = voxels[i, 0]
            point.y = voxels[i, 1]
            point.z = voxels[i, 2]

            marker.points.append(point)

        markerArray.markers.append(marker)

        self.voxel_publisher.publish(markerArray)

    def pixel2World(self, camera_info, image_x, image_y, depth_image):

        # print("(image_y,image_x): ",image_y,image_x)
        # print("depth image: ", depth_image.shape[0], depth_image.shape[1])

        if image_y >= depth_image.shape[0] or image_x >= depth_image.shape[1]:
            return False, None

        depth = depth_image[image_y, image_x]

        if math.isnan(depth) or depth < 0.05 or depth > 1.0:

            depth = []
            for i in range(-2, 2):
                for j in range(-2, 2):
                    if (
                        image_y + i >= depth_image.shape[0]
                        or image_x + j >= depth_image.shape[1]
                    ):
                        return False, None
                    pixel_depth = depth_image[image_y + i, image_x + j]
                    if not (
                        math.isnan(pixel_depth)
                        or pixel_depth < 50
                        or pixel_depth > 1000
                    ):
                        depth += [pixel_depth]

            if len(depth) == 0:
                return False, None

            depth = np.mean(np.array(depth))

        depth = depth / 1000.0  # Convert from mm to m

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
        scaled_B = B * scale

        # Calculate translation using centroids
        A_centered = A - np.mean(A, axis=0)
        B_centered = scaled_B - np.mean(scaled_B, axis=0)

        # Calculate rotation using scipy

        R, rmsd = Rotation.align_vectors(A_centered, B_centered)

        # print("R: ",R.as_matrix().shape)
        # print("Scaled B: ",np.mean(scaled_B, axis=0).shape)

        t = np.mean(A, axis=0) - R.as_matrix() @ np.mean(scaled_B, axis=0)

        return scale, R.as_matrix(), t


if __name__ == "__main__":

    aruco_perception = ArUcoPerception()
    rospy.spin()
