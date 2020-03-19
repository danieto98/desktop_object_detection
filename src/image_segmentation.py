#!/usr/bin/env python
import rospy
from rtabmap_ros.msg import RGBDImage
from cv_bridge import CvBridge
import numpy as np
import cv2
from matplotlib import pyplot as plt
from desktop_object_detection.msg import SegAndDist
from math import sqrt

# Image Segmenter class
class ImageSegmenter:
    # Constructor
    def __init__(self, publisher):
        self.bridge = CvBridge()
        self.publisher = publisher

    # rgbd_image callback
    def callback(self, data):
        # Get RGB image from data
        cv_rgb_image = self.bridge.imgmsg_to_cv2(data.rgb, desired_encoding="passthrough")
        cv_rgb_image.setflags(write=1)

        # Initialize output message
        output_msg = SegAndDist()
        output_msg.original_image = self.bridge.cv2_to_imgmsg(cv_rgb_image, encoding="passthrough")

        # Convert image to grayscale
        gray = cv2.cvtColor(cv_rgb_image,cv2.COLOR_BGR2GRAY)

        # Apply thresholding
        ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        
        # Morphological opening for noise removal
        kernel = np.ones((3,3),np.uint8)
        opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
        
        # Find sure background
        sure_bg = cv2.dilate(opening,kernel,iterations=3)

        # Apply distance transform to find foreground
        dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
        dist_transform = cv2.convertScaleAbs(dist_transform)
        ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
        
        # Find unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg,sure_fg)

        # Marker labeling
        ret, markers = cv2.connectedComponents(sure_fg)
        markers = markers+1
        markers[unknown==255] = 0

        # Watershed transform for segmentation
        markers = cv2.watershed(cv_rgb_image,markers)

        # Find bounding rectangles
        markers1 = markers.astype(np.uint8)
        ret, m2 = cv2.threshold(markers1, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
        _, contours, hierarchy = cv2.findContours(m2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_poly = [None]*len(contours)
        boundRect = [None]*len(contours)
        for i, c in enumerate(contours):
            contours_poly[i] = cv2.approxPolyDP(c, 3, True)
            boundRect[i] = cv2.boundingRect(contours_poly[i])

        # Put bounding rectangle pixel coordinates in output
        output_msg.bounding_rectangle_coords = []
        for i in range(len(contours)):
            output_msg.bounding_rectangle_coords = output_msg.bounding_rectangle_coords + [int(boundRect[i][0]), int(boundRect[i][1]), int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])]

        # Get depth image
        cv_depth_image = self.bridge.imgmsg_to_cv2(data.depth, desired_encoding="passthrough")
        cv_depth_image.astype("float32")

        # Get parameters from camera matrix
        camera_matrix = np.array(data.depthCameraInfo.K).reshape([3, 3])
        cx = camera_matrix[0, 2]
        cy = camera_matrix[1, 2]
        fx_inv = 1.0 / camera_matrix[0, 0]
        fy_inv = 1.0 / camera_matrix[1, 1]

        # Get distance of each point in matrix in cm
        dist = np.zeros((cv_depth_image.shape[0], cv_depth_image.shape[1]), np.uint8)
        for i in range(cv_depth_image.shape[0]):
            for j in range(cv_depth_image.shape[1]):
                z = cv_depth_image[i, j]
                x = z * ((i - cx) * fx_inv)
                y = z * ((j - cy) * fy_inv)
                dist[i, j] = sqrt(x*x + y*y + z*z)

        # For each contour, get its closest distance and contour image. Put them in output message
        output_msg.segmentation_contours = []
        output_msg.distances = []
        for c in range(len(contours)):
            contour_image = np.zeros((cv_rgb_image.shape[0], cv_rgb_image.shape[1]), np.uint8)
            cv2.drawContours(contour_image, contours, c, [255], thickness=cv2.FILLED)
            img_msg = self.bridge.cv2_to_imgmsg(contour_image, encoding="passthrough")
            output_msg.segmentation_contours = output_msg.segmentation_contours + [img_msg]
            first = True
            mindist = 0
            for i in range(cv_depth_image.shape[0]):
                for j in range(cv_depth_image.shape[1]):
                    if contour_image[i, j] == 255:
                        if first:
                            mindist = dist[i, j]
                            first = False
                        elif dist[i, j] < mindist:
                            mindist = dist[i, j]
            output_msg.distances = output_msg.distances + [int(mindist)]

        # Publish output message
        self.publisher.publish(output_msg)


# Main function
if __name__ == '__main__':
    # Initialize ROS node
    rospy.init_node("image_segmentation", anonymous=True)

    # Advertise publisher
    pub = rospy.Publisher("segmentation_and_distance", SegAndDist, queue_size = 10)

    # Initialize ImageSegmenter
    segmenter = ImageSegmenter(pub)

    # Attach ImageSegmenter callback to rgbd_image
    rospy.Subscriber("rgbd_image", RGBDImage, segmenter.callback)
    rospy.spin()