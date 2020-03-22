import cv2
import numpy as np
import os
import sys
import math

directory = ""
result_dir = ""

# Get directory paths
if len(sys.argv) == 3:
    directory = sys.argv[1]
    result_dir = sys.argv[2]
else:
    exit("Usage: downscale_images.py input_dataset_path output_dataset_path")

# Iterate over all child directory names
child_dirs = next(os.walk(directory))[1]
for child_dir in child_dirs:
    # Iterate over all images in the given directory
    for filename in os.listdir(directory + "/" + child_dir):
        # If file is a jpg or png image
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # Get image from filename
            original = cv2.imread(directory + "/" + child_dir + "/" + filename, cv2.IMREAD_COLOR)

            # Get gray image
            gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

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
            ret, sure_fg = cv2.threshold(dist_transform,0.4*dist_transform.max(),255,0)

            # Find unknown region
            sure_fg = np.uint8(sure_fg)
            unknown = cv2.subtract(sure_bg,sure_fg)

            # Marker labeling
            ret, markers = cv2.connectedComponents(sure_fg)
            markers = markers+1
            markers[unknown==255] = 0

            # Watershed transform for segmentation
            markers = cv2.watershed(original,markers)

            # Find bounding rectangles
            markers1 = markers.astype(np.uint8)
            ret, m2 = cv2.threshold(markers1, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
            _, contours, hierarchy = cv2.findContours(m2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours_poly = [None]*len(contours)
            boundRect = [None]*len(contours)
            for i, c in enumerate(contours):
                contours_poly[i] = cv2.approxPolyDP(c, 3, True)
                boundRect[i] = cv2.boundingRect(contours_poly[i])

            # Expand rectangles to squares, enlarge them and pick the one closest to the center
            # that is at least 350x350 pixels
            pix_val = 75
            first = True
            real_center = [int(original.shape[1]/2.0), int(original.shape[0]/2.0)]
            center_dist = 0
            corn_1 = [0, 0]
            corn_2 = [0, 0]
            for i in range(len(contours)):
                corner_1 = [int(boundRect[i][0]), int(boundRect[i][1])]
                corner_3 = [int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])]
                if corner_1[0] != 0 or corner_1[1] != 0 or corner_3[0] != original.shape[1] or corner_3[1] != original.shape[0]:
                    corner_2 = [int(boundRect[i][0]), int(boundRect[i][1]+boundRect[i][3])]
                    corner_4 = [int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1])]
                    side_1_size = math.sqrt(float(corner_1[1] - corner_2[1])*float(corner_1[1] - corner_2[1]) + float(corner_1[0] - corner_2[0])*float(corner_1[0] - corner_2[0]))
                    side_2_size = math.sqrt(float(corner_3[1] - corner_2[1])*float(corner_3[1] - corner_2[1]) + float(corner_3[0] - corner_2[0])*float(corner_3[0] - corner_2[0]))
                    result_corner_2 = [0, 0]
                    corner_1 = [corner_1[0] - pix_val, corner_1[1] - pix_val]
                    if side_1_size > side_2_size:
                        result_corner_2 = [int(boundRect[i][0]+boundRect[i][3]+pix_val),int(boundRect[i][1]+boundRect[i][3]+pix_val)]
                    else:
                        result_corner_2 = [int(boundRect[i][0]+boundRect[i][2]+pix_val), int(boundRect[i][1]+boundRect[i][2]+pix_val)]
                    if result_corner_2[0] - corner_1[0] > 350 :
                        center = [float(corner_1[0]+corner_2[0])/2.0, float(corner_1[1]+corner_2[1])/2.0]
                        distance_to_center = math.sqrt(float(center[0] - real_center[0])*float(center[0] - real_center[0]) + float(center[1] - real_center[1])*float(center[1] - real_center[1]))
                        if first:
                            center_dist = distance_to_center
                            corn_1 = corner_1
                            corn_2 = result_corner_2
                            first = False
                        elif distance_to_center < center_dist:
                            center_dist = distance_to_center
                            corn_1 = corner_1
                            corn_2 = result_corner_2

            # Crop image to selected square, scale down to 220x220 and save to second directory
            crop = original[corn_1[1]:corn_2[1], corn_1[0]:corn_2[0]]
            if crop.shape[0] == crop.shape[1] and crop.shape[0] > 0 and crop.shape[1] > 0:
                scaled = cv2.resize(crop, (220, 220), interpolation = cv2.INTER_AREA)
                cv2.imwrite(result_dir + "/" + child_dir + "/" + filename, scaled)