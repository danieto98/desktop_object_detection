# Original RGB image which was segmented
sensor_msgs/Image original_image

# Segmented contour images
# (contour is white, bg is black)
# Same size as original_image
sensor_msgs/Image[] segmentation_contours

# n*4 matrix with coordinates of rectangles enclosing contours
# [[pt1.x, pt1.y, pt2.x, pt2.y],
# [......,......,......,......],
# [......,......,......,......]]
uint32[] bounding_rectangle_coords

# distance (in cm) to closest point in contour
uint32[] distances