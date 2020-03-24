# Original RGB image which was segmented
sensor_msgs/Image original_image

# n*4 matrix with coordinates of rectangles enclosing contours
# [[pt1.x, pt1.y, pt2.x, pt2.y],
# [......,......,......,......],
# [......,......,......,......]]
uint32[] bounding_rectangle_coords

# distance (in cm) to closest point in contour
uint32[] distances