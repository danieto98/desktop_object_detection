#!/usr/bin/env python
import rospy
from cv_bridge import CvBridge
from desktop_object_detection.msg import SegAndDist
from desktop_object_detection.msg import Results
import tensorflow as tf
import cv2
import numpy as np
import sys
from tensorflow.keras.preprocessing import image

# Recognizer class
class Recognizer:
	def __init__(self, model_path, publisher):
		self.publisher = publisher
		self.model = tf.keras.models.load_model(model_path)
		self.model.add(tf.keras.layers.Softmax())
		self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
		self.bridge = CvBridge()
		
		# List of class names
		self.CLASSES = ["Scientific Calculator", "Apple", "Computer Mouse"]
		
		# Previously recorded results
		self.prevRecs = [[],[],[]]

		# Get ROS parameter for prediction percentage
		self.predict_percent = rospy.get_param("/predict_percent", 0.4)
	def callback(self, data):
		# Get original RGB image from data
		original_image = self.bridge.imgmsg_to_cv2(data.original_image, desired_encoding="passthrough")

		# Get the number of recognized contours
		contour_nr = len(data.distances)

		# Iterate over all  recognized contours
		for i in range(contour_nr):
			# Crop original image to bounding rectangle
			crop = original_image[data.bounding_rectangle_coords[i*4+1]:data.bounding_rectangle_coords[i*4+3], data.bounding_rectangle_coords[i*4]:data.bounding_rectangle_coords[i*4+2]]
			if crop.shape[0] == crop.shape[1] and crop.shape[0] > 0 and crop.shape[1] > 0:
				# Scale rectangle to a 220x220 image
				scaled = cv2.resize(crop, dsize=(220, 220), interpolation = cv2.INTER_CUBIC)

				# Initialize output message and set distance to object
				output = Results()
				output.distance = data.distances[i]

				# Convert scaled image to a format recognizable by tensorflow
				img = scaled[...,::-1].astype(np.float32) / 255.0

				# Get image array
				x = image.img_to_array(img)
				x = np.expand_dims(x, axis=0)

				# Get tensorflow result class and put into output message if confidence greater than 80%
				predictions = self.model.predict(x)
				idx = np.argmax(predictions[0])
				rospy.loginfo("Predicted class: %s. Percentage: %f", self.CLASSES[idx], predictions[0][idx])
				if predictions[0][idx] > self.predict_percent:
					output.className = self.CLASSES[idx]
					self.prevRecs[idx] = self.prevRecs[idx] + [output.distance]

					# If this object had been recognized before in a position differing by at least epsilon, label it as dynamic
					epsilon = 5
					if len(self.prevRecs[idx]) > 1:
						if (self.prevRecs[idx][-2] + epsilon < self.prevRecs[idx][-1]) or (self.prevRecs[idx][-2] - epsilon > self.prevRecs[idx][-1]):
							output.type = "Dynamic"
						else:
							output.type = "Static"
					else:
						output.type = "Static"

					# Publish output message
					self.publisher.publish(output)

# Main function
if __name__ == '__main__':
	# Initialize ROS node
	rospy.init_node("cnn", anonymous=True, disable_signals=True)

	# Get model path from argv
	model_path = ""
	if len(sys.argv) == 4:
		model_path = sys.argv[1]
	else:
		rospy.logerr("Usage: excel_logger.py model_path")
		exit(1)

	# Advertise publisher
	pub = rospy.Publisher("results", Results, queue_size = 10)

	# Initialize Recognizer
	recog = Recognizer(model_path, pub)

	# Attach Recognizer callback to segmentation_and_distance
	rospy.Subscriber("segmentation_and_distance", SegAndDist, recog.callback)
	rospy.spin()