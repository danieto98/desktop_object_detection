#!/usr/bin/env python
import rospy
from desktop_object_detection.msg import Results
import sys
from openpyxl import load_workbook
from random import seed
from random import randint

# Result message streamer class
class ResultStream:
	def __init__(self, publisher):
		# Retrieve publisher
		self.publisher = publisher

		# Set classes and types used
		self.classes = ["Apple", "Scientific Calculator", "Computer Mouse"]
		self.types = ["Static", "Dynamic"]

		# Seed random number generator
		seed(1)
	def iterate(self):
		# Randomly fill the results message
		res = Results()
		res.className = self.classes[randint(0,len(self.classes)-1)]
		res.type = self.types[randint(0,len(self.types)-1)]
		res.distance = randint(0, 150)

		# Publish results message
		self.publisher.publish(res)
		

# Main function
if __name__ == '__main__':
	# Initialize ROS node
	rospy.init_node("result_message_stream", anonymous=True)

	# Advertise publisher
	pub = rospy.Publisher("results", Results, queue_size = 10)

	# Initialize streamer class
	streamer = ResultStream(pub)

	# Iterate until node is shut down
	rate = rospy.Rate(10)
	while not rospy.is_shutdown():
		streamer.iterate()
		rate.sleep()