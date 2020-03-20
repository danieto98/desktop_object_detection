#!/usr/bin/env python
import rospy
from desktop_object_detection.msg import Results
from openpyxl import Workbook
import signal, os
import sys

# Excel logger class
class ExcelLogger:
	def __init__(self, filename):
		# Set Excel filename
		self.filename = filename

		# Open Excel workbook and initialize sheet
		self.wb = Workbook()
		self.ws = self.wb.active
		self.ws.title = "Data Dump"

		# Initialize object index
		self.nr = 1

		# Output first line of Excel sheet (table labels)
		self.ws.append(["Object No.", "Object Class", "Static/Dynamic", "Distance (from Camera)"])

	# Results callback function
	def callback(self, data):
		# Output values to Excel file
		object_str = "Object %d" % (self.nr)
		distance_str = "%d cm" % data.distance
		self.ws.append([object_str, data.className, data.type, distance_str])

		# Add 1 to object index
		self.nr = self.nr + 1

	# SIGINT handler to save file before exiting
	def signal_handler(self, signum, frame):
		if signum == signal.SIGINT:
			self.save()
			exit(0)

	# Save file
	def save(self):
		self.wb.save(filename = self.filename)


# Main function
if __name__ == '__main__':
	# Initialize ROS node
	rospy.init_node("excel_logger", anonymous=True, disable_signals=True)

	# Get filename from argv
	filename = ""
	if len(sys.argv) == 4:
		filename = sys.argv[1]
	else:
		rospy.logerr("Usage: excel_logger.py filepath")
		exit(1)

	# Initialize excel logger
	logger = ExcelLogger(filename)
	rospy.on_shutdown(logger.save)

	# Register SIGINT signal handler
	signal.signal(signal.SIGINT, logger.signal_handler)
	signal.siginterrupt(signal.SIGINT, False)

	# Subscribe to results message
	rospy.Subscriber("results", Results, logger.callback)
	rospy.spin()