#!/usr/bin/env python
import rospy
from desktop_object_detection.msg import Results

from openpyxl import Workbook

class ExcelLogger:
	def __init__(self, filename):
		self.wb = Workbook()
		self.ws = self.wb.active
		self.ws.title = "Data Dump"
		self.nr = 1
		self.ws.append(["Object No.", "Object Class", "Static/Dynamic", "Distance (from Camera)"])
	def callback(self, data):
		str = "Object %d" % (self.nr)
		self.ws.append([str, data.className, data.type, data.distance])
		self.nr = self.nr + 1

if __name__ == '__main__':
	logger = ExcelLogger("test.xlsx")
	rospy.init_node("excel_logger", anonymous=True)
	rospy.Subscriber("results", Results, logger.callback)
	rospy.spin()