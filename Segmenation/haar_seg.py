import numpy as np
import cv2 as cv
import time 

danger_cascade=cv.CascadeClassifier("../xml_files/triangular_cascade.xml")
# danger_cascade_triangle = cv.CascadeClassifier("./xml_files/triangular_cascade_lbp.xml")
for i in range (0,17):
	img=cv.imread("img" + str(i) + ".ppm")
	img = cv.resize(img, ((img.shape[1]*3/4), (img.shape[0]*3/4)))
	gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
	print(gray.shape[0])
	alpha = time.time()
	# danger_traingle, aaa_traingle, bbb_traingle = danger_cascade_triangle.detectMultiScale3(gray, scaleFactor=1.3, minNeighbors=20, outputRejectLevels = True)
	danger, aaa, bbb=danger_cascade.detectMultiScale3(gray, scaleFactor=1.14, minNeighbors=18, outputRejectLevels = True)
	# danger=danger_cascade.detectMultiScale(gray, scaleFactor=1.26, minNeighbors=16)
	print(time.time()-alpha)
	# print(aaa)
	# print(bbb)
	# print(aaa_traingle)
	# print(bbb_traingle)
	i=0
	c=0
	d=0
	e=0
	# for (x,y,w,h) in danger:
	# 	if(bbb[i] > 2):
	# 		c = c+1
	# 		cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
	# 		# print(x,y,x+w,y+h)
	# 	i=i+1
	for (x,y,w,h) in danger:
		c = c+1
		cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
		# print(x,y,x+w,y+h)
		i=i+1
	# print()
	# for (x,y,w,h) in danger_traingle:
	# 	if(bbb_traingle[e] > 2):
	# 		d = d+1
	# 		cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
	# 		print(x,y,x+w,y+h)
	# 	e=e+1
	# print("No of circle segments detected-", c)
	# print("No of traingle segments detected-", e)
	# cv.imshow('gray',img)
	# cv.waitKey(5000)