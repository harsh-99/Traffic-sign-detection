import numpy as np
import cv2 as cv

danger_cascade=cv.CascadeClassifier("../xml_files/cascade.xml")
for imgname in []:
	img=cv.imread(imgname)
	gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
	danger=danger_cascade.detectMultiScale(gray,1.3,50)
	for (x,y,w,h) in danger:
		cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
	cv.imshow('img',img)
	cv.waitKey(5000)