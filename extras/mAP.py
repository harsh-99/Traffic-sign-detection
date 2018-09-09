import csv
import cv2 as cv
import numpy as np

troi_cascade=cv.CascadeClassifier("cascade_triangular.xml")
croi_cascade=cv.CascadeClassifier("cascade_circular_lbp_with_background.xml")

def check(a, b, l):
	if l[1] == 0:
		return 0
	else:
		for i in range(l[1]):
			if a > l[3+i][0]:
				if a < l[3+i][2]:
					if b > l[3+i][1]:
						if b < l[3+i][3]:
							return 1
	return 0

def makestring(i, arr):
	s = []
	if i < 10:
		s.append("0000" + str(i) + ".ppm")
	elif i < 100:
		s.append("000" + str(i) + ".ppm")
	else:
		s.append("00" + str(i) + ".ppm")
	s.append(0)
	s.append([])
	for j in range(len(arr)):
		if arr[j][0] == s[0]:
			s[1] = s[1] + 1
			s[2].append(j)
	for k in range(s[1]):
		s.append([arr[s[2][k]][1], arr[s[2][k]][2], arr[s[2][k]][3], arr[s[2][k]][4], arr[s[2][k]][5]])
	return s

def convert(l):
	a = []
	a.append(l[0])
	for i in range(len(l) - 1):
		a.append(int(l[i+1]))
	return a

array = []
with open('gt.txt') as csvfile:
	spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
	mylist = list(spamreader)
	for listindex in range(len(mylist)):
		mylist[listindex] = mylist[listindex][0].split(";")
		mylist[listindex] = convert(mylist[listindex])
	array = mylist

betterarray = []

for i in range(900):
	betterarray.append(makestring(i, array))

tp = 0
fp = 0
total_signs = 0

for i in range(900):
	total_signs += betterarray[i][1]
	img = cv.imread(betterarray[i][0], 1)
	imgr=cv.resize(img,(3*img.shape[1]/4,3*img.shape[0]/4))
	gray=cv.cvtColor(imgr,cv.COLOR_BGR2GRAY)
	troi=troi_cascade.detectMultiScale(gray,1.25,5)
	croi=croi_cascade.detectMultiScale(gray,1.25,2)
	for (x,y,w,h) in troi:
		c = check(4*(x+w/2)/3,4*(y+h/2)/3,betterarray[i])
		if c == 1:
			tp += 1
		else:
			fp += 1
#		print(4*(x)/3)
#		print(4*(x+w)/3)
#		print(4*(y)/3)
#		print(4*(y+h)/3)
		cv.rectangle(imgr,(x,y),(x+w,y+h),(0,0,255),2)	
	for (x,y,w,h) in croi:
		c = check(4*(x+w/2)/3,4*(y+h/2)/3,betterarray[i])
		if c == 1:
			tp += 1
		else:
			fp += 1
#		print(4*(x)/3)
#		print(4*(x+w)/3)
#		print(4*(y)/3)
#		print(4*(y+h)/3)	
		cv.rectangle(imgr,(x,y),(x+w,y+h),(0,0,255),2)
#	cv.namedWindow("img")
#	cv.imshow("img", imgr)
#	cv.waitKey(0)	
fn = total_signs - tp
print("True positives = " + str(tp))
print("False positives = " + str(fp))
print("False negatives = " + str(fn))
print("mAP = " +str(tp*1./(tp+fp)))
