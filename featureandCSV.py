#To create the training data to a csv file
#Input feature of surf- 16*64
#Input feature of HOG 3780
#total input feature is 4804

import cv2 
import pandas as pd 
import numpy as np 
import os


from sklearn.cluster import KMeans

n_clusters = 16

test_bow = 0
data = None
hog = cv2.HOGDescriptor()

low = int(raw_input('Enter lower index : '))
high = int(raw_input('Enter higher index : '))
print(low)
print(high)
file = open("data.txt","w+")

'''
im = cv2.imread("Images_HOG/Obstacles/0.png")
img = cv2.resize(im,(64,128))

h = hog.compute(img)
t = np.array(h)
print(t.shape)
prediction = 1

file.write(str(prediction) + " ")
l = 1
for a in range(len(t)):
	s = str(t[a])
	s = s.replace('[','')
	s = s.replace(']','')
	file.write(str(l) + ":" + s + " ")
	l+=1
file.write("\n")

t = np.insert(t,len(t),prediction)
data = np.array(t)
'''

for i in range(low, high+1):
	print( "The current folder :" + str(i))
	xyz = 'traffic_sign_dataset/training/Images/' + str(i) + '/' 
	k = 0
	for j in os.listdir(xyz):
		# if (test_bow > 1):
		# 	break

		# test_bow+=1

		print("The current image :" + str(k))
		im = cv2.imread(xyz + str(k) + ".ppm")
		try : 
			k += 1
			# cv2.imshow('a',im)
			#cv2.waitKey(1)
			img = cv2.resize(im,(100,100))
			img_hog = cv2.resize(img,(64,128))

			surf = cv2.xfeatures2d.SURF_create(hessianThreshold=0)
			kp,des = surf.detectAndCompute(img,None)
			# img4 = cv2.drawKeypoints(img,kp,None,(255,0,0),0)

			kmeans = None
			surf_centers = None

			if(des is not None and des.shape[0] > n_clusters):
				print('SURF : ' + str(des.shape))
				print(des.shape)

				kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(des)
				surf_centers = kmeans.cluster_centers_

				print(kmeans.labels_.shape)
				# print(kmeans.cluster_centers_.shape)
				# print(time.time() - t)

			else:
				continue

			h = hog.compute(img_hog)
			print('h')
			t = np.array(h)
			print("The HOG array shape is ",t.shape)
			print(surf_centers.shape)
			print('bow')
			t = t.reshape([1,-1])
			surf_centers = surf_centers.reshape([1,-1])

			t = np.hstack((t,surf_centers))
			print("The shape after stacking is",t.shape)
			print('bw2')
			# surf=cv2.xFeatures2d.surf_create()

			prediction = i + 1 
			# kp_surf,des_surf=surf.de
			l = 1
			# file.write(str(prediction) + " ")
			# for a in range(len(t)):
			# 	s = str(t[a])
			# 	s = s.replace('[','')
			# 	s = s.replace(']','')
			# 	file.write(str(l) + ":" + s + " ")
			# 	l+=1
			# file.write("\n")

			print("The stacked array is ",t)

			q = np.array([prediction])
			q = q.reshape([1,1])
			t = np.hstack((t,q))
			
			print(t)
			print('yohoo')
			print(t.shape)

			if i == low and k == 1 :
				print('fst')
				data = np.array(t)
			else :
				print('in')
				data = np.vstack((data,t))
				print('in')
			print('data')
		except : 
			continue
'''
for i in range(0,294):
	im = cv2.imread("Images_HOG/Non-obstacles/" + str(i) + ".png")
	cv2.imshow('a',im)
	#cv2.waitKey(1)
	img = cv2.resize(im,(64,128))

	h = hog.compute(img)
	t = np.array(h)
	prediction = 0

	file.write(str(prediction) + " ")

	l = 1
	for a in range(len(t)):
		s = str(t[a])
		s = s.replace('[','')
		s = s.replace(']','')
		file.write(str(l) + ":" + s + " ")
		l+=1
	file.write("\n")

	t = np.insert(t,len(t),prediction)
	data = np.vstack((data,t))
'''

data_csv = pd.DataFrame(data)
print(data_csv)
data_csv.to_csv("data" + str(low) +".csv")

