import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import tensorflow as tf 
from tensorflow.python.framework import ops
from sklearn.cluster import KMeans
import cv2
import time

time_it = time.time()

ops.reset_default_graph()

image_name = '12.ppm'
im = cv2.imread(image_name)
r = im
time_it = time.time()
###Normalize image###
normalised_red = np.zeros((im.shape[0],im.shape[1]), np.uint8)
# for i in range(0, im.shape[0]) :
# 	for j in range(0, im.shape[1]) :
# 		a = int(im[i,j,0]) + int(im[i,j,1]) + int(im[i,j,2])
# 		normalised_red[i,j] = 255*float(im[i,j,2])/(a + 1e-7)
im_cast = im.astype(int)
im_sum = np.sum(im_cast,axis=2,keepdims=True)
# print(im_sum)
normalised_red = (1.0*im/(1.0*im_sum + 1e-7))*255.0
normalised_red = normalised_red[:,:,2].astype(np.uint8)	
# print(normalised_red)
cv2.imshow('a',normalised_red)
cv2.waitKey(0)

norm_hist = cv2.calcHist([normalised_red], [0], None, [256], [0, 256])
norm_hist_rev = norm_hist[::- 1]
sum_pix = np.sum(norm_hist_rev)
cum_sum_pix = np.cumsum(norm_hist_rev)
allowed_pixs = sum_pix*0.002 ########### adjust the number of pixels you want ##########
thresh = 255 - np.argmax(norm_hist_rev > allowed_pixs)
g,thresholded = cv2.threshold(normalised_red, thresh, 255, cv2.THRESH_BINARY)
kernel = np.ones((3,3),np.uint8)
erosion = cv2.erode(thresholded, kernel, iterations = 1)
dilate = cv2.dilate(erosion, kernel, iterations = 0)
median = cv2.medianBlur(dilate, 3)
image_contour, contours, hierarchy = cv2.findContours(median, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

roi = []
for i in range(len(contours)) :
	area= cv2.contourArea(contours[i])
	if area < 600 :
		continue
	x, y, w, h = cv2.boundingRect(contours[i])
	if h <= 0 or w <= 0 :
		continue
	x_tl = x - 5
	y_tl = y - 5
	x_br = x + h + 5
	y_br = y + w + 5
	if x_tl < 0 : 
		x_tl = 0
	if y_tl < 0:
		y_tl = 0
	if x_br >= im.shape[0] :
		x_br = im.shape[0] - 1
	if y_br >= im.shape[1] : 
		y_br = im.shape[1] - 1
	if x_tl > x_br :
		x_br, x_tl = x_tl, x_br
	if y_tl > y_br :
		y_br, y_tl = y_tl, y_br
	# img = cv2.rectangle(im, (x - 5,y - 5), (x + w + 5, y + h + 5), (0, 255, 0), 2)
	roi.append(im[y_tl:y_br, x_tl:x_br])
# print(len(roi))
time_it = time.time() - time_it

print("\nDeepank ka time : " + str(time_it) + "\n")

##Visualising###
plt.hist(normalised_red.ravel(), 256, [0,256])
cv2.namedWindow('final_output', cv2.WINDOW_NORMAL) 
cv2.imshow('Input Image', im)
cv2.waitKey(0)

###Neural Net###

##Definitions###
tf.reset_default_graph()
num_features = 4804
reg = 0.01
epsilon = tf.constant(1e-3) #Batch norm parameter
decay = tf.constant(0.999) #For exponentially moving average

num_classes = 43

hidden_1 = 500 #layers in first hidden
hidden_2 = 100 #second layer

###Placeholders###
X_tf = tf.placeholder(name='X_tf',shape=[None,num_features],dtype=tf.float32)
y_tf = tf.placeholder(name='y_tf',shape=[None,num_classes],dtype=tf.float32)
is_train = tf.placeholder(name='is_train',dtype=tf.bool)

W1 = tf.get_variable(name='W1',shape=[num_features,hidden_1],initializer=tf.contrib.layers.xavier_initializer(seed=1))
b1 = tf.get_variable(name='b1',shape=[1,hidden_1],initializer=tf.zeros_initializer())
W2 = tf.get_variable(name='W2',shape=[hidden_1,hidden_2],initializer=tf.contrib.layers.xavier_initializer(seed=1))
b2 = tf.get_variable(name='b2',shape=[1,hidden_2],initializer=tf.zeros_initializer())
W3 = tf.get_variable(name='W3',shape=[hidden_2,num_classes],initializer=tf.contrib.layers.xavier_initializer(seed=1))
b3 = tf.get_variable(name='b3',shape=[1,num_classes],initializer=tf.zeros_initializer())

###Computation###
running_mean1 = tf.Variable(tf.zeros([hidden_1]),trainable=False)
running_var1 = tf.Variable(tf.zeros([hidden_1]))
running_mean2 = tf.Variable(tf.zeros([hidden_2]))
running_var2 = tf.Variable(tf.zeros([hidden_2]))
running_mean3 = tf.Variable(tf.zeros([num_classes]))
running_var3 = tf.Variable(tf.zeros([num_classes]))

scale1 = tf.Variable(tf.ones([hidden_1]))
beta1 = tf.Variable(tf.zeros([hidden_1]))
scale2 = tf.Variable(tf.ones([hidden_2]))
beta2 = tf.Variable(tf.zeros([hidden_2]))
scale3 = tf.Variable(tf.ones([num_classes]))
beta3 = tf.Variable(tf.zeros([num_classes]))

z1 = tf.matmul(X_tf,W1) + b1
z1_BN = tf.nn.batch_normalization(z1, running_mean1,running_var1,beta1, scale1,epsilon)

a1 = tf.nn.relu(z1)

z2 = tf.matmul(a1,W2) + b2
z2_BN = tf.nn.batch_normalization(z2, running_mean2,running_var2,beta2, scale2,epsilon)  

a2 = tf.nn.relu(z2)

z3 = tf.matmul(a2,W3) + b3
z3_BN = tf.nn.batch_normalization(z3, running_mean3,running_var3,beta3, scale3,epsilon)   

#No batch norm applied here
a3 = tf.nn.softmax(z3)

y_hat = tf.argmax(a3,axis=1)

###Load Pre-trained Weights###
saver = tf.train.Saver()
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
saver.restore(sess,'saved_model_neural_network/model_bn.ckpt')

###For the SURF keypoints###
n_clusters = 16

hog = cv2.HOGDescriptor()

# ti = cv2.imread("Test_Images/2.png",1)
# roi.append(ti)
# ti = cv2.imread("Test_Images/1.ppm",1)
# roi.append(ti)
# ti = cv2.imread("Test_Images/2.ppm",1)
# roi.append(ti)
# ti = cv2.imread("Test_Images/3.ppm",1)
# roi.append(ti)
# ti = cv2.imread("Test_Images/4.ppm",1)
# roi.append(ti)
# ti = cv2.imread("Test_Images/5.ppm",1)
# roi.append(ti)
# ti = cv2.imread("Test_Images/6.ppm",1)
# roi.append(ti)
# ti = cv2.imread("Test_Images/42 .ppm",1)
# roi.append(ti)
# print(len(roi))

# for r in roi:
# cv2.imshow("R",r)
# cv2.waitKey(0)

# print("Shape : " + str(r.shape))
img = cv2.resize(r,(100,100))
img_hog = cv2.resize(r,(64,128))

time_it = time.time()
# cv2.imshow('SURF',img)
# cv2.imshow('HOG',img_hog)
# cv2.waitKey(0)

surf = cv2.xfeatures2d.SURF_create(hessianThreshold=0)
kp,des = surf.detectAndCompute(img,None)
# img4 = cv2.drawKeypoints(img,kp,None,(255,0,0),0)

time_it = time.time()- time_it
print("SURF " + str(time_it))
kmeans = None
surf_centers = None

if(des is not None and des.shape[0] > n_clusters):
	# print('SURF : ' + str(des.shape))
	# print(des.shape)

	kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(des)
	surf_centers = kmeans.cluster_centers_

	# print('KMeans : ' + str(kmeans.labels_.shape))
# else:
# 	continue
# time_it = time.time() 
# time_it = time.time()
h = hog.compute(img_hog)
# time_it = time.tim e() - time_it
# print("HOG : " + str(time_it) + "\n\n")

# time_it = time.time()
t = np.array(h)
t = t.reshape([1,-1])
surf_centers = surf_centers.reshape([1,-1])

t = np.hstack((t,surf_centers))

# print("AAAAAAA : " + str(t.shape))
t = t.reshape([1,-1])

# pred = sess.run([y_hat],feed_dict={X_tf:t,is_train:False})
time_it = time.time() - time_it
print("Time : " + str(time_it))
cv2.imshow('Image',r)
cv2.waitKey(0)

print(pred)

# sess.close()