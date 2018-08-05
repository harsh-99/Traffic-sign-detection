import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread("6.jpeg")################ enter path to file ####################
normalised_red = np.zeros((img.shape[0],img.shape[1]), np.uint8)
for i in range(0, img.shape[0]) :
	for j in range(0, img.shape[1]) :
		a = int(img[i,j,0]) + int(img[i,j,1]) + int(img[i,j,2])
		normalised_red[i,j] = 255*float(img[i,j,2])/(a + 1e-7)
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

# sure background area
sure_bg = cv2.dilate(median, kernel, iterations = 3)
# cv2.imshow('sure bg',  sure_bg)
# Finding sure foreground area
dist_transform = cv2.distanceTransform(median, cv2.DIST_L2, 5)
# cv2.imshow('transform', dist_transform)
ret, sure_fg = cv2.threshold(dist_transform, 0.3*dist_transform.max(), 255, 0)
# cv2.imshow('sure fg',  sure_fg)
# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)

# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)
# Add one to all labels so that sure background is not 0, but 1
markers = markers + 1

# Now, mark the region of unknown with zero
markers[unknown == 255] = 0

markers = cv2.watershed(img,markers)
cropx = img.shape[1] - 2
cropy = img.shape[0] - 2
startx = img.shape[1]//2 - cropx//2
starty = img.shape[0]//2 - cropy//2
markers_cropped = markers[starty : starty + cropy, startx : startx + cropx]
i, j = np.where(markers_cropped == -1)
maxX = np.max(i)
minX = np.min(i)
maxY = np.max(j)
minY = np.min(j)
print maxX, minX, maxY, minY
# Getting ROI
roi = img[minX:maxX, minY:maxY]
cv2.imwrite('traffic_sign.jpeg', roi)
cv2.imshow('roi', roi)

img[markers == -1] = [0, 255, 0]


plt.hist(normalised_red.ravel(), 256, [0,256])
# plt.show()
# cv2.imshow('thresh', thresholded)
cv2.imshow('final_output', img)
# cv2.imshow('image_dilate', dilate)
# cv2.imshow('median', median)
cv2.waitKey(0)