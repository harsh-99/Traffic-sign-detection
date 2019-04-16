# Traffic Sign Detection

Most of the current state of art solution to this well researched problem is using different types of deep learning based pipeline. Their choice of using model is based to priority difference given to computation time and accuracy. The faster one uses YOLO, SSD etc and model with better accuracy uses Feature pyramid network, Faster RCNN, Segnet etc. Here we propose a conventional approach of traffic sign detection i.e. using Traditional feature descriptor like SURF and HOG. 
 
## Region Proposal

We trained a Haar cascade classifier to propose ROI, the results for the same can seen /output/roi_haar. The method worked pretty well with a decent accuracy. The time taken by the haar Region of proposal is 0.1 seconds. We have also tried to implement the cascade classifier using cuda enabled opencv. The code for the same can be found at /GPU_implemetation/haar.cpp. The only boundation to implement this to have cuda enabled opencv in the machine.   
 ![alt text](https://github.com/harsh-99/Traffic-sign-detection/blob/new/Outputs/roi_haar/roiimg1.jpg)
 ![alt text](https://github.com/harsh-99/Traffic-sign-detection/blob/new/Outputs/roi_haar/roiimg4.jpg)
 ![alt text](https://github.com/harsh-99/Traffic-sign-detection/blob/new/Outputs/roi_haar/roiimg3.jpg)

## Feature descriptors

- Using feature descriptor like SURF, SIFT, HOG

We tried both of the possible approaches. Firstly we used the traditional approach i.e. Histogram of oriented gradient + SVM which gives us an classification accuracy of 70% which is very poor in current scenario. Secondly we trained a deep learning neural network which was trained on Speeded up roboust feature(SURF) and Histogram of oriented gradient which showed major change in accuracy. As the output of SURF is not fixed so we used K-mean clustering to make output fixed and we clustre the feature who are close to each other. Intuitivly it is correct to say that we should merge the features which are close to each other. So that we can get different tyoe of feature. We acheived an accuracy of 97% on GTSRB dataset (43 classes). Further we used a CNN for reducing the time. The input to CNN was appended feature of SURF and HOG. But using an appended HOG and SURF features does not makes sense. The drawback was time of computation SURF takes 44 millisecond on GPU i.e. 0.045 per region proposal. To further reduce time we used an end to end CNN for classification which is much faster as compared to computing SURF and clustering them and then using CNN.   

### codes for feature descriptors 
```
featureandCSV.py 
```
It computes feature (SURF + HOG), cluster SURF feature and then append both the feature and sabe them to a CSV file. 

### Neural Network

There are just two notebook available here, one takes input of SURF+HOG in Neural network i.e. a fully connected network. Other architecture involves CNN with fully connected layers. 


### Xml files

There are many XML files available in the xml_files/ folder which contains output xml files which we obtained by training Harr for region proposal. There are files, used to detect triangular signs, circular signs, and some specific signs such as Stop and Bumper sign.  

### Codes

```
GPU_implementation/haar.cpp - Used to detect region of interest on GPU
GPU_implementation/SURF.cpp - Used to detect SURF features on GPU
Neural Network/CNN_end_to_end.ipynb - End to End CNN architecture
Neural Network/NN_TRAFFIC_SIGN.ipynb - End to End Neural Network architecture
Segmentation/final_segmentation - Uses basic image processing technique to detect the traffic sign.
```
