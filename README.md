# Traffic Sign Detection

Most of the current state of art solution to this well researched problem is using different types of deep learning based pipeline. Their choice of using model is based to priority difference given to computation time and accuracy. The faster one uses YOLO, SSD etc and model with better accuracy uses Feature pyramid network, Faster RCNN, Segnet etc. Here we propose a conventional approach of traffic sign detection i.e. using Traditional feature descriptor like SURF and HOG. The pipeline of our proposed algorithm is as following. First of all we propose Region Proposal folowed by calculating feature descriptor followed by a neural network. Each of the method is explained properly in sub-section. 
 

## Region Proposal

Firstly we tried a basic image processing based pipeline for traffic sign detection. Here in this, we know that traffic sign generally have red color so we take the red channel of the given image and apply threshold. We thresholded top 20% of total pixels. After this we used watershed algorithm as well as contour detection to propose region out of that.(The code for this is /segmentation/fianl_segmentation_ip.py). After this we tried to propose using Haar. We trained a Haar cascade classifier to propose ROI, the results for the same can seen /output/roi_haar. The method worked pretty well with a decent accuracy. The time taken by the haar Region of proposal is 0.1 seconds. We have also tried to implement the cascade classifier using cuda enabled opencv. The code for the same can be found at /GPU_implemetation/haar.cpp. The only boundation to implement this to have cuda enabled opencv in the machine.   
 ![alt text](https://github.com/harsh-99/Traffic-sign-detection/blob/new/Outputs/roi_haar/roiimg1.jpg)
 ![alt text](https://github.com/harsh-99/Traffic-sign-detection/blob/new/Outputs/roi_haar/roiimg4.jpg)
 ![alt text](https://github.com/harsh-99/Traffic-sign-detection/blob/new/Outputs/roi_haar/roiimg3.jpg)

<<<<<<< HEAD

## Feature descriptors
=======
 ![alt text](https://github.com/harsh-99/Traffic-sign-detection/blob/new/Outputs/roi_haar/roiimg1.jpg)
 ![alt text](https://github.com/harsh-99/Traffic-sign-detection/blob/new/Outputs/roi_haar/roiimg4.jpg)
 ![alt text](https://github.com/harsh-99/Traffic-sign-detection/blob/new/Outputs/roi_haar/roiimg3.jpg)
### Feature descriptors
>>>>>>> 602b48f2d0f22caf54164ef10ae0b9b877dc9fd1

There are two ways to proceed after getting region of interest-:
- Using feature descriptor like SURF, SIFT, HOG
- Using an end to end Deep learning Model
We tried both of the possible approaches. Firstly we used the traditional approach i.e. Histogram of oriented gradient + SVM which gives us an classification accuracy of 70% which is very poor in current scenario. Secondly we trained a deep learning neural network which was trained on Speeded up roboust feature(SURF) and Histogram of oriented gradient which showed major change in accuracy. As the output of SURF is not fixed so we used K-mean clustering to make output fixed and we clustre the feature who are close to each other. Intuitivly it is correct to say that we should merge the features which are close to each other. So that we can get different tyoe of feature. We acheived an accuracy of 97% on GTSRB dataset (43 classes). Further we used a CNN for reducing the time. The input to CNN was appended feature of SURF and HOG. But using an appended HOG and SURF features does not makes sense. The drawback was time of computation SURF takes 44 millisecond on GPU i.e. 0.045 per region proposal. To further reduce time we used an end to end CNN for classification which is much faster as compared to computing SURF and clustering them and then using CNN.   

### codes for feature descriptors 
```
featureandCSV.py 
```
It computes feature (SURF + HOG), cluster SURF feature and then append both the feature and sabe them to a CSV file. 

### Neural Network

A step by step series of examples that tell you how to get a development env running

Say what the step will be

```
Give the example
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo

## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc

