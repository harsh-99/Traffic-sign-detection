
#include <iostream>

#include "opencv2/opencv_modules.hpp"

// #ifdef HAVE_OPENCV_XFEATURES2D

#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/cudafeatures2d.hpp"
#include "opencv2/xfeatures2d/cuda.hpp"
#include <sstream>
#include <chrono>
#include <unistd.h>

using namespace std;
using namespace cv;
using namespace cv::cuda;



int main()
{	for(int j=0; j<18;j++){
    	stringstream ss;
        ss << "img"<<j<<".ppm";
        string s;
        ss >> s;
        cv::Mat img = cv::imread(s, 0);
        cv::cuda::GpuMat frame_gpu(img);
    	cv::cuda::printShortCudaDeviceInfo(cv::cuda::getDevice());
    	auto start = chrono::steady_clock::now();

    	SURF_CUDA surf;

    // detecting keypoints & computing descriptors
    	GpuMat keypoints1GPU;
    	GpuMat descriptors1GPU;
    	surf(frame_gpu, GpuMat(), keypoints1GPU, descriptors1GPU);
    	// surf(img2, GpuMat(), keypoints2GPU, descriptors2GPU);
    	auto end = chrono::steady_clock::now();
    	cout << chrono::duration_cast<chrono::milliseconds>(end-start).count() << endl;
    	cout << "FOUND " << keypoints1GPU.size() << " keypoints on first image" << endl;
    	cout << "FOUND " << descriptors1GPU.size() << " descriptors on first image" << endl;

	    // matching descriptors
	    // Ptr<cv::cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher(surf.defaultNorm());
	    // vector<DMatch> matches;
	    // matcher->match(descriptors1GPU, descriptors2GPU, matches);

	    // downloading results
	    // vector<KeyPoint> keypoints1, keypoints2;
	    // vector<float> descriptors1, descriptors2;
	    // surf.downloadKeypoints(keypoints1GPU, keypoints1);
	    // // surf.downloadKeypoints(keypoints2GPU, keypoints2);
	    // surf.downloadDescriptors(descriptors1GPU, descriptors1);
	    // // surf.downloadDescriptors(descriptors2GPU, descriptors2);

	    // // drawing the results
	    // Mat img_matches;
	    // drawMatches(Mat(img1), keypoints1, Mat(img2), keypoints2, matches, img_matches);

	    // namedWindow("matches", 0);
	    // imshow("matches", img_matches);
	    // waitKey(0);

}
}