#include <opencv2/core.hpp>
#include <opencv2/objdetect.hpp>           // CascadeClassifier
#include <opencv2/cudaobjdetect.hpp>       // Mat, Point, Scalar, Size
#include <opencv2/core/cuda.hpp>           // cuda::isCompatible
#include <opencv2/highgui.hpp>             // imshow, waitKey
#include <opencv2/imgcodecs.hpp>           // imread
#include <opencv2/imgproc.hpp>             // cvtColor, equalizeHist
#include <iostream>
#include <sstream>
#include <chrono>
#include <unistd.h>
using namespace std;
// using namespace cv; 

int main(){
    for(int j=0; j<18;j++){
        stringstream ss;
        ss << "img"<<j<<".ppm";
        string s;
        ss >> s;
        cv::Mat frame = cv::imread(s, 0);
        cv::Ptr <cv::cuda::CascadeClassifier> cascade_gpu_ptr = cv::cuda::CascadeClassifier::create("/home/harsh/opencv_project/circular_cascade.xml");
        auto start = chrono::steady_clock::now();
        cv::cuda::GpuMat frame_gpu(frame);
        cv::cuda::GpuMat faces_gpu;
        vector<cv::Rect> face_rects;
        // cascade_gpu_ptr->setMaxNumObjects(1.15);
        cascade_gpu_ptr->setScaleFactor(1.1);
        cascade_gpu_ptr->setMinNeighbors(5);
        cascade_gpu_ptr->detectMultiScale(frame_gpu, faces_gpu);
        cascade_gpu_ptr->convert(faces_gpu, face_rects);
        auto end = chrono::steady_clock::now();
        cout<< face_rects.size()<< endl;
        cout << "Hello my CPP is changing" << endl;
        cout<< chrono::duration_cast<chrono::milliseconds>(end-start).count() << endl;
        for(int i=0;i<face_rects.size();++i){
            cv::rectangle(frame,face_rects[i],cv::Scalar(255));
        }

        cv::imshow("Faces",frame);
        cv::waitKey(2000);
    }
}