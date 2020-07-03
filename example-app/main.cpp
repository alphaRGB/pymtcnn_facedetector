#include<iostream>
#include<string>
#include<opencv2/opencv.hpp>
#include<vector>
#include"../src/mtcnn.h"
#include"../src/imgProcess.h"

void drawBoxes(cv::Mat& image, std::vector<Bbox>& boxes){
    for(auto box:boxes){
        cv::rectangle(image, box.rect(), cv::Scalar(0,255,255), 2);
    }
}


int detectCamera(){

    const char* modelPath = "./models";
    cv::Mat frame;
    cv::VideoCapture cap;
    if(!cap.open(0)){
        std::cerr<<"Failed to open camera!"<<std::endl;
        return -1;
    }

    MTCNN mtcnn;
    mtcnn.load(modelPath,2);

    cv::TickMeter meter;
    float scale = 0.25f;
    cap>>frame;
    while (true)
    {
        cap>>frame;
        if (frame.empty())
            break;

        meter.reset();
        meter.start();
        cv::Mat image = frame.t();
        //cv::rotate(frame, frame, cv::ROTATE_90_COUNTERCLOCKWISE);
        cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
        std::vector<Bbox> boxes;
        mtcnn.detect(image, boxes, scale);
        meter.stop();
        
        printf("num=%d, time=%f ms\n", boxes.size(), meter.getTimeMilli());
        drawBoxes(frame, boxes);
        cv::imshow("MTCNN_MNN", frame);
        if(cv::waitKey(33)=='q')
            break;
    }
    
    return 0;
}


int main(){

    detectCamera();
}
