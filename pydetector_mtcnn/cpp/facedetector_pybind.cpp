#include<iostream>
#include<string>
#include<opencv2/opencv.hpp>
#include<vector>
#include"../../src/mtcnn.h"
#include"../../src/imgProcess.h"
#include"pybind11/pybind11.h"
#include"pybind11/numpy.h"
#include"pybind11/stl.h"

namespace py=pybind11;


class Box
{
public:
    int x;
    int y;
    int width;
    int height;
    float score;
public:
    Box(int x, int y, int w, int h, float score){
        this->x = x;
        this->y = y;
        this->width = w;
        this->height = h;
        this->score = score;
    }
    ~Box(){}
};


class FaceDetector
{
private:
    MTCNN mtcnn;
    float scale_;
public:
    FaceDetector(std::string model_path, int num_thread, float scale=1.0f){
        mtcnn.load(model_path, num_thread);
        scale_ = scale;
    }
    ~FaceDetector(){}

    std::vector<Box> detect(const py::array_t<uint8_t>& img_np_bgr){
        assert(img_np_bgr.ndim()==3);
        auto handle = img_np_bgr.request();
        int rows = handle.shape[0];
        int cols = handle.shape[1];
        int channels = handle.shape[2];
        assert(channels==3);
        cv::Mat frame = cv::Mat(rows, cols, CV_8UC3, (uint8_t*)handle.ptr).clone();
        cv::Mat image = frame.t();
        cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

        std::vector<Bbox> boxes;
        mtcnn.detect(image, boxes, scale_);
        
        std::vector<Box> outs;
        for(auto &item:boxes){
            outs.push_back({item.rect().x, item.rect().y, item.rect().width, item.rect().height, item.score});
        }

        return outs;      
    }

};


PYBIND11_MODULE(pydetector, m){
    m.doc() = "Face Detector based on MTCNN";
    
    py::class_<Box>(m, "Box")
        .def(py::init<int, int, int, int, float>())
        .def_readwrite("x", &Box::x)
        .def_readwrite("y", &Box::y)
        .def_readwrite("width", &Box::width)
        .def_readwrite("height", &Box::height)
        .def_readwrite("score", &Box::score);
    
    py::class_<FaceDetector>(m, "FaceDetector")
        .def(py::init<std::string, int, float>(), py::arg("model_path"), py::arg("num_thread"), py::arg("scale"))
        .def("detect", &FaceDetector::detect, py::arg("img_bgr"), py::return_value_policy::reference);

}


