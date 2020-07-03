cmake_minimum_required(VERSION 3.4.1)

set(CMAKE_CXX_FLAGS "-std=c++11")

set(CMAKE_PREFIX_PATH "/home/wei/Ubuntu/Libs/opencv-3.4.3/install")
set(OpenCV_INCLUDE_DIRS "/home/wei/Ubuntu/Libs/opencv-3.4.3/install/include") 
find_package(OpenCV REQUIRED)

set(MNN_HOME "/home/wei/Ubuntu/Libs/mnn/install")
set(MNN_LIB_DIR ${MNN_HOME}/lib)
set(MNN_INCLUDE_DIR 
    ${MNN_HOME}/include/MNN
)

set(PYBIND11_INCLUDE_DIR "/home/wei/Ubuntu/Libs/pybind11-2.5.0/include")
set(PYTHON_INCLUDE_DIR "/home/wei/Ubuntu/Softwares/anaconda3/include/python3.7m")

link_directories(${MNN_LIB_DIR})

add_library(
    pydetector
    SHARED
    "./cpp/facedetector_pybind.cpp"
)

target_link_libraries(pydetector facedetector-mtcnn MNN  ${OpenCV_LIBS})
target_include_directories(pydetector PRIVATE ${MNN_INCLUDE_DIR} ${PYBIND11_INCLUDE_DIR} ${PYTHON_INCLUDE_DIR} ${MNN_HOME}/include)

