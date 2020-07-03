cmake_minimum_required(VERSION 3.4.1)

set(CMAKE_CXX_FLAGS "-std=c++11 -O3")


set(CMAKE_PREFIX_PATH "/home/wei/Ubuntu/Libs/opencv-3.4.3/install")
set(OpenCV_INCLUDE_DIRS "/home/wei/Ubuntu/Libs/opencv-3.4.3/install/include") 
find_package(OpenCV REQUIRED)

set(MNN_HOME "/home/wei/Ubuntu/Libs/mnn/install")
set(MNN_LIB_DIR ${MNN_HOME}/lib)
set(MNN_INCLUDE_DIR 
    ${MNN_HOME}/include/MNN
)

link_directories(${MNN_LIB_DIR})

add_library(
    facedetector-mtcnn
    SHARED
    "./src/imgProcess.cpp"
    "./src/Bbox.cpp"
    "./src/mtcnn.cpp"
    "./src/net.cpp"
)

message(${MNN_INCLUDE_DIR})

target_include_directories(facedetector-mtcnn PRIVATE ${MNN_INCLUDE_DIR} ${MNN_HOME}/include)
target_link_libraries(facedetector-mtcnn MNN  ${OpenCV_LIBS})

# example
add_executable(
    facedetect-demo
    "./example-app/main.cpp"
)
target_link_libraries(facedetect-demo facedetector-mtcnn MNN  ${OpenCV_LIBS})
target_include_directories(facedetect-demo PRIVATE ${MNN_INCLUDE_DIR} ${MNN_HOME}/include)


# pydetector
add_subdirectory("./pydetector_mtcnn")

