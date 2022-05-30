cmake_minimum_required(VERSION 3.0.2)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
project(object_detection)
enable_language(CUDA)
set(CMAKE_CXX_COMPILIER "/usr/bin/g++")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O3 -Wno-write-strings")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath -Wl,$ORIGIN")

find_package(CUDA REQUIRED)

find_package(catkin REQUIRED COMPONENTS
    roscpp
    roslib
    image_transport
    camera_info_manager
    sensor_msgs
    bounding_box_msgs
    cv_bridge
)


find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED)


catkin_package(
    DEPENDS
        roscpp
        image_transport
        camera_info_manager
        sensor_msgs
        image_geometry
)






include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

file(GLOB_RECURSE sources modules/*.hpp modules/*.cpp modules/*.h modules/*.cu extra/*.h)
add_library(detector SHARED ${sources})

target_include_directories(detector PRIVATE extra/ modules/ ${OpenCV_INCLUDE_DIRS} ${CUDA_TOOLKIT_ROOT_DIR}/include)
target_link_libraries(detector ${YAML_CPP_LIBRARIES} yaml-cpp nvinfer nvinfer_plugin nvparsers "stdc++fs")
add_executable(object_detector src/object_detection.cpp)
target_include_directories(object_detector PRIVATE modules/ extra/)
target_link_libraries(object_detector detector ${OpenCV_LIBS})

target_link_libraries(object_detector
   ${catkin_LIBRARIES}
   ${Boost_LIBRARIES}
)



