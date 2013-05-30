cmake_minimum_required(VERSION 2.8.3)
project(rgbd_graph_segmentation)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS cv_bridge pcl pcl_ros image_geometry dynamic_reconfigure)

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
   INCLUDE_DIRS include
   LIBRARIES rgbd_graph_segmentation
   CATKIN_DEPENDS cv_bridge pcl pcl_ros image_geometry dynamic_reconfigure
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(include ${catkin_INCLUDE_DIRS})

## Declare a cpp library
add_library(rgbd_graph_segmentation
    src/segment.cpp
    src/wrapper.cpp)

# Felzenszwalb segmenter
set(BUILD_FLAGS "-Wall -Wextra -Werror -std=c++0x -Wno-unknown-pragmas -Wno-deprecated-declarations -Wno-unused-function -Wno-unused-parameter -Wno-invalid-offsetof ")
set_target_properties(rgbd_graph_segmentation PROPERTIES COMPILE_FLAGS ${BUILD_FLAGS} )

