cmake_minimum_required(VERSION 2.8)
project(bgk_traversability_mapping)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3")

find_package(catkin REQUIRED COMPONENTS
  tf
  roscpp
  rospy
  cv_bridge
  image_transport
  # pcl library
  pcl_ros
  pcl_conversions
  # msgs
  std_msgs
  sensor_msgs
  geometry_msgs
  nav_msgs
)

find_package(PCL REQUIRED QUIET)
find_package(OpenCV REQUIRED QUIET)

catkin_package(
  INCLUDE_DIRS include
  DEPENDS PCL
)

# include directories
include_directories(
  include 
  ${catkin_INCLUDE_DIRS} 
  ${PCL_INCLUDE_DIRS} 
  ${OpenCV_INCLUDE_DIRS}
)

# link directories
link_directories(
  include
  ${OpenCV_LIBRARY_DIRS}
  ${PCL_LIBRARY_DIRS}
)

###########
## Build ##
###########

# Filter
add_executable(bgk_trav_filter src/bgk_trav_filter.cpp)
target_link_libraries(bgk_trav_filter ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES})

# Mapping
add_executable(bgk_trav_map src/bgk_trav_map.cpp)
target_link_libraries(bgk_trav_map ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES})
