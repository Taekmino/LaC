cmake_minimum_required(VERSION 3.0.2)
project(gaussian_map)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  sensor_msgs
  cv_bridge
  pcl_ros
  pcl_conversions
  image_transport
  message_filters
  nav_msgs
  tf
)

find_package(OpenCV REQUIRED)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

link_directories(
  ${PCL_LIBRARY_DIRS}
)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS roscpp sensor_msgs nav_msgs cv_bridge pcl_ros pcl_conversions
)

## Executables
add_executable(gaussian_map_node src/depth_rgb_map_node.cpp)
add_executable(cost_map src/cost_map.cpp)

target_link_libraries(gaussian_map_node
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${OpenCV_LIBS}
)

target_link_libraries(cost_map
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${OpenCV_LIBS}
)
