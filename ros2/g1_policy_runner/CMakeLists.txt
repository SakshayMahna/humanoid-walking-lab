cmake_minimum_required(VERSION 3.8)
project(g1_policy_runner)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2 REQUIRED)

# ONNX Runtime C++ is installed by hand into /usr/local (docker/Dockerfile), as in Diego's README
find_library(ONNXRUNTIME_LIB onnxruntime PATHS /usr/local/lib REQUIRED)

add_executable(policy_runner
  src/main.cpp
  src/policy_runner.cpp
)
target_include_directories(policy_runner PRIVATE include /usr/local/include)
target_compile_features(policy_runner PRIVATE cxx_std_17)
ament_target_dependencies(policy_runner rclcpp geometry_msgs sensor_msgs nav_msgs std_msgs tf2)
target_link_libraries(policy_runner ${ONNXRUNTIME_LIB})

install(TARGETS policy_runner DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY launch config DESTINATION share/${PROJECT_NAME})

ament_package()
