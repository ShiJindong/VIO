CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
project(quaternion_lie_algebra_update)

#find_package(Eigen3 3.1.0 REQUIRED)
include_directories("/usr/include/eigen3")
find_package(Sophus REQUIRED)

include_directories(
        ${Eigen3_INCLUDE_DIRS}
        ${Sophus_INCLUDE_DIRS}
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/OUTPUT)
add_executable(quaternion_lie_algebra_update quaternion_lie_algebra_update.cpp)
target_link_libraries(quaternion_lie_algebra_update ${Sophus_LIBRARIES} fmt)
