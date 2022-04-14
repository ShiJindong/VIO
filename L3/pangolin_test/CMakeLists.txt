cmake_minimum_required( VERSION 2.8 )
project( pangolin_test )

#sophus
#find_package( Sophus REQUIRED )

# 添加Eigen头文件
# find_package(Eigen3 REQUIRED)
include_directories("/usr/include/eigen3")

find_package(Pangolin REQUIRED)


#set(Sophus_INCLUDE_DIRS "/usr/local/include") 
#set(Sophus_INCLUDE_DIRS "/usr/local/include/sophus") 
#set(Sophus_LIBRARIES "/usr/local/lib/libSophus.so")

include_directories(
${Eigen3_INCLUDE_DIRS}
${Pangolin_INCLUDE_DIRS}
)

add_executable( pangolin_test pangolin_test.cpp)
target_link_libraries( pangolin_test ${Pangolin_LIBRARIES} )
# target_link_libraries( pangolin_test )