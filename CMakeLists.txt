cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project(face_registion)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

ADD_COMPILE_OPTIONS(-std=c++11 )
ADD_COMPILE_OPTIONS(-std=c++14 )
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )
set(PCL_DIR "/usr/lib/x86_64-linux-gnu/cmake/pcl/PCLConfig.cmake")

list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )
find_package(Eigen3 3.3.4 QUIET REQUIRED NO_MODULE)
find_package(Boost COMPONENTS program_options REQUIRED )
#find_package(PCL 1.7
	     #PATHS /usr/include/pcl-1.10/pcl #look here
	     #NO_DEFAULT_PATH
	     #REQUIRED)
find_package(PCL REQUIRED)
find_package(Ceres REQUIRED )

include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})
 
include_directories( ${CERES_INCLUDE_DIRS} )

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (FCCF FCCF.cpp)
target_link_libraries (FCCF ${CERES_LIBRARIES} ${PCL_LIBRARIES})