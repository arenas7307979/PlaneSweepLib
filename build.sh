#!/bin/bash

ROOT_3RD=~/workspace/3rdParty

# C++ Configuration.
CMAKE_BIN=${ROOT_3RD}/cmake_repo/cmake-3.13.3/install/bin/cmake
CMAKE_MODULE_PATH=${ROOT_3RD}/cmake_repo/cmake-3.13.3/install/share/cmake-3.13/Modules
CMAKE_CXX_STANDARD=14
CMAKE_CXX_STANDARD_REQUIRED=ON


# Library Folder Structure.
CMAKE_BUILD_TYPE=Debug
OpenCV_DIR=${ROOT_3RD}/opencv331/installd/share/OpenCV
Eigen3_DIR=${ROOT_3RD}/eigen334/install/share/eigen3/cmake/

# GLOGS
GLOG_INCLUDE_DIRS=/usr/include
GLOG_LIBRARY_DIRS=/usr/lib/x86_64-linux-gnu/
GLOG_LIBRARIES=glog

# GFLAGS
GFLAGS_INCLUDE_DIRS=/usr/include
GFLAGS_LIBRARY_DIRS=/usr/lib/x86_64-linux-gnu/
GFLAGS_LIBRARIES=gflags

cd ../

if [ ! -e ./build ]; then
  mkdir build
fi
cd build

${CMAKE_BIN} \
  -D CMAKE_CXX_STANDARD=${CMAKE_CXX_STANDARD} \
  -D CMAKE_CXX_STANDARD_REQUIRED=${CMAKE_CXX_STANDARD_REQUIRED} \
  -D CMAKE_MODULE_PATH=${CMAKE_MODULE_PATH} \
  -D CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} \
  -D Eigen3_DIR=${Eigen3_DIR} \
  -D OpenCV_DIR=${OpenCV_DIR} \
  -D GLOG_INCLUDE_DIRS=${GLOG_INCLUDE_DIRS} \
  -D GLOG_LIBRARY_DIRS=${GLOG_LIBRARY_DIRS} \
  -D GLOG_LIBRARIES=${GLOG_LIBRARIES} \
  -D GFLAGS_INCLUDE_DIRS=${GFLAGS_INCLUDE_DIRS} \
  -D GFLAGS_LIBRARY_DIRS=${GFLAGS_LIBRARY_DIRS} \
  -D GFLAGS_LIBRARIES=${GFLAGS_LIBRARIES} \
  ../PlaneSweepLib

make -j32

cd ../PlaneSweepLib
