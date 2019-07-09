project(PSL)
cmake_minimum_required (VERSION 2.6)

#set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CONFIGURATION_TYPES ${CMAKE_BUILD_TYPE} CACHE STRING "" FORCE)
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin )

find_package(Eigen3 REQUIRED CONFIG PATHS ${Eigen3_DIR} NO_DEFAULT_PATH)
find_package(OpenCV REQUIRED CONFIG PATHS ${OpenCV_DIR} NO_DEFAULT_PATH)
find_package(CUDA REQUIRED)
set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE OFF)
set(Boost_USE_STATIC_LIBS ON)
find_package(Boost REQUIRED COMPONENTS system filesystem program_options)

include_directories(${EIGEN3_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})

include_directories("src/PSL")
add_subdirectory(src)
