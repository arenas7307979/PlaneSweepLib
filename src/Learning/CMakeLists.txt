
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

cuda_compile(CUDA_SRC_FILES
  cuda_buffer.cu
  cuda_image.cu
  cuda_plane_sweep.cu
  cuda_plane_sweep_common.cu
  OPTIONS
  -std=c++14
)

add_library(psl STATIC
  ${CUDA_SRC_FILES}
  cuda_buffer.cc
  camera_matrix.cc
  depth_map.cc
  color_map_jet.cc
  plane_sweep_stereo.cc
)

#target_include_directories(psl PUBLIC
#  ${CMAKE_CURRENT_SOURCE_DIR}
#)

add_executable(plane_sweep_stereo_test 
  plane_sweep_stereo_test.cc)

target_link_libraries(plane_sweep_stereo_test
  psl
  ${OpenCV_LIBS}
  ${GLOG_LIBRARIES}
  ${GFLAGS_LIBRARIES}
  ${CUDA_CUDART_LIBRARY}
)
