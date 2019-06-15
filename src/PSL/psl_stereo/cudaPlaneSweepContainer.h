
#ifndef CUDAPLANESWEEPCONTAINER_H_
#define CUDAPLANESWEEPCONTAINER_H_

// PSL
#include <psl_cudaBase/deviceBuffer.h>
#include <psl_cudaBase/deviceImage.h>

namespace PSL_CUDA {

struct BestPlaneBuffer {
  DeviceBuffer<int> best_plane_idx;
  DeviceBuffer<float> best_costs;
  DeviceBuffer<float> second_best_costs;
};
}

#endif // CUDAPLANESWEEPCONTAINER_H_
