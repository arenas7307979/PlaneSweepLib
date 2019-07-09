

// Copyright 2016 Christian Haene (ETH Zuerich)

// PSL is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// PSL is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with PSL.  If not, see <http://www.gnu.org/licenses/>.

#ifndef PLANE_SWEEP_STEREO_INTL_BUFFER_H_
#define PLANE_SWEEP_STEREO_INTL_BUFFER_H_

// Original Cuda
#include <cuda_buffer.h>

namespace vis {
namespace mvs {

struct SAD {
  cuda::CudaBuffer<float> dev_box_filter;
};

struct ZNCC {};

struct SubPixel {
  cuda::CudaBuffer<float> dev_subpix_accum_prev1;
  cuda::CudaBuffer<float> dev_subpix_accum_prev2;
  cuda::CudaBuffer<float> dev_subpix_plane_offsets;
  cuda::CudaBuffer<float> dev_2nd_best_costs;
};

struct Common {
  cuda::CudaBuffer<float> dev_cost_accum;
};

struct BestK {
  cuda::CudaBuffer<float> dev_cost_accum_before;
  cuda::CudaBuffer<float> dev_cost_accum_after;
};

struct BestPlane {
  cuda::CudaBuffer<int> dev_best_plane_idx;
  cuda::CudaBuffer<float> dev_best_scores;
};

}  // namespace mvs
}  // namespace vis

#endif  // PLANE_SWEEP_STEREO_INTL_BUFFER_H_
