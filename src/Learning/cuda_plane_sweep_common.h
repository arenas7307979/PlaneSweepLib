
// This file is part of PlaneSweepLib (PSL)

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

#ifndef CUDA_PLANE_SWEEP_COMMON_H_
#define CUDA_PLANE_SWEEP_COMMON_H_

// Cuda
#include <cuda_runtime.h>

namespace vis {
namespace cuda {
namespace mvs {

const int PLANE_SWEEP_BLOCK_WIDTH = 32;
const int PLANE_SWEEP_BLOCK_HEIGHT = 8;

// BOX FILTER
const int PLANE_SWEEP_BOX_FILTER_NUM_THREADS = 128;
const int PLANE_SWEEP_BOX_FILTER_ROWS_PER_THREAD = 20;

__forceinline__ __device__ void WarpCoordinateViaHomography(
    const int x, const int y, const float h11, const float h12, const float h13,
    const float h21, const float h22, const float h23, const float h31,
    const float h32, const float h33, float *xw, float *yw) {
  float xw_raw = h11 * x + h12 * y + h13;
  float yw_raw = h21 * x + h22 * y + h23;
  float zw_raw = h31 * x + h32 * y + h33;

  *xw = xw_raw / zw_raw;
  *yw = yw_raw / zw_raw;
}

}  // namespace mvs
}  // namespace cuda
}  // namespace vis

#endif  // CUDA_PLANE_SWEEP_COMMON_H_
