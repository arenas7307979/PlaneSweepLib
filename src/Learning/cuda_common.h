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

#ifndef CUDA_COMMON_H_
#define CUDA_COMMON_H_

// Glog
#include <glog/logging.h>

namespace vis {
namespace cuda {

const int TILE_WIDTH = 16;
const int TILE_HEIGHT = 16;

inline int GetNumTiles(int total_size, int tile_size) {
  const int div = total_size / tile_size;
  return total_size % tile_size == 0 ? div : div + 1;
}

#define CUDA_CHECK_CALL(cuda_call)                                       \
  {                                                                      \
    cudaError err = cuda_call;                                           \
    CHECK(cudaSuccess == err) "Cuda Error: " << cudaGetErrorString(err); \
  }

#define CUDA_CHECK_ERROR                                                 \
  {                                                                      \
    cudaError err = cudaGetLastError();                                  \
    CHECK(cudaSuccess == err) "Cuda Error: " << cudaGetErrorString(err); \
  }

}  // namespace cuda
}  // namespace vis

#endif  // CUDA_COMMON_H_
