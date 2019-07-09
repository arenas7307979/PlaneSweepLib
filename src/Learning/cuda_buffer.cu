
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

// Self Header
#include <cuda_buffer.h>

// Original
#include <cuda_common.h>

namespace {

template <typename T>
__global__ void ClearKernel(vis::cuda::CudaBuffer<T> cuda_buffer, T value) {

  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < cuda_buffer.GetWidth() && y < cuda_buffer.GetHeight()) {
    cuda_buffer(x, y) = value;
  }
}
}

namespace vis {
namespace cuda {

template <typename T> CudaBuffer<T>::CudaBuffer() { m_dev_ptr = nullptr; }

template <typename T>
void CudaBuffer<T>::AllocatePitched(int width, int height) {

  // Pitch will return from this function call.
  CUDA_CHECK_CALL(
      cudaMallocPitch(&m_dev_ptr, &m_dev_pitch, width * sizeof(T), height));
  this->m_width = width;
  this->m_height = height;
}

template <typename T>
void CudaBuffer<T>::ReallocatePitched(int width, int height) {
  if (this->m_dev_ptr != nullptr) {
    // If already satisfied.
    if (width == this->m_width && height == this->m_height) {
      return;
    }
    // Dump first
    Deallocate();
  }
  AllocatePitched(width, height);
}

template <typename T> void CudaBuffer<T>::Clear(T value) {

  dim3 grid_dim(GetNumTiles(m_width, TILE_WIDTH),
                GetNumTiles(m_height, TILE_HEIGHT));
  dim3 block_dim(TILE_WIDTH, TILE_HEIGHT);

  ClearKernel<<<grid_dim, block_dim>>>(*this, value);
}

template <typename T>
void CudaBuffer<T>::Download(T *host_ptr, size_t host_pitch) {
  CUDA_CHECK_CALL(cudaMemcpy2D(host_ptr, host_pitch, m_dev_ptr, m_dev_pitch,
                               m_width * sizeof(T), m_height,
                               cudaMemcpyDeviceToHost);)
}

template <typename T>
void CudaBuffer<T>::Upload(T *host_ptr, size_t host_pitch) {

  CUDA_CHECK_CALL(cudaMemcpy2D(m_dev_ptr, m_dev_pitch, host_ptr, host_pitch,
                               m_width * sizeof(T), m_height,
                               cudaMemcpyHostToDevice);)
}

template <typename T> void CudaBuffer<T>::Deallocate() {
  CUDA_CHECK_CALL(cudaFree((void *)m_dev_ptr);)
}

// Template instantiation.
template class CudaBuffer<float>;
template class CudaBuffer<int>;
}
}