
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

#ifndef CUDA_BUFFER_H_
#define CUDA_BUFFER_H_

// System
#include <string>

// Cuda
#include <cuda_runtime.h>

namespace vis {
namespace cuda {

template <typename T>
class CudaBuffer {
 public:
  CudaBuffer();

  // Host and device functions.
  inline __host__ __device__ T* GetDevAddr() const { return m_dev_ptr; }
  inline __host__ __device__ int GetWidth() const { return m_width; }
  inline __host__ __device__ int GetHeight() const { return m_height; }
  inline __host__ __device__ int GetPitch() const { return (int)m_dev_pitch; }
  inline __host__ __device__ bool IsAllocated() const {
    return m_dev_ptr != nullptr;
  }
  inline __host__ __device__ int2 GetSize() const {
    return make_int2(m_width, m_height);
  }

  // Device only functions.
  inline __device__ T& operator()(size_t x, size_t y) {
    return *((T*)((char*)m_dev_ptr + y * m_dev_pitch) + x);
  }
  inline __device__ const T& operator()(size_t x, size_t y) const {
    return *((T*)((char*)m_dev_ptr + y * m_dev_pitch) + x);
  }

  // Host only functions.
  void AllocatePitched(int width, int height);
  void ReallocatePitched(int width, int height);
  void Clear(T value);
  void Download(T* host_ptr, size_t dst_pitch);
  void DoanloadAndDisplay(int wait_time, T min_val, T max_val,
                          std::string window_title = "Buffer");
  void Upload(T* host_ptr, size_t dst_pitch);
  void Deallocate();

 private:
  T* m_dev_ptr;
  int m_width, m_height;
  size_t m_dev_pitch;
};

}  // namespace cuda
}  // namespace vis

#endif  // CUDA_BUFFER_H_
