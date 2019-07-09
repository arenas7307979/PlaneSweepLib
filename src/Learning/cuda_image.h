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

#ifndef CUDA_IMAGE_H_
#define CUDA_IMAGE_H_

// Cuda
#include <cuda_runtime.h>

// OpenCV
#include <opencv2/core.hpp>

namespace vis {
namespace cuda {

class CudaImage {
 public:
  CudaImage() { m_dev_ptr = nullptr; }

  // Host and device functions.
  inline __host__ __device__ unsigned char *GetDevAddr() const {
    return m_dev_ptr;
  };
  inline __host__ __device__ int GetWidth() const { return m_width; }
  inline __host__ __device__ int GetHeight() const { return m_height; }
  inline __host__ __device__ int GetNumChannels() const {
    return m_num_channels;
  }
  inline __host__ __device__ int GetPitch() const { return (int)m_dev_pitch; }
  inline __host__ __device__ bool IsAllocated() const {
    return m_dev_ptr != nullptr;
  }
  inline __host__ __device__ int2 GetSize() const {
    return make_int2(m_width, m_height);
  }

  // Device only functions.
  inline __device__ unsigned char &operator()(unsigned int x, unsigned int y) {
    return *((m_dev_ptr + y * m_dev_pitch) + x);
  };
  inline __device__ const unsigned char &operator()(unsigned int x,
                                                    unsigned int y) const {
    return *((m_dev_ptr + y * m_dev_pitch) + x);
  };
  inline __device__ unsigned char &operator()(unsigned int x, unsigned int y,
                                              unsigned int c) {
    return *(m_dev_ptr + y * m_dev_pitch + x * m_num_channels + c);
  };
  inline __device__ const unsigned char &operator()(unsigned int x,
                                                    unsigned int y,
                                                    unsigned int c) const {
    return *(m_dev_ptr + y * m_dev_pitch + x * m_num_channels + c);
  };

  // Host only functions.
  void AllocatePitched(int width, int height, int num_channels);
  void AllocatePitchedAndUpload(const cv::Mat &img);
  void ReallocatePitchedAndUpload(const cv::Mat &img);
  void Download(cv::Mat &img);
  void Clear(unsigned char value);
  void Deallocate();

 private:
  unsigned char *m_dev_ptr;
  int m_width;
  int m_height;
  int m_num_channels;
  size_t m_dev_pitch;
};
}  // namespace cuda
}  // namespace vis

#endif  // CUDA_IMAGE_H_
