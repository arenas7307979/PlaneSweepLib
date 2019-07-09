
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
#include <cuda_image.h>

// Glog
#include <glog/logging.h>

// Original
#include <cuda_common.h>

namespace {

__global__ void ClearKernel(vis::cuda::CudaImage cuda_img,
                            unsigned char value) {

  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < cuda_img.GetWidth() * cuda_img.GetNumChannels() &&
      y < cuda_img.GetHeight()) {
    cuda_img(x, y) = value;
  }
}
}

namespace vis {
namespace cuda {

void CudaImage::AllocatePitched(const int width, const int height,
                                const int num_channels) {
  this->m_width = width;
  this->m_height = height;
  this->m_num_channels = num_channels;

  CUDA_CHECK_CALL(
      cudaMallocPitch(&m_dev_ptr, &m_dev_pitch, width * num_channels, height);)
}

void CudaImage::AllocatePitchedAndUpload(const cv::Mat &img) {
  CHECK(img.type() == CV_8UC1 || img.type() == CV_8UC4)
      << "Only BRGA and Gray Scale is supported.";
  this->m_width = img.cols;
  this->m_height = img.rows;
  if (img.type() == CV_8UC1) {
    this->m_num_channels = 1;
  } else {
    this->m_num_channels = 4;
  }
  Deallocate();
  AllocatePitched(this->m_width, this->m_height, this->m_num_channels);
  CUDA_CHECK_CALL(cudaMemcpy2D(this->m_dev_ptr, this->m_dev_pitch, img.data,
                               img.step, this->m_width * this->m_num_channels,
                               this->m_height, cudaMemcpyHostToDevice);)
}

void CudaImage::ReallocatePitchedAndUpload(const cv::Mat &img) {
  CHECK(img.type() == CV_8UC1 || img.type() == CV_8UC4)
      << "Only BRGA and Gray Scale is supported.";
  if ((this->m_num_channels == 1 && img.type() != CV_8UC4) ||
      (this->m_num_channels == 4 && img.type() != CV_8UC1) ||
      (this->m_width != img.cols) || (this->m_height != img.rows)) {
    Deallocate();
    AllocatePitchedAndUpload(img);
  } else {
    CUDA_CHECK_CALL(cudaMemcpy2D(this->m_dev_ptr, this->m_dev_pitch, img.data,
                                 img.step, this->m_width * this->m_num_channels,
                                 this->m_height, cudaMemcpyHostToDevice);)
  }
}

void CudaImage::Download(cv::Mat &img) {
  if (this->m_num_channels == 4) {
    img = cv::Mat(cv::Size(this->m_width, this->m_height), CV_8UC4);
  } else {
    img = cv::Mat(cv::Size(this->m_width, this->m_height), CV_8UC1);
  }
  CUDA_CHECK_CALL(cudaMemcpy2D(img.data, img.step, this->m_dev_ptr,
                               this->m_dev_pitch,
                               this->m_width * this->m_num_channels,
                               this->m_height, cudaMemcpyDeviceToHost);)
}

void CudaImage::Clear(unsigned char value) {

  dim3 grid_dim(GetNumTiles(this->m_width * this->m_num_channels, TILE_WIDTH),
                GetNumTiles(this->m_width * this->m_num_channels, TILE_HEIGHT));
  dim3 block_dim(GetNumTiles(TILE_WIDTH, TILE_HEIGHT));

  // Call Kernel Func.
  ClearKernel<<<grid_dim, block_dim>>>(*this, value);
}

void CudaImage::Deallocate() {
  CUDA_CHECK_CALL(cudaFree((void *)this->m_dev_ptr);)
  this->m_dev_ptr = nullptr;
}

} // namespace cuda
} // namespace vis