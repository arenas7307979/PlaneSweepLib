
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

// Cuda
#include <cuda_runtime.h>

// Glog
#include <glog/logging.h>

// Eigen
#include <Eigen/Core>
#include <Eigen/Dense>

// Original Cuda
#include <cuda_buffer.h>
#include <cuda_common.h>
#include <cuda_image.h>
#include <cuda_plane_sweep_common.h>

// Original
#include <depth_map.h>
#include <plane_sweep_stereo_intl_buffer.h>

namespace vis {
namespace cuda {
namespace mvs {

// Cuda Related Buffer

bool g_initialize_done;

// Channel Format
cudaChannelFormatDesc g_channel_texture;
cudaChannelFormatDesc g_channel_cost_texture;

// Texture
texture<uchar1, cudaTextureType2D, cudaReadModeNormalizedFloat> g_img_texture;
texture<unsigned char, cudaTextureType2D> g_img_texture_wo_interpolation;
texture<float, cudaTextureType2D> g_cost_texture;

// Cuda Related Buffer

void PlaneSweepInitializeTexture() {

  if (g_initialize_done) {
    return;
  }

  // Channel initialiation.
  {
    g_channel_texture =
        cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);
    g_channel_cost_texture =
        cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
  }

  // Gray scale texture initialization.
  {
    g_img_texture.addressMode[0] = cudaAddressModeWrap;
    g_img_texture.addressMode[1] = cudaAddressModeWrap;
    g_img_texture.filterMode = cudaFilterModeLinear;
    g_img_texture.normalized = true;
  }

  // Gray scale texture wo interpolation initialization.
  {
    g_img_texture_wo_interpolation.addressMode[0] = cudaAddressModeWrap;
    g_img_texture_wo_interpolation.addressMode[1] = cudaAddressModeWrap;
    g_img_texture_wo_interpolation.filterMode = cudaFilterModePoint;
    g_img_texture_wo_interpolation.normalized = false;
  }

  // Cost texture
  {
    g_cost_texture.addressMode[0] = cudaAddressModeClamp;
    g_cost_texture.addressMode[1] = cudaAddressModeClamp;
    g_cost_texture.filterMode = cudaFilterModePoint;
    g_cost_texture.normalized = false;
  }

  g_initialize_done = true;
}

__global__ void PlaneSweepAbsDiffAccumGrayScaleKernel(
    const float h11, const float h12, const float h13, const float h21,
    const float h22, const float h23, const float h31, const float h32,
    const float h33, const float accum_scale0, const CudaImage dev_ref_img,
    const CudaImage dev_src_img, CudaBuffer<float> dev_cost_buff) {
  // Index
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

  const int width = dev_ref_img.GetWidth();
  const int height = dev_ref_img.GetHeight();

  if (x < width && y < height) {
    // Apply homography from ref -> src
    float xw, yw;
    WarpCoordinateViaHomography(x, y, h11, h12, h13, h21, h22, h23, h31, h32,
                                h33, &xw, &yw);
    const float u = (xw + 0.5f) / (float)dev_src_img.GetWidth();
    const float v = (yw + 0.5f) / (float)dev_src_img.GetHeight();

    // Pixel value at corresp source image.
    const float1 pix = tex2D(g_img_texture, u, v);
    float diff =
        fabs((float)(dev_ref_img(x, y) - __saturatef(fabs(pix.x)) * 255));

    // Register cost for this pix.
    dev_cost_buff(x, y) += accum_scale0 * diff;
  }
}

void PlaneSweepAbsDiffAccum(const std::vector<float> &H,
                            const float accum_scale0, CudaImage &dev_ref_img,
                            CudaImage &dev_src_img,
                            CudaBuffer<float> &dev_cost_buff) {

  CHECK(H.size() == 9) << "Invalid Homography Matrix.";

  dim3 grid_dim(
      GetNumTiles(dev_ref_img.GetWidth(), mvs::PLANE_SWEEP_BLOCK_WIDTH),
      GetNumTiles(dev_ref_img.GetHeight(), mvs::PLANE_SWEEP_BLOCK_HEIGHT));
  dim3 block_dim(mvs::PLANE_SWEEP_BLOCK_WIDTH, mvs::PLANE_SWEEP_BLOCK_HEIGHT);

  // Bind src image to texture.
  CUDA_CHECK_CALL(cudaBindTexture2D(0, g_img_texture, dev_src_img.GetDevAddr(),
                                    g_channel_texture, dev_src_img.GetWidth(),
                                    dev_src_img.GetHeight(),
                                    dev_src_img.GetPitch());)

  float h11 = H[0], h12 = H[1], h13 = H[2], h21 = H[3], h22 = H[4], h23 = H[5],
        h31 = H[6], h32 = H[7], h33 = H[8];
  PlaneSweepAbsDiffAccumGrayScaleKernel<<<grid_dim, block_dim>>>(
      h11, h12, h13, h21, h22, h23, h31, h32, h33, accum_scale0, dev_ref_img,
      dev_src_img, dev_cost_buff);

  // Unbind src image.
  CUDA_CHECK_CALL(cudaUnbindTexture(g_img_texture);)
}

__global__ void PlaneSweepBoxFilterCostKernel(
    vis::cuda::CudaBuffer<float> dev_filtered_cost_buff, int radius_x,
    int radius_y) {

  // Dynamically allocated shared memory.
  extern __shared__ float col_sum[];

  const int x = blockIdx.x * PLANE_SWEEP_BOX_FILTER_NUM_THREADS + threadIdx.x -
                blockIdx.x * 2 * radius_x;
  const int y = blockIdx.y * PLANE_SWEEP_BOX_FILTER_ROWS_PER_THREAD;
  const int width = dev_filtered_cost_buff.GetWidth();
  const int height = dev_filtered_cost_buff.GetHeight();

  if (x < (width + 2 * radius_x) && y < height) {
    col_sum[threadIdx.x] = 0.0f;

    // Sum up window.
    {
      int x_cost = x - radius_x;
      int y_cost = y - radius_y;
      // Sum up row direction.
      for (int i = 0; i <= 2 * radius_y; i++) {
        col_sum[threadIdx.x] += tex2D(g_cost_texture, x_cost, y_cost);
        y_cost++;
      }
      __syncthreads();

      if (threadIdx.x + 2 * radius_x < PLANE_SWEEP_BOX_FILTER_NUM_THREADS &&
          x < width && y < height) {
        float sum = 0;
        // Sum up column direction.
        for (int i = 0; i <= 2 * radius_x; i++) {
          sum = sum + col_sum[i + threadIdx.x];
        }
        dev_filtered_cost_buff(x, y) = sum;
      }
      __syncthreads();
    }

    {
      int x_cost = x - radius_x;
      int y_cost = y - radius_y;
      for (int row = 1;
           row < PLANE_SWEEP_BOX_FILTER_ROWS_PER_THREAD && (y + row < height);
           row++) {
        col_sum[threadIdx.x] =
            col_sum[threadIdx.x] - tex2D(g_cost_texture, x_cost, y_cost);
        col_sum[threadIdx.x] =
            col_sum[threadIdx.x] +
            tex2D(g_cost_texture, x_cost, y_cost + 2 * radius_y + 1);
        y_cost++;
        __syncthreads();

        if (threadIdx.x + 2 * radius_x < PLANE_SWEEP_BOX_FILTER_NUM_THREADS &&
            x < width) {
          float sum = 0;
          for (int i = 0; i <= 2 * radius_x; i++) {
            sum = sum + col_sum[i + threadIdx.x];
          }
          dev_filtered_cost_buff(x, y + row) = sum;
        }
        __syncthreads();
      }
    }
  }
}

void PlaneSweepBoxFilterCost(
    vis::cuda::CudaBuffer<float> &dev_cost_buff,
    vis::cuda::CudaBuffer<float> &dev_filtered_cost_buff, int radius_x,
    int radius_y) {

  // Bind to 2d texture.
  CUDA_CHECK_CALL(
      cudaBindTexture2D(0, g_cost_texture, dev_cost_buff.GetDevAddr(),
                        g_channel_cost_texture, dev_cost_buff.GetWidth(),
                        dev_cost_buff.GetHeight(), dev_cost_buff.GetPitch());)

  const int shared_mem_size =
      PLANE_SWEEP_BOX_FILTER_NUM_THREADS * sizeof(float);

  dim3 grid_num(GetNumTiles(dev_filtered_cost_buff.GetWidth(),
                            PLANE_SWEEP_BOX_FILTER_NUM_THREADS - 2 * radius_x),
                GetNumTiles(dev_filtered_cost_buff.GetHeight(),
                            PLANE_SWEEP_BOX_FILTER_ROWS_PER_THREAD));
  dim3 block_num(PLANE_SWEEP_BOX_FILTER_NUM_THREADS, 1);

  // Run kernel
  PlaneSweepBoxFilterCostKernel<<<grid_num, block_num, shared_mem_size>>>(
      dev_filtered_cost_buff, radius_x, radius_y);
  CUDA_CHECK_ERROR

  // Unbind texture
  CUDA_CHECK_CALL(cudaUnbindTexture(g_cost_texture);)
}

__global__ void
PlaneSweepUpdateBestPlaneKernel(const int curr_plane_idx,
                                const CudaBuffer<float> dev_cost_buff,
                                vis::mvs::BestPlane dev_best_plane_buff) {
  //
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < dev_cost_buff.GetWidth() && y < dev_cost_buff.GetHeight()) {
    if (dev_cost_buff(x, y) < dev_best_plane_buff.dev_best_scores(x, y)) {
      dev_best_plane_buff.dev_best_plane_idx(x, y) = curr_plane_idx;
      dev_best_plane_buff.dev_best_scores(x, y) = dev_cost_buff(x, y);
    }
  }
}

void PlaneSweepUpdateBestPlane(const int curr_plane_idx,
                               const CudaBuffer<float> &dev_cost_buff,
                               vis::mvs::BestPlane &dev_best_plane_buff) {

  dim3 grid_dim(
      GetNumTiles(dev_cost_buff.GetWidth(), PLANE_SWEEP_BLOCK_WIDTH),
      GetNumTiles(dev_cost_buff.GetHeight(), PLANE_SWEEP_BLOCK_HEIGHT));
  dim3 block_dim(PLANE_SWEEP_BLOCK_WIDTH, PLANE_SWEEP_BLOCK_HEIGHT);

  PlaneSweepUpdateBestPlaneKernel<<<grid_dim, block_dim>>>(
      curr_plane_idx, dev_cost_buff, dev_best_plane_buff);
  CUDA_CHECK_ERROR
}

__global__ void PlaneSweepComputeBestDepthKernel(
    vis::cuda::CudaBuffer<int> dev_best_plane_idxs, float *dev_plane_addr,
    size_t dev_plane_pitch, float *dev_best_depth_addr,
    size_t dev_best_depth_pitch, float3 K_ref_inv_col1, float3 K_ref_inv_col2,
    float3 K_ref_inv_col3) {

  const int width = dev_best_plane_idxs.GetWidth();
  const int height = dev_best_plane_idxs.GetHeight();
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

  // TODO: Modify here for better readability.
  // Look at each pix in best_plane_idx and calculate depth.
  if (x < width && y < height) {
    const int best_plane_idx = dev_best_plane_idxs(x, y);
    float3 plane_normal;
    plane_normal.x = dev_plane_addr[best_plane_idx];
    plane_normal.y =
        *((float *)((char *)dev_plane_addr + dev_plane_pitch) + best_plane_idx);
    plane_normal.z = *((float *)((char *)dev_plane_addr + 2 * dev_plane_pitch) +
                       best_plane_idx);
    const float planeD =
        *((float *)((char *)dev_plane_addr + 3 * dev_plane_pitch) +
          best_plane_idx);

    float3 K_ref_inv_Tn;
    K_ref_inv_Tn.x = K_ref_inv_col1.x * plane_normal.x +
                     K_ref_inv_col1.y * plane_normal.y +
                     K_ref_inv_col1.z * plane_normal.z;

    K_ref_inv_Tn.y = K_ref_inv_col2.x * plane_normal.x +
                     K_ref_inv_col2.y * plane_normal.y +
                     K_ref_inv_col2.z * plane_normal.z;

    K_ref_inv_Tn.z = K_ref_inv_col3.x * plane_normal.x +
                     K_ref_inv_col3.y * plane_normal.y +
                     K_ref_inv_col3.z * plane_normal.z;

    float3 pos;
    pos.x = x;
    pos.y = y;
    pos.z = 1;

    const float denom = K_ref_inv_Tn.x * pos.x + K_ref_inv_Tn.y * pos.y +
                        K_ref_inv_Tn.z * pos.z;

    *((float *)((char *)dev_best_depth_addr + y * dev_best_depth_pitch) + x) =
        -planeD / denom;
  }
}

void PlaneSweepComputeBestDepth(const Eigen::Matrix3d &K_ref,
                                vis::cuda::CudaBuffer<int> dev_best_plane_idxs,
                                std::vector<Eigen::Vector4d> &planes,
                                vis::DepthMap<float, double> &best_depth) {

  const int width = dev_best_plane_idxs.GetWidth();
  const int height = dev_best_plane_idxs.GetHeight();
  const int num_planes = planes.size();

  // Make consecutive array of plane coeffs.
  std::vector<float> plane_coeffs;
  plane_coeffs.reserve(num_planes * 4);

  // TODO: Modify for better readability.
  for (int j = 0; j < 4; j++) {
    for (int i = 0; i < num_planes; i++) {
      plane_coeffs.push_back(planes[i](j));
    }
  }

  // Allocate device memory for planes to upload to GPU.
  float *dev_plane_addr;
  size_t dev_plane_pitch;
  CUDA_CHECK_CALL(cudaMallocPitch(&dev_plane_addr, &dev_plane_pitch,
                                  sizeof(float) * num_planes, 4);)
  CUDA_CHECK_CALL(cudaMemcpy2D(dev_plane_addr, dev_plane_pitch,
                               &plane_coeffs[0], sizeof(float) * num_planes,
                               sizeof(float) * num_planes, 4,
                               cudaMemcpyHostToDevice);)

  Eigen::Matrix3d K_inv = K_ref.inverse();
  float3 K_ref_inv_col1;
  K_ref_inv_col1.x = K_inv(0, 0);
  K_ref_inv_col1.y = K_inv(1, 0);
  K_ref_inv_col1.z = K_inv(2, 0);
  float3 K_ref_inv_col2;
  K_ref_inv_col2.x = K_inv(0, 1);
  K_ref_inv_col2.y = K_inv(1, 1);
  K_ref_inv_col2.z = K_inv(2, 1);
  float3 K_ref_inv_col3;
  K_ref_inv_col3.x = K_inv(0, 2);
  K_ref_inv_col3.y = K_inv(1, 2);
  K_ref_inv_col3.z = K_inv(2, 2);

  // Allocate device memory for best depth to GPU.
  float *dev_best_depth_addr;
  size_t dev_best_depth_pitch;
  CUDA_CHECK_CALL(cudaMallocPitch(&dev_best_depth_addr, &dev_best_depth_pitch,
                                  sizeof(float) * width, height);)

  // Compute best depth
  dim3 grid_dim(GetNumTiles(width, PLANE_SWEEP_BLOCK_WIDTH),
                GetNumTiles(height, PLANE_SWEEP_BLOCK_HEIGHT));
  dim3 block_dim(PLANE_SWEEP_BLOCK_WIDTH, PLANE_SWEEP_BLOCK_HEIGHT);
  PlaneSweepComputeBestDepthKernel<<<grid_dim, block_dim>>>(
      dev_best_plane_idxs, dev_plane_addr, dev_plane_pitch, dev_best_depth_addr,
      dev_best_depth_pitch, K_ref_inv_col1, K_ref_inv_col2, K_ref_inv_col3);
  CUDA_CHECK_ERROR

  // Download result to device.
  CUDA_CHECK_CALL(cudaMemcpy2D(best_depth.GetDataPtr(),
                               best_depth.GetWidth() * sizeof(float),
                               dev_best_depth_addr, dev_best_depth_pitch,
                               best_depth.GetWidth() * sizeof(float),
                               best_depth.GetHeight(), cudaMemcpyDeviceToHost);)

  CUDA_CHECK_CALL(cudaFree(dev_plane_addr);)
  CUDA_CHECK_CALL(cudaFree(dev_best_depth_addr);)
}
}
}
}