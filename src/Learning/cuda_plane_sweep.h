
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

#ifndef CUDA_PLANE_SWEEP_H_
#define CUDA_PLANE_SWEEP_H_

// System
#include <vector>

// Eigen
#include <Eigen/Core>

// Original Cuda
#include <cuda_buffer.h>
#include <cuda_image.h>

// Original
#include <depth_map.h>

namespace vis {
namespace mvs {
struct BestPlane;
}
}  // namespace vis

namespace vis {
namespace cuda {
namespace mvs {

void PlaneSweepInitializeTexture();

void PlaneSweepAbsDiffAccum(const std::vector<float> &H,
                            const float accum_scale0, CudaImage &dev_ref_img,
                            CudaImage &dev_src_img,
                            CudaBuffer<float> &dev_cost_buff);

void PlaneSweepBoxFilterCost(
    vis::cuda::CudaBuffer<float> &dev_cost_buff,
    vis::cuda::CudaBuffer<float> &dev_filtered_cost_buff, int radius_x,
    int radius_y);

void PlaneSweepUpdateBestPlane(const int curr_plane_idx,
                               const CudaBuffer<float> &dev_cost_buff,
                               vis::mvs::BestPlane &dev_best_plane_buff);

void PlaneSweepComputeBestDepth(const Eigen::Matrix3d &k_ref,
                                vis::cuda::CudaBuffer<int> dev_best_plane_idxs,
                                std::vector<Eigen::Vector4d> &planes,
                                vis::DepthMap<float, double> &best_depth);

}  // namespace mvs
}  // namespace cuda
}  // namespace vis

#endif  // CUDA_PLANE_SWEEP_H_
