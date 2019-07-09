
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

// System
#include <map>
#include <memory>
#include <string>
#include <vector>

// Gflags
#include <gflags/gflags.h>

// Glog
#include <glog/logging.h>

// OpenCV
#include <opencv2/core.hpp>

// Original
#include <camera_matrix.h>
#include <cuda_image.h>
#include <hash.h>

namespace vis {
namespace mvs {

struct PlaneSweepStereoParams {
  PlaneSweepStereoParams()
      : enable_color(false),
        num_planes(256),
        img_scale(1.0),
        min_z(0.25),
        max_z(10.0),
        match_window_width(7),
        match_window_height(7) {}

  bool enable_color;
  int num_planes;
  double img_scale;
  double min_z, max_z;
  int match_window_width, match_window_height;
};

struct PlaneSweepStereoBuffers;
using LocalizedCudaImage =
    std::pair<CameraMatrix<double>, vis::cuda::CudaImage>;

class PlaneSweepStereo {
 public:
  explicit PlaneSweepStereo();
  ~PlaneSweepStereo();

  // Deleting copy constructor and = operator.
  PlaneSweepStereo(const PlaneSweepStereo& psl) = delete;
  PlaneSweepStereo& operator=(const PlaneSweepStereo& obj) = delete;

  bool Initialize(const PlaneSweepStereoParams& params);
  bool Run(const int ref_img_idx);
  bool Finalize();
  int AddImage(const cv::Mat& img, const CameraMatrix<double>& cam);
  bool DrawColoredDepthMap(cv::Mat& img);

 private:
  // Algorithm main method.
  bool PrepareBuffer(const int ref_img_idx);
  bool CompuateAccumulationScales(double& scale0, double& scale1);
  bool AccumulateCostForPlane(const int ref_img_idx,
                              const Eigen::Vector4d& hyp_plane,
                              const double scale0, const double scale1);
  bool FilterAccumulatedCost();
  bool UpdateBestPlaneBuffer(const unsigned int curr_plane_idx);
  bool ComputeBestDepth(const unsigned int ref_img_idx);
  bool ReleaseBuffer();

  // Helper method.
  bool ClearCostAccumulationBuffer();
  bool DownloadBestCost();
  bool ComputeUniqunessRatio();

 private:
  PlaneSweepStereoParams m_params;
  std::unique_ptr<PlaneSweepStereoBuffers> m_buffers;
};

}  // namespace mvs
}  // namespace vis