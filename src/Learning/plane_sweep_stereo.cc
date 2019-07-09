
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
#include <plane_sweep_stereo.h>

// System
#include <iostream>
#include <limits>
#include <map>
#include <memory>

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

// Eigen
#include <Eigen/Core>

// Original Cuda
#include <cuda_buffer.h>
#include <cuda_image.h>
#include <cuda_plane_sweep.h>

// Original
#include <camera_matrix.h>
#include <plane_sweep_stereo_intl_buffer.h>

namespace {

bool ComputeHomographyToReferenceImage(const Eigen::Matrix<double, 4, 1>& plane,
                                       const vis::CameraMatrix<double>& ref_cam,
                                       const vis::CameraMatrix<double>& src_cam,
                                       std::vector<float>& H) {
  H.resize(9);
  const Eigen::Matrix<double, 3, 3> ref_K = ref_cam.GetK();
  const Eigen::Matrix<double, 3, 3> src_K = src_cam.GetK();
  const Eigen::Matrix<double, 3, 3> ref_R = ref_cam.GetR();
  const Eigen::Matrix<double, 3, 3> src_R = src_cam.GetR();
  const Eigen::Matrix<double, 3, 1> ref_T = ref_cam.GetT();
  const Eigen::Matrix<double, 3, 1> src_T = src_cam.GetT();
  const Eigen::Matrix<double, 3, 1> unit_n = plane.head(3);
  const Eigen::Matrix<double, 3, 3> rel_R = src_R * ref_R.transpose();
  const Eigen::Matrix<double, 3, 1> rel_T = rel_R * ref_T - src_T;

  Eigen::Matrix<double, 3, 3> Hmat =
      src_K * (rel_R + rel_T * unit_n.transpose() / plane(3)) * ref_K.inverse();

  H[0] = (float)Hmat(0, 0);
  H[1] = (float)Hmat(0, 1);
  H[2] = (float)Hmat(0, 2);
  H[3] = (float)Hmat(1, 0);
  H[4] = (float)Hmat(1, 1);
  H[5] = (float)Hmat(1, 2);
  H[6] = (float)Hmat(2, 0);
  H[7] = (float)Hmat(2, 1);
  H[8] = (float)Hmat(2, 2);

  return true;
}

bool ConvertCvMatToCudaImage(const cv::Mat& img, const double scale,
                             const bool color_enabled,
                             vis::cuda::CudaImage& dev_img) {
  cv::Mat tmp;
  if (scale < 1.0) {
    cv::resize(img, tmp, cv::Size(), scale, scale, CV_INTER_AREA);
  } else if (scale > 1.0) {
    cv::resize(img, tmp, cv::Size(), scale, scale, CV_INTER_LINEAR);
  } else {
    tmp = img.clone();
  }

  if (color_enabled && img.type() == CV_8UC3) {
    cv::cvtColor(tmp, tmp, CV_BGR2BGRA);
  } else if (!color_enabled && img.type() == CV_8UC3) {
    cv::cvtColor(tmp, tmp, CV_BGR2GRAY);
  } else if (!color_enabled && img.type() == CV_8UC1) {
  } else {
    LOG(ERROR) << "Invalid color and channel settings.";
  }
  dev_img.AllocatePitchedAndUpload(tmp);
}

double ComputeLargestBaseline(
    const int ref_cam_idx, std::map<int, vis::mvs::LocalizedCudaImage>& cams) {
  double largest_baseline = 0.0;
  Eigen::Matrix<double, 3, 1> ref_C = cams[ref_cam_idx].first.GetC();

  for (const auto& cam : cams) {
    const Eigen::Matrix<double, 3, 1> C = cam.second.first.GetC();
    double baseline = (C - ref_C).norm();

    if (largest_baseline < baseline) {
      largest_baseline = baseline;
    }
  }
  return largest_baseline;
}

void CreatePlanes(const int ref_cam_idx,
                  std::map<int, vis::mvs::LocalizedCudaImage>& cams,
                  const int num_planes, const double near_z, const double far_z,
                  std::vector<Eigen::Vector4d>& planes) {
  double baseline = ComputeLargestBaseline(ref_cam_idx, cams);

  planes.resize(num_planes);

  const Eigen::Matrix3d& K = cams[ref_cam_idx].first.GetK();
  double focal = (K(0, 0) + K(1, 1)) / 2.0;

  // Disp = Baseline * focal / Z
  double min_disp = baseline * focal / far_z;
  double max_disp = baseline * focal / near_z;

  // Disp step.
  double disp_step = (max_disp - min_disp) / (num_planes - 1);

  // Plane is perpendicular to z axis.
  for (int i = 0; i < num_planes; i++) {
    planes[i].setZero();
    planes[i](2) = -1;
    planes[i](3) = baseline * focal / (max_disp - i * disp_step);
  }
}

}  // namespace

namespace vis {
namespace mvs {

struct PlaneSweepStereoBuffers {
  int next_img_idx;
  std::map<int, LocalizedCudaImage> dev_img_map;
  std::vector<Eigen::Vector4d> hyp_planes;
  SAD sad;
  ZNCC zncc;
  SubPixel subpix;
  Common common;
  BestK bestK;
  BestPlane best_plane;
  DepthMap<float, double> best_depth;
};

PlaneSweepStereo::PlaneSweepStereo()
    : m_params(), m_buffers(new PlaneSweepStereoBuffers()) {}

PlaneSweepStereo::~PlaneSweepStereo() {}

bool PlaneSweepStereo::Initialize(const PlaneSweepStereoParams& params) {
  LOG(INFO) << "PlaneSweepStereo::Initialize()";
  m_params = params;
  vis::cuda::mvs::PlaneSweepInitializeTexture();
  return true;
}

bool PlaneSweepStereo::Run(const int ref_img_idx) {
  LOG(INFO) << "PlaneSweepStereo::Run()";

  // Step 0. Prepare Hypothetical Planes
  CreatePlanes(ref_img_idx, m_buffers->dev_img_map, m_params.num_planes,
               m_params.min_z, m_params.max_z, m_buffers->hyp_planes);

  // Step 1. Buffer Preparaion.
  PrepareBuffer(ref_img_idx);

  // Step 2. Compute Accumulation Scale
  double accum_scale0, accum_scale1;
  CompuateAccumulationScales(accum_scale0, accum_scale1);

  for (unsigned int plane_idx = 0; plane_idx < m_buffers->hyp_planes.size();
       plane_idx++) {
    // Step 3. Compute Cost for THIS plane.
    AccumulateCostForPlane(ref_img_idx, m_buffers->hyp_planes[plane_idx],
                           accum_scale0, accum_scale1);

    // Step 4. Update Best Plane Buffer.
    UpdateBestPlaneBuffer(plane_idx);
  }

  // Step 5. Result Calculation.
  ComputeBestDepth(ref_img_idx);

  return true;
}

bool PlaneSweepStereo::Finalize() {
  LOG(INFO) << "PlaneSweepStereo::Finalize()";
  return true;
}

int PlaneSweepStereo::AddImage(const cv::Mat& img,
                               const CameraMatrix<double>& cam) {
  vis::cuda::CudaImage dev_img;
  ConvertCvMatToCudaImage(img, m_params.img_scale, m_params.enable_color,
                          dev_img);
  int id = m_buffers->next_img_idx;
  m_buffers->next_img_idx++;
  CameraMatrix<double> tmp_cam = cam;
  tmp_cam.ScaleK(m_params.img_scale, m_params.img_scale);
  m_buffers->dev_img_map[id] = std::make_pair(tmp_cam, dev_img);
  return id;
}

bool PlaneSweepStereo::DrawColoredDepthMap(cv::Mat& img) {
  m_buffers->best_depth.ComputeColoredDepthMat(m_params.min_z, m_params.max_z,
                                               img);
}

// Algorithm main routine.
bool PlaneSweepStereo::PrepareBuffer(const int ref_img_idx) {
  cuda::CudaImage& ref_img = m_buffers->dev_img_map[ref_img_idx].second;

  // Common Buffer
  {
    m_buffers->common.dev_cost_accum.ReallocatePitched(ref_img.GetWidth(),
                                                       ref_img.GetHeight());
  }

  // SAD Buffer
  {
    m_buffers->sad.dev_box_filter.ReallocatePitched(ref_img.GetWidth(),
                                                    ref_img.GetHeight());
  }

  // Result Buffer
  {
    m_buffers->best_plane.dev_best_plane_idx.ReallocatePitched(
        ref_img.GetWidth(), ref_img.GetHeight());
    m_buffers->best_plane.dev_best_plane_idx.Clear(0);
    m_buffers->best_plane.dev_best_scores.ReallocatePitched(
        ref_img.GetWidth(), ref_img.GetHeight());
    m_buffers->best_plane.dev_best_scores.Clear(
        std::numeric_limits<float>::max());
  }

  // Output Buffer
  {
    vis::CameraMatrix<double>& ref_cam =
        m_buffers->dev_img_map[ref_img_idx].first;
    m_buffers->best_depth = vis::DepthMap<float, double>(
        ref_img.GetWidth(), ref_img.GetHeight(), ref_cam);
  }

  return true;
}

bool PlaneSweepStereo::CompuateAccumulationScales(double& scale0,
                                                  double& scale1) {
  scale0 = static_cast<double>(1.0 / (m_buffers->dev_img_map.size() - 1));
  scale1 = 0.0;
  return true;
}

bool PlaneSweepStereo::AccumulateCostForPlane(const int ref_img_idx,
                                              const Eigen::Vector4d& hyp_plane,
                                              const double scale0,
                                              const double scale1) {
  const int width = m_buffers->dev_img_map[ref_img_idx].second.GetWidth();
  const int height = m_buffers->dev_img_map[ref_img_idx].second.GetHeight();
  m_buffers->common.dev_cost_accum.Clear(0);

  for (const auto& localized_img : m_buffers->dev_img_map) {
    // Skip if this is reference image.
    if (localized_img.first == ref_img_idx) {
      continue;
    }
    std::vector<float> H;
    ComputeHomographyToReferenceImage(hyp_plane,
                                      m_buffers->dev_img_map[ref_img_idx].first,
                                      localized_img.second.first, H);

    cuda::mvs::PlaneSweepAbsDiffAccum(
        H, scale0, m_buffers->dev_img_map[ref_img_idx].second,
        m_buffers->dev_img_map[localized_img.first].second,
        m_buffers->common.dev_cost_accum);
  }

  cv::Mat tmp1(height, width, CV_32FC1);
  cv::Mat tmp2(height, width, CV_8UC1);
  m_buffers->common.dev_cost_accum.Download((float*)tmp1.data, tmp1.step);
  tmp1.convertTo(tmp2, CV_8UC1);
  cv::resize(tmp2, tmp2, cv::Size(), 1.0, 1.0);
  cv::imshow("After", tmp2);
  cv::waitKey(10);
  FilterAccumulatedCost();

  return true;
}

bool PlaneSweepStereo::FilterAccumulatedCost() {
  vis::cuda::mvs::PlaneSweepBoxFilterCost(
      m_buffers->common.dev_cost_accum, m_buffers->sad.dev_box_filter,
      m_params.match_window_width / 2, m_params.match_window_height / 2);
  vis::cuda::CudaBuffer<float> temp = m_buffers->sad.dev_box_filter;
  m_buffers->sad.dev_box_filter = m_buffers->common.dev_cost_accum;
  m_buffers->common.dev_cost_accum = temp;
  return true;
}

bool PlaneSweepStereo::UpdateBestPlaneBuffer(
    const unsigned int curr_plane_idx) {
  vis::cuda::mvs::PlaneSweepUpdateBestPlane(
      curr_plane_idx, m_buffers->common.dev_cost_accum, m_buffers->best_plane);

  return true;
}

bool PlaneSweepStereo::ComputeBestDepth(const unsigned int ref_img_idx) {
  vis::cuda::CudaImage ref_img = m_buffers->dev_img_map[ref_img_idx].second;

  const Eigen::Matrix3d K_ref =
      m_buffers->dev_img_map[ref_img_idx].first.GetK();
  vis::cuda::mvs::PlaneSweepComputeBestDepth(
      K_ref, m_buffers->best_plane.dev_best_plane_idx, m_buffers->hyp_planes,
      m_buffers->best_depth);

  return true;
}

bool PlaneSweepStereo::ReleaseBuffer() {
  // Common Buffer
  {
    if (m_buffers->common.dev_cost_accum.IsAllocated()) {
      m_buffers->common.dev_cost_accum.Deallocate();
    }
  }

  // SAD Buffer
  {
    if (m_buffers->sad.dev_box_filter.IsAllocated()) {
      m_buffers->sad.dev_box_filter.Deallocate();
    }
  }

  // Result Buffer
  {
    if (m_buffers->best_plane.dev_best_plane_idx.IsAllocated()) {
      m_buffers->best_plane.dev_best_plane_idx.Deallocate();
    }
    if (m_buffers->best_plane.dev_best_scores.IsAllocated()) {
      m_buffers->best_plane.dev_best_scores.Deallocate();
    }
  }
  return true;
}

// Heloper method.
bool PlaneSweepStereo::ClearCostAccumulationBuffer() { return true; }

bool PlaneSweepStereo::DownloadBestCost() { return true; }

bool PlaneSweepStereo::ComputeUniqunessRatio() { return true; }

}  // namespace mvs
}  // namespace vis