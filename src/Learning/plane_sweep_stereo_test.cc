
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
#include <fstream>
#include <iostream>
#include <string>

// Gflags
#include <gflags/gflags.h>

// Glog
#include <glog/logging.h>

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

// Original
#include <cuda_image.h>
#include <plane_sweep_stereo.h>

DEFINE_string(image_path,
              "/home/koichi/workspace/3rdParty/plane_sweep_lib/PlaneSweepLib/"
              "data/pinholeCamera/niederdorf1/DSC00533.JPG",
              "");

std::string kmat_file_path =
    "/home/koichi/workspace/3rdParty/plane_sweep_lib/PlaneSweepLib/data/"
    "pinholeCamera/niederdorf1/K.txt";

std::string cammat_file_path =
    "/home/koichi/workspace/3rdParty/plane_sweep_lib/PlaneSweepLib/data/"
    "pinholeCamera/niederdorf1/model-0-cams.txt";

std::vector<std::string> test_img_path{
    "/home/koichi/workspace/3rdParty/plane_sweep_lib/PlaneSweepLib/data/"
    "pinholeCamera/niederdorf1/DSC00533.JPG",
    "/home/koichi/workspace/3rdParty/plane_sweep_lib/PlaneSweepLib/data/"
    "pinholeCamera/niederdorf1/DSC00534.JPG",
    "/home/koichi/workspace/3rdParty/plane_sweep_lib/PlaneSweepLib/data/"
    "pinholeCamera/niederdorf1/DSC00535.JPG"};

namespace {

Eigen::Matrix3d LoadKMatrixFromFile(const std::string &filepath) {
  Eigen::Matrix3d K = Eigen::Matrix3d::Identity();
  {
    // try to load k matrix
    std::ifstream kMatrixStr;
    kMatrixStr.open(filepath.c_str());

    CHECK(kMatrixStr.is_open()) << "Error opening K matrix file.";

    {
      kMatrixStr >> K(0, 0);
      kMatrixStr >> K(0, 1);
      kMatrixStr >> K(0, 2);
      kMatrixStr >> K(1, 1);
      kMatrixStr >> K(1, 2);
    }
  }
  return K;
}

std::map<int, vis::CameraMatrix<double>> LoadCameraMatrixFromFile(
    const std::string &filepath, const Eigen::Matrix3d &K) {
  std::map<int, vis::CameraMatrix<double>> cameras;
  {
    // Poses are computed with Christopher Zach's publicly available V3D Soft
    std::ifstream posesStr;
    posesStr.open(filepath.c_str());

    CHECK(posesStr.is_open()) << "Could not load camera poses";
    int numCameras;
    posesStr >> numCameras;
    for (int c = 0; c < numCameras; c++) {
      int id;
      posesStr >> id;

      Eigen::Matrix<double, 3, 3> R;
      Eigen::Matrix<double, 3, 1> T;

      for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
          posesStr >> R(i, j);
        }
        posesStr >> T(i);
      }
      cameras[id].SetKRT(K, R, T);
    }
  }
  return cameras;
}
}  // namespace

int main(int argc, char **argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  FLAGS_alsologtostderr = 1;
  FLAGS_stderrthreshold = google::GLOG_INFO;
  google::InitGoogleLogging(argv[0]);
  LOG(INFO) << "plane_sweep_test.cc Starts.";

  Eigen::Matrix3d K = LoadKMatrixFromFile(kmat_file_path);

  std::map<int, vis::CameraMatrix<double>> cammap =
      LoadCameraMatrixFromFile(cammat_file_path, K);

  vis::mvs::PlaneSweepStereoParams params;
  params.img_scale = 0.25;
  params.min_z = 0.109379;
  params.max_z = 4.37517;

  vis::mvs::PlaneSweepStereo pss;
  pss.Initialize(params);

  for (int i = 0; i < 3; i++) {
    cv::Mat img = cv::imread(test_img_path[i]);
    pss.AddImage(img, cammap[i]);
  }

  int ref_img_idx = 1;
  pss.Run(ref_img_idx);

  cv::Mat depth_map;
  pss.DrawColoredDepthMap(depth_map);
  cv::imshow("Depth Map", depth_map);
  cv::waitKey(0);

  pss.Finalize();

  return 0;
}