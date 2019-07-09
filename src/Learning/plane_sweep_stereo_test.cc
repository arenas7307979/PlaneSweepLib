
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

// Boost
#include <boost/filesystem.hpp>
#include <boost/foreach.hpp>

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

DEFINE_string(data_folder_path, "./data/pinholeCamera/niederdorf1", "");
DEFINE_string(kmat_file_name, "K.txt", "");
DEFINE_string(cammat_file_name, "model-0-cams.txt", "");
DEFINE_double(min_disp_factor, 5.0, "");
DEFINE_double(max_disp_factor, 100.0, "");

namespace {

bool ExtractAbsolutePath(const std::string &path, std::string &directory_path) {
  boost::filesystem::path p{path};
  if (p.is_absolute()) {
    directory_path = path;
  } else {
    boost::filesystem::path abs_dir = boost::filesystem::absolute(p);
    directory_path = std::string(abs_dir.c_str());
  }
  return true;
}

void RaiseAllFilesInDirectoryInternal(
    const std::string &dirpath, std::vector<std::string> &img_path_list,
    std::vector<std::string> &img_filename_list) {
  std::string abs_dir_path;
  ExtractAbsolutePath(dirpath, abs_dir_path);
  namespace fs = boost::filesystem;

  std::cout << abs_dir_path << std::endl;
  const fs::path path(abs_dir_path);

  BOOST_FOREACH (const fs::path &p, std::make_pair(fs::directory_iterator(path),
                                                   fs::directory_iterator())) {
    if (!fs::is_directory(p)) {
      img_filename_list.push_back(p.filename().string());
      img_path_list.push_back(fs::absolute(p).string());
    }
  }

  std::sort(img_filename_list.begin(), img_filename_list.end());
  std::sort(img_path_list.begin(), img_path_list.end());
}

void RaiseAllFilesInDirectory(const std::string &dirpath,
                              std::vector<std::string> &filelist,
                              const std::vector<std::string> &exts) {
  std::vector<std::string> tmp_img_path_list, tmp_file_name_list;
  RaiseAllFilesInDirectoryInternal(dirpath, tmp_img_path_list,
                                   tmp_file_name_list);

  for (size_t i = 0; i < tmp_img_path_list.size(); i++) {
    std::string abs_path = tmp_img_path_list[i];
    std::string filename = tmp_file_name_list[i];
    for (auto ext : exts) {
      if (ext.size() == 0) {
        continue;
      }

      if (abs_path.find(ext) == abs_path.size() - ext.size()) {
        filelist.push_back(abs_path);
        break;
      }
    }
  }
}

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

void CalculateMinMaxRange(std::vector<int> image_idx_to_be_used,
                          std::map<int, vis::CameraMatrix<double>> &cameras,
                          double &min_z, double &max_z) {
  double avg_distance = 0.0;
  int num_distance = 0;

  for (auto idx1 : image_idx_to_be_used) {
    if (cameras.count(idx1) == 1) {
      for (auto idx2 : image_idx_to_be_used) {
        if (idx1 != idx2 && cameras.count(idx2) == 1) {
          Eigen::Vector3d dist = cameras[idx1].GetC() - cameras[idx2].GetC();
          avg_distance += dist.norm();
          num_distance++;
        }
      }
    }
  }

  CHECK_GT(num_distance, 1)
      << "Could not compute average distance, less than two cameras found.";

  avg_distance /= num_distance;
  min_z = FLAGS_min_disp_factor * avg_distance;
  max_z = FLAGS_max_disp_factor * avg_distance;

  LOG(INFO) << "Minimum Z : " << min_z << ", Maximum Z : " << max_z;
}

}  // namespace

int main(int argc, char **argv) {
  // 0. System setup.
  google::ParseCommandLineFlags(&argc, &argv, true);
  FLAGS_alsologtostderr = 1;
  FLAGS_stderrthreshold = google::GLOG_INFO;
  google::InitGoogleLogging(argv[0]);

  // 1. Extract all necessary file paths.
  std::vector<std::string> k_mat_file, cammat_file, image_file_paths;
  {
    RaiseAllFilesInDirectory(FLAGS_data_folder_path, k_mat_file,
                             std::vector<std::string>{FLAGS_kmat_file_name});
    CHECK_EQ(k_mat_file.size(), 1) << "Need 1 K.txt file.";
    RaiseAllFilesInDirectory(FLAGS_data_folder_path, cammat_file,
                             std::vector<std::string>{FLAGS_cammat_file_name});
    CHECK_EQ(cammat_file.size(), 1) << "Need 1 cammat file";
    RaiseAllFilesInDirectory(FLAGS_data_folder_path, image_file_paths,
                             std::vector<std::string>{".JPG"});
    CHECK_GT(image_file_paths.size(), 1) << "Need at least 2 image files.";
  }

  // 2. Load necessary files. (K matrix.)
  Eigen::Matrix3d K = LoadKMatrixFromFile(k_mat_file[0]);
  std::map<int, vis::CameraMatrix<double>> cammap =
      LoadCameraMatrixFromFile(cammat_file[0], K);

  // 3. Parameter settings.
  // Specify image idxes to be used.
  std::vector<int> image_idx_to_be_used{0, 1, 2};
  vis::mvs::PlaneSweepStereoParams params;
  params.img_scale = 0.25;
  CalculateMinMaxRange(image_idx_to_be_used, cammap, params.min_z,
                       params.max_z);

  // 4. Initialize algorithm
  vis::mvs::PlaneSweepStereo pss;
  pss.Initialize(params);
  for (auto idx : image_idx_to_be_used) {
    cv::Mat img = cv::imread(image_file_paths[idx]);
    CHECK(!img.empty()) << "The image " << image_file_paths[idx]
                        << " does not exist.";
    pss.AddImage(img, cammap[idx]);
  }

  // 5. Run algorithm.
  int ref_img_idx = image_idx_to_be_used[image_idx_to_be_used.size() / 2];
  pss.Run(ref_img_idx);

  // 6. Visualize final result
  cv::Mat depth_map;
  pss.DrawColoredDepthMap(depth_map);
  cv::imshow("Depth Map", depth_map);
  cv::waitKey(0);

  // 7. Terminate algorithm.
  pss.Finalize();

  return 0;
}