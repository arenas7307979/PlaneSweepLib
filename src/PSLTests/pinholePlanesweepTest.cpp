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

// System
#include <fstream>
#include <iostream>

// Glog
#include <glog/logging.h>

// Boost
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

// Eigen
#include <Eigen/Dense>

// OpenCV
#include <opencv2/highgui/highgui.hpp>

// PSL
#include <psl_base/cameraMatrix.h>
#include <psl_base/exception.h>
#include <psl_stereo/cudaPlaneSweep.h>

namespace {

void MakeOutputFolder(std::string folderName) {
  if (!boost::filesystem::exists(folderName)) {
    if (!boost::filesystem::create_directory(folderName)) {
      std::stringstream errorMsg;
      errorMsg << "Could not create output directory: " << folderName;
      PSL_THROW_EXCEPTION(errorMsg.str().c_str());
    }
  }
}

Eigen::Matrix3d LoadKMatrixFromFile(const std::string &filepath) {

  Eigen::Matrix3d K = Eigen::Matrix3d::Identity();
  {
    // try to load k matrix
    std::ifstream kMatrixStr;
    kMatrixStr.open(filepath.c_str());

    if (!kMatrixStr.is_open()) {
      PSL_THROW_EXCEPTION("Error opening K matrix file.")
    }

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

std::map<int, PSL::CameraMatrix<double>>
LoadCameraMatrixFromFile(const std::string &filepath,
                         const Eigen::Matrix3d &K) {

  std::map<int, PSL::CameraMatrix<double>> cameras;
  {
    // Poses are computed with Christopher Zach's publicly available V3D Soft
    std::ifstream posesStr;
    posesStr.open(filepath.c_str());

    if (!posesStr.is_open()) {
      PSL_THROW_EXCEPTION("Could not load camera poses");
    }

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

      cameras[id].setKRT(K, R, T);
    }
  }
  return cameras;
}

void LoadImageFilePaths(const std::string &filepath,
                        const std::string &data_dir,
                        std::vector<std::string> &image_file_names,
                        std::vector<std::string> &image_file_paths) {

  {
    std::string imageListFile = filepath;
    std::ifstream imagesStream;
    imagesStream.open(imageListFile.c_str());
    CHECK(imagesStream.is_open()) << "Could not load images list file";

    {
      std::string imageFileName;
      while (imagesStream >> imageFileName) {
        image_file_names.push_back(imageFileName);
        image_file_paths.push_back(data_dir + "/" + imageFileName);
      }
    }

    CHECK_EQ(image_file_names.size(), 25)
        << "The dataset does not contain 25 images";
  }
}

void CalculateMinMaxRange(const int image_num,
                          std::map<int, PSL::CameraMatrix<double>> &cameras,
                          float &minZ, float &maxZ) {
  {
    // Each of the datasets contains 25 cameras taken in 5 rows
    // The reconstructions are not metric. In order to have an idea about the
    // scale
    // everything is defined with respect to the average distance between the
    // cameras.
    double avgDistance = 0;
    int numDistances = 0;

    for (unsigned int i = 0; i < image_num - 1; i++) {
      if (cameras.count(i) == 1) {
        for (unsigned int j = i + 1; j < image_num; j++) {
          if (cameras.count(j) == 1) {
            Eigen::Vector3d distance = cameras[i].getC() - cameras[j].getC();

            avgDistance += distance.norm();
            numDistances++;
          }
        }
      }
    }
    CHECK_GT(numDistances, 1)
        << "Could not compute average distance, less than two cameras found";

    avgDistance /= numDistances;
    std::cout << "Cameras have an average distance of " << avgDistance << "."
              << std::endl;

    minZ = (float)(2.5f * avgDistance);
    maxZ = (float)(100.0f * avgDistance);
    std::cout << "  Z range :  " << minZ << "  - " << maxZ << std::endl;
  }
}

PSL::CudaPlaneSweep
CreateCudaPlaneSweepWithdDefaultConfigration(const float minZ,
                                             const float maxZ) {

  PSL::CudaPlaneSweep cPS;
  // Scale the images down to 0.25 times the original side length
  cPS.setScale(0.25);
  cPS.setZRange(minZ, maxZ);
  cPS.setMatchWindowSize(7, 7);
  cPS.setNumPlanes(256);
  cPS.setOcclusionMode(PSL::PLANE_SWEEP_OCCLUSION_NONE);
  cPS.setPlaneGenerationMode(PSL::PLANE_SWEEP_PLANEMODE_UNIFORM_DISPARITY);
  cPS.setMatchingCosts(PSL::PLANE_SWEEP_SAD);
  cPS.setSubPixelInterpolationMode(PSL::PLANE_SWEEP_SUB_PIXEL_INTERP_INVERSE);
  cPS.enableOutputBestDepth();
  cPS.enableColorMatching(false);
  cPS.enableOutputBestCosts(false);
  cPS.enableOuputUniquenessRatio(false);
  cPS.enableOutputCostVolume(false);
  cPS.enableSubPixel();

  return cPS;
}

int UploadImageToDevice(const std::vector<std::string> &image_file_paths,
                        const int image_num_to_upload,
                        std::map<int, PSL::CameraMatrix<double>> &cameras,
                        PSL::CudaPlaneSweep &cps) {
  // now we upload the images
  int ref_img_id_in_cps = -1;
  int ref_idx = image_num_to_upload / 2;
  for (int img_idx = 0; img_idx < image_num_to_upload; img_idx++) {
    // load the image from disk
    std::string imageFileName = image_file_paths[img_idx];
    cv::Mat image = cv::imread(imageFileName);
    CHECK(!image.empty()) << "Failed to load image";
    CHECK(cameras.count(img_idx) == 1) << "Camera for image was not loaded, "
                                          "something is wrong with the dataset";

    int img_id_in_cps = cps.addImage(image, cameras[img_idx]);
    if (img_idx == ref_idx) {
      ref_img_id_in_cps = img_id_in_cps;
    }
  }
  return ref_img_id_in_cps;
}

std::string GetEnumString(PSL::PlaneSweepMatchingCosts matching_cost) {
  std::string label = "";
  switch (matching_cost) {
  case PSL::PlaneSweepMatchingCosts::PLANE_SWEEP_SAD:
    label = "PLANE_SWEEP_SAD";
    break;
  case PSL::PlaneSweepMatchingCosts::PLANE_SWEEP_ZNCC:
    label = "PLANE_SWEEP_ZNCC";
    break;
  }
  return label;
}

std::string GetEnumString(PSL::PlaneSweepOcclusionMode occlusion_mode) {
  std::string label = "";
  switch (occlusion_mode) {
  case PSL::PlaneSweepOcclusionMode::PLANE_SWEEP_OCCLUSION_NONE:
    label = "PLANE_SWEEP_OCCLUSION_NONE";
    break;
  case PSL::PlaneSweepOcclusionMode::PLANE_SWEEP_OCCLUSION_REF_SPLIT:
    label = "PLANE_SWEEP_OCCLUSION_REF_SPLIT";
    break;
  case PSL::PlaneSweepOcclusionMode::PLANE_SWEEP_OCCLUSION_BEST_K:
    label = "PLANE_SWEEP_OCCLUSION_BEST_K";
    break;
  }
  return label;
}

void PinholePlaneSweepTest(const PSL::PlaneSweepMatchingCosts matching_cost,
                           const PSL::PlaneSweepOcclusionMode occlusion_mode,
                           const int occlusion_best_k,
                           const int reference_img_id, const float min_z,
                           const float max_z, PSL::CudaPlaneSweep &cPS,
                           int window_time) {

  cPS.setMatchingCosts(matching_cost);
  cPS.setOcclusionMode(occlusion_mode);
  cPS.setOcclusionBestK(occlusion_best_k);
  cPS.process(reference_img_id);
  PSL::DepthMap<float, double> dM;
  dM = cPS.getBestDepth();
  cv::Mat refImage = cPS.downloadImage(reference_img_id);

  /*
MakeOutputFolder("pinholeTestResults/grayscaleZNCC/BestK/");
cv::imwrite("pinholeTestResults/grayscaleZNCC/BestK/refImg.png", refImage);
dM.saveInvDepthAsColorImage(
    "pinholeTestResults/grayscaleZNCC/BestK/invDepthCol.png", min_z,
    reference_img_id);
    */

  std::string title =
      GetEnumString(matching_cost) + " - " + GetEnumString(occlusion_mode);
  cv::Mat inv_depth_mat;
  // dM.ComputeDepthMat(min_z, max_z, inv_depth_mat);
  dM.ComputeColoredDepthMat(min_z, max_z, inv_depth_mat);
  cv::imshow("Reference Image : " + title, refImage);
  cv::imshow("Depth Map : " + title, inv_depth_mat);
  cv::waitKey(window_time);
}
}

int OriginalTest(int argc, char *argv[]) {

  std::string dataFolder;
  boost::program_options::options_description desc("Allowed options");
  desc.add_options()("help", "Produce help message")(
      "dataFolder", boost::program_options::value<std::string>(&dataFolder)
                        ->default_value("DataPinholeCamera/niederdorf2"),
      "One of the data folders for pinhole planesweep provided with the plane "
      "sweep code.");

  boost::program_options::variables_map vm;
  boost::program_options::store(
      boost::program_options::command_line_parser(argc, argv)
          .options(desc)
          .run(),
      vm);
  boost::program_options::notify(vm);

  if (vm.count("help")) {
    std::cout << desc << std::endl;
    return 1;
  }

  // Load camera matrix from file.
  Eigen::Matrix3d K = LoadKMatrixFromFile(dataFolder + "/K.txt");

  // Load camera poses from files.
  std::map<int, PSL::CameraMatrix<double>> cameras =
      LoadCameraMatrixFromFile(dataFolder + "/model-0-cams.txt", K);

  // Load image filenames.
  std::vector<std::string> imageFileNames, image_file_paths;
  LoadImageFilePaths(dataFolder + "/images.txt", dataFolder, imageFileNames,
                     image_file_paths);

  // Compute average distance.
  float minZ, maxZ;
  CalculateMinMaxRange(imageFileNames.size(), cameras, minZ, maxZ);

  //
  MakeOutputFolder("pinholeTestResults");

  // Color
  {
    PSL::CudaPlaneSweep cPS =
        CreateCudaPlaneSweepWithdDefaultConfigration(minZ, maxZ);
    cPS.enableColorMatching(true);

    // Use 5 color images.
    {
      int ref_img_id = UploadImageToDevice(image_file_paths, 5, cameras, cPS);

      PinholePlaneSweepTest(
          PSL::PlaneSweepMatchingCosts::PLANE_SWEEP_SAD,
          PSL::PlaneSweepOcclusionMode::PLANE_SWEEP_OCCLUSION_NONE, 0,
          ref_img_id, minZ, maxZ, cPS, 0);

      PinholePlaneSweepTest(
          PSL::PlaneSweepMatchingCosts::PLANE_SWEEP_SAD,
          PSL::PlaneSweepOcclusionMode::PLANE_SWEEP_OCCLUSION_REF_SPLIT, 0,
          ref_img_id, minZ, maxZ, cPS, 0);
    }

    // Use 25 color images.
    {
      int ref_img_id = UploadImageToDevice(image_file_paths, 25, cameras, cPS);

      PinholePlaneSweepTest(
          PSL::PlaneSweepMatchingCosts::PLANE_SWEEP_SAD,
          PSL::PlaneSweepOcclusionMode::PLANE_SWEEP_OCCLUSION_BEST_K, 5,
          ref_img_id, minZ, maxZ, cPS, 0);
    }
  }

  // Gray
  // First tests compute a depth map for the middle image of the first row
  {

    PSL::CudaPlaneSweep cPS =
        CreateCudaPlaneSweepWithdDefaultConfigration(minZ, maxZ);
    cPS.enableColorMatching(false);

    // Use 5 gray images.
    {
      int ref_img_id = UploadImageToDevice(image_file_paths, 5, cameras, cPS);

      PinholePlaneSweepTest(
          PSL::PlaneSweepMatchingCosts::PLANE_SWEEP_SAD,
          PSL::PlaneSweepOcclusionMode::PLANE_SWEEP_OCCLUSION_NONE, 5,
          ref_img_id, minZ, maxZ, cPS, 0);

      PinholePlaneSweepTest(
          PSL::PlaneSweepMatchingCosts::PLANE_SWEEP_ZNCC,
          PSL::PlaneSweepOcclusionMode::PLANE_SWEEP_OCCLUSION_NONE, 5,
          ref_img_id, minZ, maxZ, cPS, 0);

      PinholePlaneSweepTest(
          PSL::PlaneSweepMatchingCosts::PLANE_SWEEP_SAD,
          PSL::PlaneSweepOcclusionMode::PLANE_SWEEP_OCCLUSION_REF_SPLIT, 0,
          ref_img_id, minZ, maxZ, cPS, 0);

      PinholePlaneSweepTest(
          PSL::PlaneSweepMatchingCosts::PLANE_SWEEP_ZNCC,
          PSL::PlaneSweepOcclusionMode::PLANE_SWEEP_OCCLUSION_REF_SPLIT, 0,
          ref_img_id, minZ, maxZ, cPS, 0);
    }

    // Use 25 gray images.
    {

      int ref_img_id = UploadImageToDevice(image_file_paths, 25, cameras, cPS);

      PinholePlaneSweepTest(
          PSL::PlaneSweepMatchingCosts::PLANE_SWEEP_SAD,
          PSL::PlaneSweepOcclusionMode::PLANE_SWEEP_OCCLUSION_BEST_K, 5,
          ref_img_id, minZ, maxZ, cPS, 0);

      PinholePlaneSweepTest(
          PSL::PlaneSweepMatchingCosts::PLANE_SWEEP_ZNCC,
          PSL::PlaneSweepOcclusionMode::PLANE_SWEEP_OCCLUSION_BEST_K, 5,
          ref_img_id, minZ, maxZ, cPS, 0);
    }
  }

  return 0;
}

int DebugTest(int argc, char *argv[]) {

  std::string dataFolder;
  boost::program_options::options_description desc("Allowed options");
  desc.add_options()("help", "Produce help message")(
      "dataFolder", boost::program_options::value<std::string>(&dataFolder)
                        ->default_value("DataPinholeCamera/niederdorf2"),
      "One of the data folders for pinhole planesweep provided with the plane "
      "sweep code.");

  boost::program_options::variables_map vm;
  boost::program_options::store(
      boost::program_options::command_line_parser(argc, argv)
          .options(desc)
          .run(),
      vm);
  boost::program_options::notify(vm);

  if (vm.count("help")) {
    std::cout << desc << std::endl;
    return 1;
  }

  // Load camera matrix from file.
  Eigen::Matrix3d K = LoadKMatrixFromFile(dataFolder + "/K.txt");

  // Load camera poses from files.
  std::map<int, PSL::CameraMatrix<double>> cameras =
      LoadCameraMatrixFromFile(dataFolder + "/model-0-cams.txt", K);

  // Load image filenames.
  std::vector<std::string> imageFileNames, image_file_paths;
  LoadImageFilePaths(dataFolder + "/images.txt", dataFolder, imageFileNames,
                     image_file_paths);

  // Compute average distance.
  float minZ, maxZ;
  CalculateMinMaxRange(imageFileNames.size(), cameras, minZ, maxZ);

  //
  MakeOutputFolder("pinholeTestResults");

#if 0
  // Color
  {
    PSL::CudaPlaneSweep cPS =
        CreateCudaPlaneSweepWithdDefaultConfigration(minZ, maxZ);
    cPS.enableColorMatching(true);

    // Use 5 color images.
    {
      int ref_img_id = UploadImageToDevice(image_file_paths, 25, cameras, cPS);

      PinholePlaneSweepTest(
          PSL::PlaneSweepMatchingCosts::PLANE_SWEEP_SAD,
          PSL::PlaneSweepOcclusionMode::PLANE_SWEEP_OCCLUSION_NONE, 0,
          ref_img_id, minZ, maxZ, cPS, 0);


      PinholePlaneSweepTest(
          PSL::PlaneSweepMatchingCosts::PLANE_SWEEP_SAD,
          PSL::PlaneSweepOcclusionMode::PLANE_SWEEP_OCCLUSION_BEST_K, 5,
          ref_img_id, minZ, maxZ, cPS, 0);


    }
  }

#endif
#if 1
  // Gray
  // First tests compute a depth map for the middle image of the first row
  {
    PSL::CudaPlaneSweep cPS =
        CreateCudaPlaneSweepWithdDefaultConfigration(minZ, maxZ);
    cPS.enableColorMatching(false);

    int ref_img_id = UploadImageToDevice(image_file_paths, 25, cameras, cPS);

    // Use 5 gray images.
    {

      PinholePlaneSweepTest(
          PSL::PlaneSweepMatchingCosts::PLANE_SWEEP_SAD,
          PSL::PlaneSweepOcclusionMode::PLANE_SWEEP_OCCLUSION_NONE, 0,
          ref_img_id, minZ, maxZ, cPS, 100);
    }

    // Use 5 gray images.
    {
      // int ref_img_id = UploadImageToDevice(image_file_paths, 5, cameras,
      // cPS);

      PinholePlaneSweepTest(
          PSL::PlaneSweepMatchingCosts::PLANE_SWEEP_ZNCC,
          PSL::PlaneSweepOcclusionMode::PLANE_SWEEP_OCCLUSION_REF_SPLIT, 0,
          ref_img_id, minZ, maxZ, cPS, 100);
    }

    // Use 5 gray images.
    {
      // int ref_img_id = UploadImageToDevice(image_file_paths, 5, cameras,
      // cPS);

      PinholePlaneSweepTest(
          PSL::PlaneSweepMatchingCosts::PLANE_SWEEP_ZNCC,
          PSL::PlaneSweepOcclusionMode::PLANE_SWEEP_OCCLUSION_BEST_K, 5,
          ref_img_id, minZ, maxZ, cPS, 0);
    }
  }
#endif

  return 0;
}

int main(int argc, char *argv[]) {
  // OriginalTest(argc, argv);
  DebugTest(argc, argv);
  return 0;
}
