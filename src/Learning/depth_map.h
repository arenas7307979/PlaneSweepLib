
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

#ifndef DEPTH_MAP_H_
#define DEPTH_MAP_H_

// Eigen
#include <Eigen/Core>

// Boost
#include <boost/shared_array.hpp>

// OpenCV
#include <opencv2/core.hpp>

// Original
#include <camera_matrix.h>

namespace vis {

template <typename T, typename U>
class DepthMap {
 public:
  DepthMap();
  DepthMap(unsigned int width, unsigned int height, const CameraMatrix<U> &cam);
  void Initialize(T value);

  // Overload
  DepthMap &operator*=(T scale);
  T &operator()(int x, int y);
  const T &operator()(int x, int y) const;

  void ComputeDepthMat(T min_z, T max_z, cv::Mat &depth_inv_mat);
  void ComputeColoredDepthMat(T min_z, T max_z, cv::Mat &depth_inv_mat);
  Eigen::Matrix<U, 4, 1> Unproject(const int x, const int y) const;

  // Setter and Getter.
  void SetCam(const CameraMatrix<U> &cam);
  const CameraMatrix<U> &GetCam() const;
  void SetRT(const Eigen::Matrix<U, 3, 4> &RT);
  T *GetDataPtr();
  unsigned int GetWidth() const;
  unsigned int GetHeight() const;

 private:
  unsigned int m_width;
  unsigned int m_height;
  boost::shared_array<T> m_depths;
  CameraMatrix<U> m_cam;
};

}  // namespace vis

#endif  // DEPTH_MAP_H_
