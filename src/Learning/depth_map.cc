
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

// Self Header
#include <depth_map.h>

// Original
#include <color_map_jet.h>

namespace vis {

template <typename T, typename U>
DepthMap<T, U>::DepthMap() : m_width(0), m_height(0), m_depths(nullptr) {}

template <typename T, typename U>
DepthMap<T, U>::DepthMap(unsigned int width, unsigned int height,
                         const CameraMatrix<U> &cam) {
  m_width = width;
  m_height = height;
  m_cam = cam;
  m_depths.reset(new T[width * height]);
}

template <typename T, typename U>
void DepthMap<T, U>::Initialize(T value) {
  for (unsigned int i = 0; i < m_width * m_height; i++) {
    m_depths[i] = value;
  }
}

template <typename T, typename U>
DepthMap<T, U> &DepthMap<T, U>::operator*=(T scale) {
  for (unsigned int i = 0; i < m_width * m_height; i++) {
    if (m_depths[i] > 0) {
      m_depths[i] *= scale;
    }
  }
  return *this;
}

template <typename T, typename U>
T &DepthMap<T, U>::operator()(int x, int y) {
  return m_depths[y * m_width + x];
}

template <typename T, typename U>
const T &DepthMap<T, U>::operator()(int x, int y) const {
  return m_depths[y * m_width + x];
}

template <typename T, typename U>
void DepthMap<T, U>::ComputeDepthMat(T min_z, T max_z, cv::Mat &depth_inv_mat) {
  cv::Mat_<T> depth_mat(m_height, m_width, this->GetDataPtr());
  cv::Mat_<T> inv_depth_mat(m_height, m_width);

  for (unsigned int y = 0; y < m_height; y++) {
    for (unsigned int x = 0; x < m_width; x++) {
      const T depth = depth_mat[y][x];
      if (depth > 0) {
        inv_depth_mat[y][x] = (1 / depth - 1 / max_z) / (1 / min_z - 1 / max_z);
      } else {
        inv_depth_mat[y][x] = 0;
      }
    }
  }
  inv_depth_mat = depth_mat;
}

template <typename T, typename U>
void DepthMap<T, U>::ComputeColoredDepthMat(T min_z, T max_z,
                                            cv::Mat &depth_inv_mat) {
  cv::Mat col_inv_depth = cv::Mat::zeros(cv::Size(m_width, m_height), CV_8UC3);

  for (int y = 0; y < col_inv_depth.rows; y++) {
    unsigned char *pixel = col_inv_depth.ptr<unsigned char>(y);
    for (int x = 0; x < col_inv_depth.cols; x++) {
      const T depth = (*this)(x, y);
      if (depth > 0) {
        int idx = (int)round(std::max((T)0, std::min(1 / depth - 1 / max_z,
                                                     1 / min_z - 1 / max_z) /
                                                (1 / min_z - 1 / max_z)) *
                             (T)255);
        pixel[0] = (int)round(g_color_map_jet[idx][2] * 255.0f);
        pixel[1] = (int)round(g_color_map_jet[idx][1] * 255.0f);
        pixel[2] = (int)round(g_color_map_jet[idx][0] * 255.0f);
      }
      pixel = pixel + 3;
    }
  }
  depth_inv_mat = col_inv_depth;
}

template <typename T, typename U>
Eigen::Matrix<U, 4, 1> DepthMap<T, U>::Unproject(int x, int y) const {
  // T -> U.
  U depth = (U)(*this)(x, y);
  if (depth <= 0) {
    return Eigen::Matrix<U, 4, 1>::Zero();
  }
  return GetCam().UnprojectPoint((U)x, (U)y, depth);
}

template <typename T, typename U>
void DepthMap<T, U>::SetCam(const CameraMatrix<U> &cam) {
  m_cam = cam;
}

template <typename T, typename U>
const CameraMatrix<U> &DepthMap<T, U>::GetCam() const {
  return m_cam;
}

template <typename T, typename U>
void DepthMap<T, U>::SetRT(const Eigen::Matrix<U, 3, 4> &RT) {
  m_cam.SetRT(RT);
}

template <typename T, typename U>
T *DepthMap<T, U>::GetDataPtr() {
  return &(m_depths[0]);
}

template <typename T, typename U>
unsigned int DepthMap<T, U>::GetWidth() const {
  return m_width;
}

template <typename T, typename U>
unsigned int DepthMap<T, U>::GetHeight() const {
  return m_height;
}

template class DepthMap<float, double>;

}  // namespace vis
