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
#include <camera_matrix.h>

// Eigen
#include <Eigen/Dense>

namespace vis {

template <typename NumericType>
CameraMatrix<NumericType>::CameraMatrix() {
  this->m_K.setIdentity();
  this->m_R.setIdentity();
  this->m_T.setZero();
}

template <typename NumericType>
CameraMatrix<NumericType>::CameraMatrix(
    const Eigen::Matrix<NumericType, 3, 4> &P) {
  SetP(P);
}

template <typename NumericType>
CameraMatrix<NumericType>::CameraMatrix(
    const Eigen::Matrix<NumericType, 3, 3> &K,
    const Eigen::Matrix<NumericType, 3, 3> &R,
    const Eigen::Matrix<NumericType, 3, 1> &T) {
  this->m_K = K;
  this->m_R = R;
  this->m_T = T;

  // scale K such that K(2,2) = 1
  NumericType scale = 1 / K(2, 2);
  this->m_K *= scale;
  RecomputeStoredValues();
}

template <typename NumericType>
CameraMatrix<NumericType>::CameraMatrix(
    const CameraMatrix<NumericType> &otherCameraMatrix) {
  this->m_K = otherCameraMatrix.m_K;
  this->m_R = otherCameraMatrix.m_R;
  this->m_T = otherCameraMatrix.m_T;
  RecomputeStoredValues();
}

template <typename NumericType>
CameraMatrix<NumericType> &CameraMatrix<NumericType>::operator=(
    const CameraMatrix<NumericType> &other_camera_matrix) {
  this->m_K = other_camera_matrix.m_K;
  this->m_R = other_camera_matrix.m_R;
  this->m_T = other_camera_matrix.m_T;
  RecomputeStoredValues();
  return *this;
}

template <typename NumericType>
void CameraMatrix<NumericType>::ScaleK(NumericType scale_x,
                                       NumericType scale_y) {
  m_K(0, 0) *= scale_x;
  m_K(0, 1) *= scale_x;
  m_K(0, 2) = (m_K(0, 2) + (NumericType)0.5) * scale_x - (NumericType)0.5;
  m_K(1, 1) *= scale_y;
  m_K(1, 2) = (m_K(1, 2) + (NumericType)0.5) * scale_y - (NumericType)0.5;
}

template <typename NumericType>
void CameraMatrix<NumericType>::SetKRT(
    const Eigen::Matrix<NumericType, 3, 3> &K,
    const Eigen::Matrix<NumericType, 3, 3> &R,
    const Eigen::Matrix<NumericType, 3, 1> &T) {
  this->m_K = K;
  this->m_R = R;
  this->m_T = T;
  RecomputeStoredValues();
}

template <typename NumericType>
void CameraMatrix<NumericType>::SetP(
    const Eigen::Matrix<NumericType, 3, 4> &P) {
  this->m_K(0, 0) = P(0, 0);
  this->m_K(0, 1) = P(0, 1);
  this->m_K(0, 2) = P(0, 2);
  this->m_K(1, 0) = P(1, 0);
  this->m_K(1, 1) = P(1, 1);
  this->m_K(1, 2) = P(1, 2);
  this->m_K(2, 0) = P(2, 0);
  this->m_K(2, 1) = P(2, 1);
  this->m_K(2, 2) = P(2, 2);

  // Calculate RQ Decomposition of M.
  // Hartley & Zissermann, p579.
  NumericType d = std::sqrt(this->m_K(2, 2) * this->m_K(2, 2) +
                            this->m_K(2, 1) * this->m_K(2, 1));
  NumericType c = -this->m_K(2, 2) / d;
  NumericType s = this->m_K(2, 1) / d;
  Eigen::Matrix<NumericType, 3, 3> Rx;
  Rx << 1, 0, 0, 0, c, -s, 0, s, c;
  this->m_K = this->m_K * Rx;

  d = std::sqrt(this->m_K(2, 2) * this->m_K(2, 2) +
                this->m_K(2, 0) * this->m_K(2, 0));
  c = this->m_K(2, 2) / d;
  s = this->m_K(2, 0) / d;

  Eigen::Matrix<NumericType, 3, 3> Ry;
  Ry << c, 0, s, 0, 1, 0, -s, 0, c;
  this->m_K = this->m_K * Ry;

  d = std::sqrt(this->m_K(1, 1) * this->m_K(1, 1) +
                this->m_K(1, 0) * this->m_K(1, 0));
  c = -this->m_K(1, 1) / d;
  s = this->m_K(1, 0) / d;

  Eigen::Matrix<NumericType, 3, 3> Rz;
  Rz << c, -s, 0, s, c, 0, 0, 0, 1;
  this->m_K = this->m_K * Rz;

  Eigen::Matrix<NumericType, 3, 3> Sign;
  Sign.setIdentity();
  if (this->m_K(0, 0) < 0) {
    Sign(0, 0) = -1;
  }
  if (this->m_K(1, 1) < 0) {
    Sign(1, 1) = -1;
  }
  if (this->m_K(2, 2) < 0) {
    Sign(2, 2) = -1;
  }

  this->m_K = this->m_K * Sign;

  this->m_R = Rx * Ry * Rz * Sign;
  this->m_R.transposeInPlace();

  this->m_T(0) = P(0, 3);
  this->m_T(1) = P(1, 3);
  this->m_T(2) = P(2, 3);

  this->m_T = this->m_K.inverse() * this->m_T;

  // scale K such that K(2,2) = 1
  NumericType scale = 1 / this->m_K(2, 2);
  this->m_K(0, 0) *= scale;
  this->m_K(0, 1) *= scale;
  this->m_K(0, 2) *= scale;
  this->m_K(1, 1) *= scale;
  this->m_K(1, 2) *= scale;
  this->m_K(2, 2) *= scale;

  RecomputeStoredValues();
}

template <typename NumericType>
void CameraMatrix<NumericType>::SetRT(
    const Eigen::Matrix<NumericType, 3, 4> &RT) {
  Eigen::Matrix<NumericType, 3, 3> R;
  Eigen::Matrix<NumericType, 3, 1> T;
  R(0, 0) = RT(0, 0);
  R(0, 1) = RT(0, 1);
  R(0, 2) = RT(0, 2);
  T(0) = RT(0, 3);
  R(1, 0) = RT(1, 0);
  R(1, 1) = RT(1, 1);
  R(1, 2) = RT(1, 2);
  T(1) = RT(1, 3);
  R(2, 0) = RT(2, 0);
  R(2, 1) = RT(2, 1);
  R(2, 2) = RT(2, 2);
  T(2) = RT(2, 3);

  this->m_R = R;
  this->m_T = T;

  RecomputeStoredValues();
}

template <typename NumericType>
const Eigen::Matrix<NumericType, 3, 3> &CameraMatrix<NumericType>::GetK()
    const {
  return this->m_K;
}

template <typename NumericType>
const Eigen::Matrix<NumericType, 3, 3> &CameraMatrix<NumericType>::GetR()
    const {
  return this->m_R;
}

template <typename NumericType>
const Eigen::Matrix<NumericType, 3, 1> &CameraMatrix<NumericType>::GetT()
    const {
  return this->m_T;
}

template <typename NumericType>
const Eigen::Matrix<NumericType, 3, 4> &CameraMatrix<NumericType>::GetP()
    const {
  return this->m_P;
}

template <typename NumericType>
const Eigen::Matrix<NumericType, 3, 1> &CameraMatrix<NumericType>::GetC()
    const {
  return this->m_C;
}

template <typename NumericType>
Eigen::Matrix<NumericType, 4, 1> CameraMatrix<NumericType>::UnprojectPoint(
    const NumericType x, const NumericType y, const NumericType depth) const {
  Eigen::Matrix<NumericType, 4, 1> point_in_cam;

  point_in_cam(0) = (x - m_K(0, 2)) * (depth / m_K(0, 0));
  point_in_cam(1) = (x - m_K(1, 2)) * (depth / m_K(1, 1));
  point_in_cam(2) = depth;
  point_in_cam(3) = 1;

  return this->m_cam_to_global * point_in_cam;
}

template <typename NumericType>
const Eigen::Matrix<NumericType, 4, 4>
    &CameraMatrix<NumericType>::GetCamToGlobal() const {
  return this->m_cam_to_global;
}

template <typename NumericType>
void CameraMatrix<NumericType>::LoadFromDepthMapDataFile(std::string filepath) {
}

template <typename NumericType>
void CameraMatrix<NumericType>::RecomputeStoredValues() {
  // compute P
  this->m_P.topLeftCorner(3, 3) = this->m_R;
  this->m_P.rightCols(1) = this->m_T;
  this->m_P = this->m_K * this->m_P;

  // compute C
  this->m_C = -this->m_R.inverse() * this->m_T;

  // compute cam2Global
  this->m_cam_to_global.setIdentity();
  this->m_cam_to_global.topLeftCorner(3, 3) = this->m_R.transpose();
  this->m_cam_to_global.topRightCorner(3, 1) = this->m_C;
}

template class CameraMatrix<double>;
template class CameraMatrix<float>;

}  // namespace vis