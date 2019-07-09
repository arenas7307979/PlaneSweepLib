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

#ifndef CAMERA_MATRIX_H_
#define CAMERA_MATRIX_H_

// Eigen
#include <Eigen/Core>

namespace vis {

template <typename NumericType>
class CameraMatrix {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CameraMatrix();
  CameraMatrix(const Eigen::Matrix<NumericType, 3, 4> &P);
  CameraMatrix(const Eigen::Matrix<NumericType, 3, 3> &K,
               const Eigen::Matrix<NumericType, 3, 3> &R,
               const Eigen::Matrix<NumericType, 3, 1> &T);

  // Copy Constructor and Assignment
  CameraMatrix(const CameraMatrix<NumericType> &other_camera_matrix);
  CameraMatrix &operator=(const CameraMatrix<NumericType> &other_camera_matrix);

  void ScaleK(const NumericType scale_x, const NumericType scale_y);
  void SetKRT(const Eigen::Matrix<NumericType, 3, 3> &K,
              const Eigen::Matrix<NumericType, 3, 3> &R,
              const Eigen::Matrix<NumericType, 3, 1> &T);
  void SetP(const Eigen::Matrix<NumericType, 3, 4> &P);
  void SetRT(const Eigen::Matrix<NumericType, 3, 4> &RT);

  const Eigen::Matrix<NumericType, 3, 3> &GetK() const;
  const Eigen::Matrix<NumericType, 3, 3> &GetR() const;
  const Eigen::Matrix<NumericType, 3, 1> &GetT() const;
  const Eigen::Matrix<NumericType, 3, 4> &GetP() const;
  const Eigen::Matrix<NumericType, 3, 1> &GetC() const;
  const Eigen::Matrix<NumericType, 4, 4> &GetCamToGlobal() const;

  Eigen::Matrix<NumericType, 4, 1> UnprojectPoint(
      const NumericType x, const NumericType y, const NumericType depth) const;

  Eigen::Matrix<NumericType, 4, 1> LocalPoint2GlobalPoint(
      const Eigen::Matrix<NumericType, 4, 1> &local_point) const;

  void LoadFromDepthMapDataFile(std::string filepath);

 private:
  void RecomputeStoredValues();

 private:
  Eigen::Matrix<NumericType, 3, 3> m_K;
  Eigen::Matrix<NumericType, 3, 3> m_R;
  Eigen::Matrix<NumericType, 3, 1> m_T;

  Eigen::Matrix<NumericType, 3, 4> m_P;
  Eigen::Matrix<NumericType, 4, 4> m_cam_to_global;
  Eigen::Matrix<NumericType, 3, 1> m_C;
};

}  // namespace vis

#endif  // CAMERA_MATRIX_H_
