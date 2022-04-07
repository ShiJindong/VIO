//
// Created by jindong on 06/12/2021.
//

#include <sophus/so3.hpp>
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Geometry>

int main()
{
    // create a rotation matrix R and corresponding quaternion q
    Eigen::Matrix3d R = Eigen::AngleAxisd(M_PI/2, Eigen::Vector3d(0,0,1)).toRotationMatrix();
    Eigen::Quaterniond q(R);

    Eigen::Vector3d w_so3(0.01, 0.02, 0.03);

    Sophus::SO3d R_updated = Sophus::SO3d(R) * Sophus::SO3d::exp(w_so3);
    Eigen::Quaterniond q_updated = q * Eigen::Quaterniond(1, 0.5 * w_so3(0), 0.5 * w_so3(1), 0.5 * w_so3(2));

    std::cout << "R_updated = " << R_updated.matrix() << std::endl << std::endl
              << "q_updated = " << q_updated.matrix() << std::endl;

}

