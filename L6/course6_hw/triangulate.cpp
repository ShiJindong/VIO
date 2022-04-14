//
// Created by hyj on 18-11-11.
//
#include <iostream>
#include <vector>
#include <random>  
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Eigenvalues>
#include <fstream>

struct Pose
{
    Pose(Eigen::Matrix3d R, Eigen::Vector3d t):Rwc(R),qwc(R),twc(t) {};
    Eigen::Matrix3d Rwc;
    Eigen::Quaterniond qwc;
    Eigen::Vector3d twc;

    Eigen::Vector2d uv;    // 这帧图像观测到的特征坐标
};


int main()
{
    int poseNums = 10;
    double radius = 8;
    double fx = 1.;
    double fy = 1.;
    std::vector<Pose> camera_pose;
    for(int n = 0; n < poseNums; ++n ) {
        double theta = n * 2 * M_PI / ( poseNums * 4); // 1/4 圆弧
        // 绕 z轴 旋转
        Eigen::Matrix3d R;
        R = Eigen::AngleAxisd(theta, Eigen::Vector3d::UnitZ());
        Eigen::Vector3d t = Eigen::Vector3d(radius * cos(theta) - radius, radius * sin(theta), 1 * sin(2 * theta));
        camera_pose.push_back(Pose(R,t));
    }

    // 随机数生成 1 个 三维特征点
    std::default_random_engine generator;
    std::uniform_real_distribution<double> xy_rand(-4, 4.0);
    std::uniform_real_distribution<double> z_rand(8., 10.);
    double tx = xy_rand(generator);
    double ty = xy_rand(generator);
    double tz = z_rand(generator);

    Eigen::Vector3d Pw(tx, ty, tz);

    // 相机测量噪声
    double noise_stddev = 50./1000.;    // 相机噪声标准差 2pixel / focal
    std::normal_distribution<double> noise_pdf(0., noise_stddev);

    // 这个特征从第三帧相机开始被观测，i=3
    int start_frame_id = 3;
    int end_frame_id = poseNums;
    for (int i = start_frame_id; i < end_frame_id; ++i) {
        Eigen::Matrix3d Rcw = camera_pose[i].Rwc.transpose();
        Eigen::Vector3d Pc = Rcw * (Pw - camera_pose[i].twc);       // Pw = Rwc * Pc + twc

        double x = Pc.x();
        double y = Pc.y();
        double z = Pc.z();

        camera_pose[i].uv = Eigen::Vector2d(x/z + noise_pdf(generator),y/z + noise_pdf(generator));
    }
    
    /// TODO::homework; 请完成三角化估计深度的代码
    // 遍历所有的观测数据，并三角化
    Eigen::Vector3d P_est;           // 结果保存到这个变量
    P_est.setZero();

    /* your code begin */
    // 构造矩阵D
    Eigen::Matrix<double, Eigen::Dynamic, 4> D;
    int size = (end_frame_id - start_frame_id) * 2;
    D.conservativeResize(size, 4);
    D.setZero();
    for(int i = start_frame_id; i < end_frame_id; ++i)
    {
        Eigen::Matrix<double, 3, 4> Pk;
        // From World to Camera:   Pw = Rwc * Pc + twc   ->    Pc = Rcw * (Pw - twc)
        auto Rcw = camera_pose[i].Rwc.transpose();
        auto tcw =  -Rcw * camera_pose[i].twc;
        Pk << Rcw, tcw;
        D.block(2*(i-start_frame_id), 0, 1, 4) = camera_pose[i].uv[0] * Pk.block(2,0,1,4) - Pk.block(0,0,1,4);
        D.block(2*(i-start_frame_id)+1, 0, 1, 4) = camera_pose[i].uv[1] * Pk.block(2,0,1,4) - Pk.block(1,0,1,4);
    }

    // SVD分解
    // 对于square matric 与 thin matric的区别: Thin unitaries are only available if your matrix type has a Dynamic number of columns (for example MatrixXd)
    Eigen::JacobiSVD<Eigen::Matrix4d> svd(D.transpose() * D, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Vector4d singular_values = svd.singularValues();       // Singular values are always sorted in decreasing order
    Eigen::Matrix4d U = svd.matrixU();
    Eigen::Matrix4d V = svd.matrixV();

    std::cout << "Singular values of Matrix DT*D: \n" << singular_values.transpose() << std::endl;
    P_est = U.block(0,3,3,1) / U(3,3);

    std::cout <<"ground truth: \n"<< Pw.transpose() <<std::endl;
    std::cout <<"your result: \n"<< P_est.transpose() <<std::endl;

    // 计算位移的RMSE
    auto e = Pw - P_est;
    double RMSE_translation = std::sqrt(e.transpose() * e);
    std::cout <<"RMSE of translation: \n"<< RMSE_translation <<std::endl;

    // TODO:: 请如课程讲解中提到的判断三角化结果好坏的方式，绘制奇异值比值变化曲线
    // 最小奇异值和第二小奇异值的比值
    double singular_value_ratio = singular_values[3] / singular_values[2];
    std::cout << "singular_value_ratio: \n" << singular_value_ratio << std::endl;
    // 将结果输出到文件，进行后续绘图
    std::string filename = "data.txt";
    std::ofstream ofs(filename, std::ofstream::app);
    if(!ofs.is_open())
        std::cout << "File " << filename << " is not found." << std::endl;
    ofs << "observe frames: " << end_frame_id - start_frame_id << "     noise stddev: " << noise_stddev
        << "    RMSE of translation: " << RMSE_translation << "        singular value ratio: "<< singular_value_ratio << "\n";

    return 0;
}
