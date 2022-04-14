#include <iostream>
#include <random>
#include "backend/vertex_inverse_depth.h"
#include "backend/vertex_pose.h"
#include "backend/edge_reprojection.h"
#include "backend/problem.h"
#include "backend/edge_prior.h"
#include <pangolin/pangolin.h>

using namespace myslam::backend;
using namespace std;

/*
 * Frame : 保存每帧的姿态和观测
 */
struct Frame {
    Frame(Eigen::Matrix3d R, Eigen::Vector3d t) : Rwc(R), qwc(R), twc(t) {};
    Eigen::Matrix3d Rwc;
    Eigen::Quaterniond qwc;
    Eigen::Vector3d twc;

    unordered_map<int, Eigen::Vector3d> featurePerId; // 该帧观测到的特征以及特征id
};

/*
 * 产生世界坐标系下的虚拟数据: 相机姿态, 特征点, 以及每帧观测
 */
void GetSimDataInWordFrame(vector<Frame> &cameraPoses, vector<Eigen::Vector3d> &points) {
    int featureNums = 30;  // 特征数目20，假设每帧都能观测到所有的特征
    int poseNums = 10;     // 相机数目3

    double radius = 8;
    for (int n = 0; n < poseNums; ++n) {
        double theta = n * 2 * M_PI / (poseNums * 4); // 1/4 圆弧
        // 绕 z轴 旋转
        Eigen::Matrix3d R;
        R = Eigen::AngleAxisd(theta, Eigen::Vector3d::UnitZ());
        Eigen::Vector3d t = Eigen::Vector3d(radius * cos(theta) - radius, radius * sin(theta), 1 * sin(2 * theta));
        cameraPoses.push_back(Frame(R, t));
    }

    // 随机数生成三维特征点
    std::default_random_engine generator;
    std::normal_distribution<double> noise_pdf(0., 1. / 1000.);  // 2pixel / focal
    for (int j = 0; j < featureNums; ++j) {
        std::uniform_real_distribution<double> xy_rand(-4, 4.0);
        std::uniform_real_distribution<double> z_rand(4., 8.);

        Eigen::Vector3d Pw(xy_rand(generator), xy_rand(generator), z_rand(generator));
        points.push_back(Pw);

        // 在每一帧上的观测量
        for (int i = 0; i < poseNums; ++i) {
            Eigen::Vector3d Pc = cameraPoses[i].Rwc.transpose() * (Pw - cameraPoses[i].twc);     // Pw = Rwc * Pc + twc
            Pc = Pc / Pc.z();  // 归一化图像平面
            Pc[0] += noise_pdf(generator);
            Pc[1] += noise_pdf(generator);
            cameraPoses[i].featurePerId.insert(make_pair(j, Pc));   // 第i个相机位姿观测到的第j个路标点的相机坐标系下的归一化坐标
        }
    }
}

void draw_curve(const vector<double, std::allocator<double>> &currentValue_vec, string name){
    // Create OpenGL window in single line
    pangolin::CreateWindowAndBind(name,640,480);
    // Data logger object
    pangolin::DataLog log;
    // Optionally add named labels
    std::vector<std::string> labels;
    labels.push_back(std::string(name));
    log.SetLabels(labels);
    // OpenGL 'view' of data. We might have many views of the same data.
    auto maxValue = max_element(currentValue_vec.begin(), currentValue_vec.end());
    pangolin::Plotter plotter(&log,0,currentValue_vec.size() - 1,0,*maxValue,1,*maxValue/20);
    plotter.SetBounds(0.0, 1.0, 0.0, 1.0);
    pangolin::DisplayBase().AddDisplay(plotter);
    // Default hooks for exiting (Esc) and fullscreen (tab).
    std::vector<double>::size_type iter = 0;
    while( !pangolin::ShouldQuit() )
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        if(iter < currentValue_vec.size()){
            log.Log(currentValue_vec[iter]);

        }
        iter++;

        // Render graph, Swap frames and Process Events
        pangolin::FinishFrame();
    }
}


void SolveProblemWithDifferentParameters(const vector<Frame> &cameras, const vector<Eigen::Vector3d> &points,
                                         const double wp, vector<double, std::allocator<double>> &used_iterations_vec,
                                         vector<double, std::allocator<double>> &used_time_vec,
                                         vector<double, std::allocator<double>> &RMSE_points_vec,
                                         vector<double, std::allocator<double>> &RMSE_pose_vec)
{
    std::cout << "----------weight of prior wp: " << wp << " -------------" << std::endl;
    Eigen::Quaterniond qic(1, 0, 0, 0);     // imu 和 camera 之间的坐标变换关系
    Eigen::Vector3d tic(0, 0, 0);
    // 构建 problem
    Problem problem(Problem::ProblemType::SLAM_PROBLEM);

    // 所有 Pose
    vector<shared_ptr<VertexPose> > vertexCams_vec;
    for (size_t i = 0; i < cameras.size(); ++i) {
        shared_ptr<VertexPose> vertexCam(new VertexPose());
        Eigen::VectorXd pose(7);
        pose << cameras[i].twc, cameras[i].qwc.x(), cameras[i].qwc.y(), cameras[i].qwc.z(), cameras[i].qwc.w();
        // 这里设定待优化相机位姿的初始值为真实值
        vertexCam->SetParameters(pose);

        // if(i < 2)
        // vertexCam->SetFixed();

        problem.AddVertex(vertexCam);
        vertexCams_vec.push_back(vertexCam);
    }

    // 所有 Point 及 edge
    std::default_random_engine generator;
    // std::normal_distribution<double> noise_pdf(0, 1.);
    std::normal_distribution<double> noise_pdf(0, 0.1);
    double noise = 0;
    vector<double> noise_invd;
    vector<shared_ptr<VertexInverseDepth> > allPoints;
    for (size_t i = 0; i < points.size(); ++i) {
        //假设所有特征点的起始帧为第0帧， 逆深度容易得到
        Eigen::Vector3d Pw = points[i];
        Eigen::Vector3d Pc = cameras[0].Rwc.transpose() * (Pw - cameras[0].twc);
        noise = noise_pdf(generator);
        double inverse_depth = 1. / (Pc.z() + noise);
        // double inverse_depth = 1. / Pc.z();
        noise_invd.push_back(inverse_depth);

        // 初始化特征 vertex
        shared_ptr<VertexInverseDepth> verterxPoint(new VertexInverseDepth());
        VecX inv_d(1);
        inv_d << inverse_depth;
        verterxPoint->SetParameters(inv_d);
        problem.AddVertex(verterxPoint);
        allPoints.push_back(verterxPoint);

        // 每个特征对应的投影误差, 第 0 帧为起始帧
        for (size_t j = 1; j < cameras.size(); ++j) {
            Eigen::Vector3d pt_i = cameras[0].featurePerId.find(i)->second;   // 第0个相机位姿第i个特征点的相机坐标系下的归一化坐标
            Eigen::Vector3d pt_j = cameras[j].featurePerId.find(i)->second;
            // 每个路标点在第0个和第j个相机位姿之间都有一条边，可以计算相应的重投影误差
            shared_ptr<EdgeReprojection> edge(new EdgeReprojection(pt_i, pt_j));
            edge->SetTranslationImuFromCamera(qic, tic);

            std::vector<std::shared_ptr<Vertex> > edge_vertex;
            // 每条边有3个顶点: 相机坐标系下路标点的逆深度 + 第0个位姿 + 第j个位姿
            edge_vertex.push_back(verterxPoint);
            edge_vertex.push_back(vertexCams_vec[0]);
            edge_vertex.push_back(vertexCams_vec[j]);
            edge->SetVertex(edge_vertex);
            problem.AddEdge(edge);
        }


        // double wp = 1e5;
        for (size_t i = 0; i < 2; ++i) {
            // 将先验的位姿残差看作额外的边，放入problem进行优化
            shared_ptr<EdgeSE3Prior> edge_prior(new EdgeSE3Prior(cameras[i].twc, cameras[i].qwc));
            std::vector<std::shared_ptr<Vertex> > edge_prior_vertex;
            edge_prior_vertex.push_back(vertexCams_vec[i]);
            edge_prior->SetVertex(edge_prior_vertex);
            // 增大先验位姿残差的权重
            edge_prior->SetInformation(edge_prior->Information() * wp);
            problem.AddEdge(edge_prior);
        }


    }

    int used_iterations = 0;
    double used_time = 0;
    problem.Solve(15, used_iterations, used_time);

    used_iterations_vec.push_back(used_iterations);
    used_time_vec.push_back(used_time);

    double RMSE_points = 0.;
    // std::cout << "\nCompare MonoBA results after opt..." << std::endl;
    for (size_t k = 0; k < allPoints.size(); k+=1) {
        // std::cout << "after opt, point " << k << " : gt " << 1. / points[k].z() << " ,noise "
        //           << noise_invd[k] << " ,opt " << allPoints[k]->Parameters() << std::endl;
        Vec1 e = Vec1(1. / points[k].z()) - allPoints[k]->Parameters();
        RMSE_points += e.transpose() * e;
    }
    RMSE_points = std::sqrt(RMSE_points/allPoints.size());
    std::cout << "RMSE of points: " << RMSE_points << std::endl;

    // std::cout<<"------------ pose translation ----------------"<<std::endl;
    double RMSE_pose = 0.;
    for (size_t i = 0; i < vertexCams_vec.size(); ++i) {
        // std::cout<<"translation after opt: "<< i <<" :"<< vertexCams_vec[i]->Parameters().head(3).transpose() << " || gt: "<<cameras[i].twc.transpose()<<std::endl;
        Vec3 e = vertexCams_vec[i]->Parameters().head(3) - cameras[i].twc;
        RMSE_pose += e.transpose() * e;
    }
    RMSE_pose = std::sqrt(RMSE_pose/vertexCams_vec.size());
    std::cout << "RMSE of pose: " << RMSE_pose << std::endl;
    /// 优化完成后，第一帧相机的 pose 平移（x,y,z）不再是原点 0,0,0. 说明向零空间发生了漂移。
    /// 解决办法： fix 第一帧和第二帧，固定 7 自由度。 或者加上非常大的先验值。

    RMSE_points_vec.push_back(RMSE_points);
    RMSE_pose_vec.push_back(RMSE_pose);
}


int main() {
    // 准备数据
    vector<Frame> cameras;
    vector<Eigen::Vector3d> points;
    GetSimDataInWordFrame(cameras, points);   // cameras: 相机世界坐标下位姿，路标点相机坐标系下的归一化坐标  points: 路标点世界坐标系下的坐标

    vector<double> wp_vec{1, 10, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8};
    vector<double, std::allocator<double>> used_iterations_vec;
    vector<double, std::allocator<double>> used_time_vec;
    vector<double, std::allocator<double>> RMSE_points_vec;
    vector<double, std::allocator<double>> RMSE_pose_vec;
    for(const auto &wp:wp_vec)
    {
        SolveProblemWithDifferentParameters(cameras, points, wp, used_iterations_vec, used_time_vec, RMSE_points_vec, RMSE_pose_vec);
        std::cout << "\n";
    }

    draw_curve(used_iterations_vec, "used iterations [steps]");
    draw_curve(used_time_vec, "used time [ms]");
    draw_curve(RMSE_points_vec, "RMSE of points");
    draw_curve(RMSE_pose_vec, "RMSE of translations");
    // problem.TestMarginalize();

    return 0;
}

