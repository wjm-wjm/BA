#include<iostream>
#include<random>
#include<vector>
#include<Eigen/Core>
#include<Eigen/Dense>
#include<unsupported/Eigen/MatrixFunctions>
using namespace std;
using namespace Eigen;

class ReprojectionError
{
public:
    ReprojectionError(double camera_id, double fx, double fy, double cx, double cy, double point_id, double u, double v){
        camera_id_ = camera_id;
        fx_ = fx;
        fy_ = fy;
        cx_ = cx;
        cy_ = cy;
        point_id_ = point_id;
        u_ = u;
        v_ = v;
    }

    Matrix3d skew_matrix(Matrix<double,3,1> d){
        Matrix3d m;
        m << 0, -d(2, 0), d(1, 0), d(2, 0), 0, -d(0, 0), -d(1, 0), d(0, 0), 0;
        return m;
    }

    void cal_Rt_se(Matrix<double,6,1> camera_in){
        Matrix<double,3,1> fa = camera_in.head(3);
        R = skew_matrix(fa).exp();
        double theta = fa.norm();
        Matrix<double, 3, 1> a = fa / theta;
        Matrix3d J = sin(theta) / theta * Matrix3d::Identity() + (1 - sin(theta) / theta) * a * a.transpose() + (1 - cos(theta)) / theta * skew_matrix(a);
        t = camera_in.block(0, 3, 3, 1);
        Matrix<double,3,1> rou = J.colPivHouseholderQr().solve(t);
        se << rou, fa;
    }

    double camera_id_, fx_, fy_, cx_, cy_, point_id_, u_, v_;
    Matrix<double, 6, 1> se;
    Matrix3d R;
    Matrix<double, 3, 1> t;
};

int
main()
{
    return 0;
}