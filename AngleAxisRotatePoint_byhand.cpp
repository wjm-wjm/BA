#include<bits/stdc++.h>
#include<Eigen/Core>
#include<Eigen/Dense>
#include "ceres/ceres.h"
#include "ceres/rotation.h"

using namespace std;
using namespace Eigen;

void AngleAxisRotatePoint_byhand(Matrix<double,3,1> angle_axis, double point[3], double result[3]){
    double theta = angle_axis.norm();
    Matrix<double, 3, 1> norm_angle_axis = angle_axis / theta;
    Matrix<double, 3, 3> skew_norm_angle_axis;
    skew_norm_angle_axis << 0, -norm_angle_axis(2, 0), norm_angle_axis(1, 0), norm_angle_axis(2, 0), 0, -norm_angle_axis(0, 0), -norm_angle_axis(1, 0), norm_angle_axis(0, 0), 0;
    Matrix<double, 3, 3> R = cos(theta) * Matrix3d::Identity() + (1 - cos(theta)) * norm_angle_axis * norm_angle_axis.transpose() + sin(theta) * skew_norm_angle_axis;
    Matrix<double, 3, 1> point_;
    point_ << point[0], point[1], point[2];
    Matrix<double, 3, 1> new_point_ = R * point_;
    for (int i = 0; i < 3;i++){
        result[i] = new_point_(i, 0);
    }
}

int main(){
    Matrix<double, 3, 1> angle_axis;
    angle_axis << -1.6943983532198115e-02, 1.1171804676513932e-02, 2.4643508831711991e-03;
    double point[3], result_my[3], result_ceres[3];
    point[0] = 7.3030995682610689e-01;
    point[1] = -2.6490818471043420e-01;
    point[2] = -1.7127892627337182e+00;

    AngleAxisRotatePoint_byhand(angle_axis, point, result_my);

    ceres::AngleAxisRotatePoint(angle_axis.data(), point, result_ceres);

    cout << "result my = " << result_my[0] << " " << result_my[1] << " " << result_my[2] << endl;
    cout << "result ceres = " << result_ceres[0] << " " << result_ceres[1] << " " << result_ceres[2] << endl;

    return 0;
}
