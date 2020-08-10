#include<bits/stdc++.h>
#include<Eigen/Core>
#include<Eigen/Dense>
#include "sophus/so3.h"
#include "sophus/se3.h"

using namespace std;
using namespace Eigen;

Matrix3d skew_matrix(Matrix<double,3,1> d){
    Matrix3d m;
    m << 0, -d(2, 0), d(1, 0), d(2, 0), 0, -d(0, 0), -d(1, 0), d(0, 0), 0;
    return m;
}

Matrix<double,6,1> se3_exp_LeftMutiply(Matrix<double,6,1> delta_se3, Matrix<double,6,1> original_se3){
    //返回的是se(3)的形式
    Matrix<double, 3, 1> rou = original_se3.block(0, 0, 3, 1);
    Matrix<double, 3, 1> fa = original_se3.block(3, 0, 3, 1);
    double theta = fa.norm();
    Matrix<double, 3, 1> a = fa / theta;

    Matrix<double, 3, 3> Jl_inverse = (theta / 2) * (1 / tan(theta / 2)) * Matrix3d::Identity() + (1 - (theta / 2) * (1 / tan(theta / 2))) * a * a.transpose() - (theta / 2) * skew_matrix(a);
    //Matrix<double, 3, 3> Jl_inverse = (theta / 2) * (cos(theta / 2) / sin(theta / 2)) * Matrix3d::Identity() + (1 - (theta / 2) * (cos(theta / 2) / sin(theta / 2))) * a * a.transpose() - (theta / 2) * skew_matrix(a);

    Matrix<double, 3, 3> Ql = 0.5 * skew_matrix(rou) + ((theta - sin(theta)) / pow(theta, 3.0)) * (skew_matrix(fa) * skew_matrix(rou) + skew_matrix(rou) * skew_matrix(fa) + skew_matrix(fa) * skew_matrix(rou) * skew_matrix(fa))
                            + ((pow(theta, 2.0) + 2 * cos(theta) - 2) / (2 * pow(theta, 4.0))) * (skew_matrix(fa) * skew_matrix(fa) * skew_matrix(rou) + skew_matrix(rou) * skew_matrix(fa) * skew_matrix(fa) - 3 * skew_matrix(fa) * skew_matrix(rou) * skew_matrix(fa))
                            + ((2 * theta - 3 * sin(theta) + theta * cos(theta)) / (2 * pow(theta, 5.0))) * (skew_matrix(fa) * skew_matrix(rou) * skew_matrix(fa) * skew_matrix(fa) + skew_matrix(fa) * skew_matrix(fa) * skew_matrix(rou) * skew_matrix(fa));

    Matrix<double, 6, 6> Fl_inverse = MatrixXd::Zero(6, 6);
    Fl_inverse.block(0, 0, 3, 3) = Jl_inverse;
    Fl_inverse.block(0, 3, 3, 3) = -Jl_inverse * Ql * Jl_inverse;
    Fl_inverse.block(3, 3, 3, 3) = Jl_inverse;

    Matrix<double, 6, 1> new_se3 = Fl_inverse * delta_se3 + original_se3;

    return new_se3;
}

Matrix<double,4,4> se3_exp_ToMatrix(Matrix<double,6,1> se3){
    Matrix<double, 3, 1> fa = se3.block(3, 0, 3, 1);
    Matrix<double, 3, 1> rou = se3.block(0, 0, 3, 1);
    double theta = fa.norm();
    Matrix<double, 3, 1> a = fa / theta;
    Matrix<double, 3, 3> R = cos(theta) * Matrix3d::Identity() + (1 - cos(theta)) * a * a.transpose() + sin(theta) * skew_matrix(a);
    Matrix<double, 3, 3> J = sin(theta) / theta * Matrix3d::Identity() + (1 - sin(theta) / theta) * a * a.transpose() + (1 - cos(theta)) / theta * skew_matrix(a);
    Matrix<double, 3, 1> t = J * rou;

    Matrix<double, 4, 4> exp_se3_matrix;
    exp_se3_matrix = MatrixXd::Zero(4, 4);
    exp_se3_matrix.block(0, 0, 3, 3) = R;
    exp_se3_matrix.block(0, 3, 3, 1) = t;
    exp_se3_matrix(3, 3) = 1;

    return exp_se3_matrix;
}

int main(){
    Matrix<double, 6, 1> original_se3;
    original_se3 << 0.695284, -0.233337, -2.37164, -0.0186706, 0.00502618, 0.00953036;
    Matrix<double, 6, 1> delta_se3;
    delta_se3 << -0.0442689, 0.0179693, -0.660739, -0.00172662, -0.00614563, 0.00706601;

    Matrix<double, 6, 1> new_se3 = se3_exp_LeftMutiply(delta_se3, original_se3);
    Matrix<double, 4, 4> exp_se3_matrix = se3_exp_ToMatrix(new_se3);

    Sophus::SE3 SE3_updated = Sophus::SE3::exp(delta_se3) * Sophus::SE3::exp(original_se3);
    Matrix<double, 6, 1> new_se3_sophus = SE3_updated.log();
    Matrix<double, 4, 4> exp_se3_matrix_sophus = SE3_updated.matrix();

    cout << "se3 my:" << endl;
    cout << new_se3 << endl;
    cout << "se3 exp matrix:" << endl;
    cout << exp_se3_matrix << endl;
    cout << "se3 sophus:" << endl;
    cout << new_se3_sophus << endl;
    cout << "se3 exp matrix sophus:" << endl;
    cout << exp_se3_matrix_sophus << endl;

    cout << endl;
    cout << se3_exp_ToMatrix(new_se3_sophus) << endl;

    return 0;
}