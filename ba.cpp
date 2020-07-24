#include<iostream>
#include<random>
#include<vector>
#include<Eigen/Core>
#include<Eigen/Dense>
#include<unsupported/Eigen/MatrixFunctions>
using namespace std;
using namespace Eigen;

class cal_error_F_E_camera
{
public:
    cal_error_F_E(double fx,double fy,double cx, double cy,int camera_id){ //fx、fy、cx、cy表示相机内参数，camera_id表示相机编号
        fx_ = fx;
        fy_ = fy;
        cx_ = cx;
        cy_ = cy;
        camera_id_ = camera_id;
    }

    void setinitialcamera_value(Matrix<double,6,1> q){ //q1对应se(3)平移向量，q2对应se(3)旋转向量,q表示se(3)
        q_ = q;
        q1_ = q.head(3);
        q2_ = q.tail(3);
    }

    Matrix3d cal_skew_matrix(Matrix<double,3,1> d){ //计算反对称矩阵
        Matrix3d m;
        m << 0, -d(2, 0), d(1, 0), d(2, 0), 0, -d(0, 0), -d(1, 0), d(0, 0), 0;
        return m;
    }

    void cal_R_t(){ //计算旋转矩阵R和平移向量t
        double theta;
        Matrix<double, 3, 1> a;
        Matrix3d J;
        theta = q2_.norm();
        a = q2_ / theta;
        J = sin(theta) / theta * MatrixXd::Identity(3, 3) + (1 - sin(theta) / theta) * a * a.transpose() + (1 - cos(theta)) / theta * cal_skew_matrix(a);
        R = cal_skew_matrix(q2_).exp();
        t = J * q1_;
    }

    Matrix<double,2,6> cal_F(Matrix<double,3,1> P){ //输入的是任意一个路标世界坐标系下的坐标
        Matrix<double, 3, 1> P_ = R * P + t; //路标这个相机坐标系下的坐标
        //误差是预测值减去观测值
        Matrix<double, 2, 3> J_eP_;
        J_eP_ << fx_ / P_(2, 0), 0, -(fx_ * P_(0, 0)) / (P_(2, 0) * P_(2, 0)), 0, fy_ / P_(2, 0), -(fy_ * P_(1, 0)) / (P_(2, 0) * P_(2, 0));
        Matrix<double, 3, 6> J_P_q;
        J_P_q << MatrixXd::Identity(3, 3), -cal_skew_matrix(P_);
        Matrix<double, 2, 6> J_eq;
        J_eq = J_eP_ * J_P_q;
        return J_eq;
    }

    Matrix<double,2,3> cal_E(Matrix<double,3,1> P){ //P是任意一个路标世界坐标系下的坐标(X,Y,Z)^T
        Matrix<double, 3, 1> P_ = R * P + t; //路标这个相机坐标系下的坐标
        //误差是预测值减去观测值
        Matrix<double, 2, 3> J_eP_;
        J_eP_ << fx_ / P_(2, 0), 0, -(fx_ * P_(0, 0)) / (P_(2, 0) * P_(2, 0)), 0, fy_ / P_(2, 0), -(fy_ * P_(1, 0)) / (P_(2, 0) * P_(2, 0));
        Matrix<double, 2, 3> J_eP;
        J_eP = J_eP_ * R;
        return J_eP;
    }

    double cal_error(Matrix<double,2,1> uv,Matrix<double,3,1> P){ //uv表示路标真实的像素坐标(u,v)^T,P是任意一个路标世界坐标系下的坐标(X,Y,Z)^T
        Matrix<double, 3, 1> P_ = R * P + t; //路标这个相机坐标系下的坐标
        //误差是预测值减去观测值
        Matrix<double, 2, 1> uv_; //uv_表示路标投影的像素坐标(u_,v_)^T
        uv_(0, 0) = fx_ * P_(0, 0) / P_(2, 0) + cx_;
        uv_(1, 0) = fy_ * P_(1, 0) / P_(2, 0) + cy_;
        return (uv_ - uv).squaredNorm();
    }

    double fx_, fy_, cx_, cy_;
    int camera_id_;
    Matrix<double, 3, 1> q1_, q2_;
    Matrix<double, 6, 1> q_;
    Matrix3d R;
    Matrix<double, 3, 1> t;
};

int main(){
    return 0;
}