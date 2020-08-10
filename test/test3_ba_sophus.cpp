#include<cstdio>
#include<iostream>
#include<fstream>
#include<random>
#include<vector>
#include<string>
#include<cmath>
#include<ctime>
#include<Eigen/Core>
#include<Eigen/Dense>
#include<unsupported/Eigen/MatrixFunctions>

#include "sophus/so3.h"
#include "sophus/se3.h"

using namespace std;
using namespace Eigen;

class load_data
{
public:
    load_data(int num_camera_parameters){
        num_camera_parameters_ = num_camera_parameters;
    }

    double* observations()       { return observations_;                   }
    double* parameter_cameras()  { return parameters_;                     }
    double* parameter_points()   { return parameters_  + num_camera_parameters_ * num_cameras_; }

    template<typename T>
    void fscanf_change(FILE* fp, const char* format, T* value){
        int flag = fscanf(fp, format, value);
        if(flag != 1){
            cout << "cannot scan data!" << endl;
        }
    }
    
    bool load_file(char* filename){
        FILE *fp = fopen(filename, "r");
        if(fp == NULL){
            return false;
        }

        fscanf_change(fp, "%d", &num_cameras_);
        fscanf_change(fp, "%d", &num_points_);
        fscanf_change(fp, "%d", &num_observations_);

        point_index_ = new int[num_observations_];
        camera_index_ = new int[num_observations_];
        observations_ = new double[2 * num_observations_];
        num_parameters_ = num_camera_parameters_ * num_cameras_ + 3 * num_points_;
        parameters_ = new double[num_parameters_];

        for (int i = 0; i < num_observations_;i++){
            fscanf_change(fp, "%d", camera_index_ + i);
            fscanf_change(fp, "%d", point_index_ + i);
            for (int j = 0; j < 2;j++){
                fscanf_change(fp, "%lf", observations_ + 2 * i + j);
            }
        }

        for (int i = 0; i < num_parameters_;i++){
            fscanf_change(fp, "%lf", parameters_ + i);
        }

        return true;
    }

    int num_camera_parameters_;
    int num_observations_;
    int num_points_;
    int num_cameras_;
    int num_parameters_;

    int* point_index_;
    int* camera_index_;
    double* observations_;
    double* parameters_;
};

class ReprojectionError
{
public:
    ReprojectionError(int camera_id, double fx, double fy, double cx, double cy, double k1, double k2, int point_id, double u, double v){
        camera_id_ = camera_id;
        fx_ = fx;
        fy_ = fy;
        cx_ = cx;
        cy_ = cy;
        k1_ = k1; //径向畸变参数k1
        k2_ = k2; //径向畸变参数k2
        point_id_ = point_id;
        distorted_u_ = u;
        distorted_v_ = v;
    }

    Matrix3d skew_matrix(Matrix<double,3,1> d){
        Matrix3d m;
        m << 0, -d(2, 0), d(1, 0), d(2, 0), 0, -d(0, 0), -d(1, 0), d(0, 0), 0;
        return m;
    }

    void undistort_points(){
        Matrix<double, 2, 1> u_v;
        u_v << distorted_u_, distorted_v_;
        Matrix<double, 2, 1> x1_y1, x2_y2, delta_x1_y1;
        double r;
        delta_x1_y1(0, 0) = 1000;
        delta_x1_y1(1, 0) = 1000;
        x1_y1(0, 0) = (u_v(0, 0) - cx_) / fx_;
        x1_y1(1, 0) = (u_v(1, 0) - cy_) / fy_;
        int step = 0;
        while (delta_x1_y1.norm() >= 1e-5){
            r = x1_y1.squaredNorm();
            x2_y2(0, 0) = x1_y1(0, 0) * (1.0 + r * (k1_ + k2_ * r));
            x2_y2(1, 0) = x1_y1(1, 0) * (1.0 + r * (k1_ + k2_ * r));

            Matrix<double, 2, 1> residual;
            residual(0, 0) = fx_ * x2_y2(0, 0) + cx_ - u_v(0, 0);
            residual(1, 0) = fy_ * x2_y2(1, 0) + cy_ - u_v(1, 0);

            Matrix<double, 2, 2> J_k;
            J_k(0, 0) = fx_ * (1 + 3 * k1_ * pow(x1_y1(0, 0), 2.0) + k1_ * pow(x1_y1(1, 0), 2.0) + 5 * k2_ * pow(x1_y1(0, 0), 4) + 6 * k2_ * pow(x1_y1(0, 0), 2.0) * pow(x1_y1(1, 0), 2.0) + k2_ * pow(x1_y1(1, 0), 4.0));
            J_k(0, 1) = fx_ * (2 * k1_ * x1_y1(0, 0) * x1_y1(1, 0) + 4 * k2_ * pow(x1_y1(0, 0), 3.0) * x1_y1(1, 0) + 4 * k2_ * x1_y1(0, 0) * pow(x1_y1(1, 0), 3.0));
            J_k(1, 0) = fy_ * (2 * k1_ * x1_y1(0, 0) * x1_y1(1, 0) + 4 * k2_ * pow(x1_y1(0, 0), 3.0) * x1_y1(1, 0) + 4 * k2_ * x1_y1(0, 0) * pow(x1_y1(1, 0), 3.0));
            J_k(1, 1) = fy_ * (1 + 3 * k1_ * pow(x1_y1(1, 0), 2.0) + k1_ * pow(x1_y1(0, 0), 2.0) + 5 * k2_ * pow(x1_y1(1, 0), 4) + 6 * k2_ * pow(x1_y1(0, 0), 2.0) * pow(x1_y1(1, 0), 2.0) + k2_ * pow(x1_y1(0, 0), 4.0));

            delta_x1_y1 = J_k.colPivHouseholderQr().solve(-residual);
            x1_y1 += delta_x1_y1;
            //cout << "residual = " << residual << endl;
            //cout << endl;
            step++;
        }
        undistorted_u_ = fx_ * x1_y1(0, 0) + cx_;
        undistorted_v_ = fy_ * x1_y1(1, 0) + cy_;
    }

    void cal_Rt_se(Matrix<double,6,1> camera_in){
        Matrix<double,3,1> fa = camera_in.head(3);
        R = skew_matrix(fa).exp();
        double theta = fa.norm();

        /*Matrix3d Omega = skew_matrix(fa);
        if (theta < 0.00001)
        {
          //TODO: CHECK WHETHER THIS IS CORRECT!!!
          R = (Matrix3d::Identity() + Omega + Omega*Omega);
        }
        else{
          Matrix3d Omega2 = Omega*Omega;

          R = (Matrix3d::Identity()
              + sin(theta)/theta *Omega
              + (1-cos(theta))/(theta*theta)*Omega2);
        }*/

        Matrix<double, 3, 1> a = fa / theta;
        //R = cos(theta) * Matrix3d::Identity() + (1 - cos(theta)) * a * a.transpose() + sin(theta) * skew_matrix(a);
        //R = AngleAxisd(theta, a).toRotationMatrix();
        Matrix3d J = sin(theta) / theta * Matrix3d::Identity() + (1 - sin(theta) / theta) * a * a.transpose() + (1 - cos(theta)) / theta * skew_matrix(a);
        t = camera_in.block(3, 0, 3, 1);
        //Matrix<double,3,1> rou = J.colPivHouseholderQr().solve(t);
        Matrix<double, 3, 1> rou = J.inverse() * t;
        //Sophus::SE3 SE3_Rt(R, t);
        //se << SE3_Rt.log();
        se << rou, fa;
    }

    Matrix<double,2,1> call_error(Matrix<double,3,1> P){
        Matrix<double, 3, 1> P_ = R * P + t;
        double P_x = P_(0, 0) / P_(2, 0);
        double P_y = P_(1, 0) / P_(2, 0);
        //double r = P_x * P_x + P_y * P_y;
        //double distortion = 1.0 + r * (k1_ + k2_ * r);
        //double predicted_u = fx_ * P_x * distortion + cx_;
        //double predicted_v = fy_ * P_y * distortion + cy_;
        double predicted_u = fx_ * P_x + cx_;
        double predicted_v = fy_ * P_y + cy_;
        Matrix<double, 2, 1> error;
        error << undistorted_u_ - predicted_u, undistorted_v_ - predicted_v; //观测值减去预测值
        return error;
    }

    Matrix<double,2,9> cal_FE(Matrix<double,3,1> P){
        Matrix<double, 3, 1> P_ = R * P + t;
        Matrix<double, 2, 3> J_eP_;
        J_eP_ << -fx_ / P_(2, 0), 0, fx_ * P_(0, 0) / (P_(2, 0) * P_(2, 0)), 0, -fy_ / P_(2, 0), fy_ * P_(1, 0) / (P_(2, 0) * P_(2, 0));
        Matrix<double, 3, 6> J_P_;
        J_P_ << Matrix3d::Identity(), -skew_matrix(P_);
        Matrix<double, 2, 6> F = J_eP_ * J_P_;
        Matrix<double, 2, 3> E = J_eP_ * R;
        Matrix<double, 2, 9> EF;
        EF << F, E;
        return EF;
    }

    int camera_id_, point_id_;
    double fx_, fy_, cx_, cy_, distorted_u_, distorted_v_, k1_, k2_, undistorted_u_, undistorted_v_;
    Matrix<double, 6, 1> se;
    Matrix3d R;
    Matrix<double, 3, 1> t;
};

class LM_GN_SchurOptimization
{
public:
    LM_GN_SchurOptimization(int num_iterations, int num_cameras, int num_points, int num_camera_parameters, double lambda, double* parameter_cameras, double* parameter_points, string method){
        num_iterations_ = num_iterations;
        num_cameras_ = num_cameras;
        num_points_ = num_points;
        num_camera_parameters_ = num_camera_parameters;
        lambda_ = lambda;
        parameter_cameras_ = parameter_cameras;
        parameter_points_ = parameter_points;
        method_ = method;
    }

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

    void optimize(){
        MatrixXd B(6 * num_cameras_, 6 * num_cameras_);
        B = MatrixXd::Zero(6 * num_cameras_, 6 * num_cameras_);

        vector<Matrix3d> C(num_points_); //储存C每个point(3, 3)的子矩阵
        vector<Matrix3d> C_inverse(num_points_); //储存C每个point(3, 3)的逆子矩阵
        vector<MatrixXd> E(num_points_); //储存E列分每个point(6 * num_cameras_, 3)的逆子矩阵
        vector<MatrixXd> E_T(num_points_); //储存E_T行分每个point(3, 6 * num_cameras_)的逆子矩阵
        //赋初值0矩阵
        for (int i = 0; i < num_points_;i++){
            C[i] = Matrix3d::Zero();
            C_inverse[i] = Matrix3d::Zero();
            E[i] = MatrixXd::Zero(6 * num_cameras_, 3);
            E_T[i] = MatrixXd::Zero(3, 6 * num_cameras_);
        }

        MatrixXd v(6 * num_cameras_, 1);
        v = MatrixXd::Zero(6 * num_cameras_, 1);
        MatrixXd w(3 * num_points_, 1);
        w = MatrixXd::Zero(3 * num_points_, 1);
        MatrixXd parameter_se(6, num_cameras_);
        parameter_se = MatrixXd::Zero(6, num_cameras_);

        /*if(method_ == "GN"){
            lambda_ = 10;
        }*/

        FILE *ff = fopen("/home/vision/Desktop/code_c_c++/my_BA/log/log.txt","w");
        cout << "number of cameras: " << num_cameras_ << " ,number of points: " << num_points_ << " ,number of observations: " << (int)terms.size() << " ,method: " << method_ << " ,initial lambda: " << lambda_ << endl;
        fprintf(ff, "number of cameras: %d ,number of points: %d ,number of observations: %d, method: %s, initial lambda: %lf\n\n", num_cameras_, num_points_, (int)terms.size(), method_.c_str(), lambda_);

        double total_time_consumption = 0;
        for (int i = 0; i < num_iterations_; i++){
            clock_t time_stt = clock(); //计时
            double error_sum = 0;
            for (int j = 0; j < (int)terms.size(); j++){
                int camera_id = terms[j]->camera_id_;
                int point_id = terms[j]->point_id_;

                Matrix<double, 6, 1> camera_in;
                for (int k = 0; k < 6;k++){
                    camera_in(k, 0) = *(parameter_cameras_ + camera_id * num_camera_parameters_ + k);
                }
                Matrix<double, 3, 1> P;
                for (int k = 0; k < 3;k++){
                    P(k, 0) = *(parameter_points_ + point_id * 3 + k);
                }

                terms[j]->undistort_points(); //对真实的2d观测去畸变
                terms[j]->cal_Rt_se(camera_in); //计算R，t和se
                parameter_se.col(camera_id) = terms[j]->se; //储存se
                Matrix<double, 2, 9> J_EF = terms[j]->cal_FE(P); //计算E，F
                Matrix<double, 2, 6> J_F = J_EF.block(0, 0, 2, 6);
                Matrix<double, 2, 3> J_E = J_EF.block(0, 6, 2, 3);
                Matrix<double, 2, 1> error = terms[j]->call_error(P); //计算误差（观测值减去预测值）

                fprintf(ff, "camera id = %d, point id = %d, error_u = %lf, error_v = %lf\n", terms[j]->camera_id_, terms[j]->point_id_, error(0, 0), error(1, 0));
                error_sum += 0.5 * error.squaredNorm();

                Matrix<double, 6, 6> J_FTJ_F = J_F.transpose() * J_F;
                Matrix<double, 6, 6> D_FTF_D = MatrixXd(J_FTJ_F.diagonal().asDiagonal());
                Matrix<double, 3, 3> J_ETJ_E = J_E.transpose() * J_E;
                Matrix<double, 3, 3> D_ETE_D = MatrixXd(J_ETJ_E.diagonal().asDiagonal());

                if(method_ == "LM"){
                    B.block(camera_id * 6, camera_id * 6, 6, 6) += J_FTJ_F + lambda_ * D_FTF_D;
                    C[point_id] += J_ETJ_E + lambda_ * D_ETE_D;
                }

                if(method_ == "GN"){
                    B.block(camera_id * 6, camera_id * 6, 6, 6) += lambda_ * J_FTJ_F;
                    C[point_id] += lambda_ * J_ETJ_E;
                }
                
                E[point_id].block(camera_id * 6, 0, 6, 3) += J_F.transpose() * J_E;
                E_T[point_id].block(0, camera_id * 6, 3, 6) += J_E.transpose() * J_F;
                v.block(camera_id * 6, 0, 6, 1) += -J_F.transpose() * error;
                w.block(point_id * 3, 0, 3, 1) += -J_E.transpose() * error;
            }

            if(i == 0){
                initial_error = error_sum;
                final_error = error_sum;
            }

            //计算delta_parameter_cameras
            MatrixXd E_C_inverse_E_T(6 * num_cameras_, 6 * num_cameras_); //储存E * C^(-1) * E^T
            E_C_inverse_E_T = MatrixXd::Zero(6 * num_cameras_, 6 * num_cameras_);
            MatrixXd E_C_inverse_w(6 * num_cameras_, 1); //储存E * C^(-1) * w
            E_C_inverse_w = MatrixXd::Zero(6 * num_cameras_, 1);
            for (int j = 0; j < num_points_; j++){
                C_inverse[j] = C[j].inverse();
                E_C_inverse_E_T += (E[j] * C_inverse[j] * E_T[j]);
                E_C_inverse_w += (E[j] * C_inverse[j] * w.block(3 * j, 0, 3, 1));
            }
            MatrixXd delta_parameter_cameras(6 * num_cameras_, 1);
            //delta_parameter_cameras = (B - E_C_inverse_E_T).colPivHouseholderQr().solve(v - E_C_inverse_w);
            delta_parameter_cameras = (B - E_C_inverse_E_T).ldlt().solve(v - E_C_inverse_w);

            //计算delta_parameter_points
            MatrixXd E_T_delta_parameter_cameras(3 * num_points_, 1);
            E_T_delta_parameter_cameras = MatrixXd::Zero(3 * num_points_, 1);
            for (int j = 0; j < num_points_; j++){
                E_T_delta_parameter_cameras.block(3 * j, 0, 3, 1) = E_T[j] * delta_parameter_cameras;
            }
            MatrixXd delta_parameter_points(3 * num_points_, 1);
            delta_parameter_points = MatrixXd::Zero(3 * num_points_, 1);
            for (int j = 0; j < num_points_; j++){
                delta_parameter_points.block(3 * j, 0, 3, 1) = C_inverse[j] * (w - E_T_delta_parameter_cameras).block(3 * j, 0, 3, 1);
            }

            if(delta_parameter_points.norm() + delta_parameter_cameras.norm() < 1e-10){
                break;
            }

            MatrixXd parameter_cameras_new(6, num_cameras_); //临时储存更新后的fa和t值
            parameter_cameras_new = MatrixXd::Zero(6, num_cameras_);

            for (int j = 0; j < num_cameras_; j++){
                //Sophus::SE3 SE3_updated = Sophus::SE3::exp(delta_parameter_cameras.block(6 * j, 0, 6, 1)) * Sophus::SE3::exp(parameter_se.col(j));
                //parameter_cameras_new.col(j).block(0, 0, 3, 1) = SE3_updated.log().block(3, 0, 3, 1);
                //parameter_cameras_new.col(j).block(3, 0, 3, 1) = SE3_updated.matrix().block(0, 3, 3, 1);
                Matrix<double, 6, 1> SE3_updated = se3_exp_LeftMutiply(delta_parameter_cameras.block(6 * j, 0, 6, 1), parameter_se.col(j));
                Matrix<double, 4, 4> SE3_updated_ToMatrix = se3_exp_ToMatrix(SE3_updated);
                parameter_cameras_new.col(j).block(0, 0, 3, 1) = SE3_updated.block(3, 0, 3, 1);
                parameter_cameras_new.col(j).block(3, 0, 3, 1) = SE3_updated_ToMatrix.block(0, 3, 3, 1);
                /*if(j == 0){
                    cout << terms[0]->R << endl;
                    cout << (Sophus::SE3::exp(parameter_se.col(j))).matrix() << endl;
                }*/
            }

            double error_sum_next = 0;
            for (int j = 0; j < (int)terms.size();j++){
                int camera_id = terms[j]->camera_id_;
                int point_id = terms[j]->point_id_;

                Matrix<double, 6, 1> camera_in = parameter_cameras_new.col(camera_id);
                Matrix<double, 3, 1> P;
                for (int k = 0; k < 3;k++){
                    P(k, 0) = *(parameter_points_ + point_id * 3 + k) + delta_parameter_points(3 * point_id + k, 0);
                }

                //terms[j]->undistort_points(); //对真实的2d观测去畸变
                terms[j]->cal_Rt_se(camera_in);
                Matrix<double, 2, 1> error = terms[j]->call_error(P);
                error_sum_next += 0.5 * error.squaredNorm();
            }

            //LM method 更新参数
            if(method_ == "LM"){
                if(error_sum_next < error_sum){
                    for (int j = 0; j < num_cameras_;j++){
                        for (int k = 0; k < 6;k++){
                            *(parameter_cameras_ + j * num_camera_parameters_ + k) = parameter_cameras_new(k, j);
                        }
                    }
                    for (int j = 0; j < num_points_;j++){
                        for (int k = 0; k < 3;k++){
                            *(parameter_points_ + j * 3 + k) += delta_parameter_points(j * 3 + k, 0);
                        }
                    }
                    final_error = error_sum_next;
                    lambda_ /= 10;
                    total_time_consumption += (clock() - time_stt) / (double)CLOCKS_PER_SEC;
                    cout << "successful step! iteration = " << i << ", error before = " << error_sum << ", error next = " << error_sum_next << " , lambda = " << lambda_ << " , time consumption = " << (clock() - time_stt) / (double)CLOCKS_PER_SEC << "s" << endl;
                    fprintf(ff, "successful step! iteration = %d, error before = %lf, error next = %lf, lambda = %lf, time consumption = %lfs\n\n", i, error_sum, error_sum_next, lambda_, 1000 * (clock() - time_stt) / (double)CLOCKS_PER_SEC);
                }else{
                    lambda_ *= 10;
                    total_time_consumption += (clock() - time_stt) / (double)CLOCKS_PER_SEC;
                    cout << "unsuccessful step! iteration = " << i << ", error before = " << error_sum << ", error next = " << error_sum_next << " , lambda = " << lambda_ << " , time consumption = " << (clock() - time_stt) / (double)CLOCKS_PER_SEC << "s" << endl;
                    fprintf(ff, "unsuccessful step! iteration = %d, error before = %lf, error next = %lf, lambda = %lf, time consumption = %lfs\n\n", i, error_sum, error_sum_next, lambda_, (clock() - time_stt) / (double)CLOCKS_PER_SEC);
                }
            }

            //GN method 更新参数 lambda = 0
            if(method_ == "GN"){
                if(error_sum_next < error_sum){
                    for (int j = 0; j < num_cameras_;j++){
                        for (int k = 0; k < 6;k++){
                            *(parameter_cameras_ + j * num_camera_parameters_ + k) = parameter_cameras_new(k, j);
                        }
                    }
                    for (int j = 0; j < num_points_;j++){
                        for (int k = 0; k < 3;k++){
                            *(parameter_points_ + j * 3 + k) += delta_parameter_points(3 * j + k, 0);
                        }
                    }
                    final_error = error_sum_next;
                    lambda_ /= 10;
                    total_time_consumption += (clock() - time_stt) / (double)CLOCKS_PER_SEC;
                    cout << "successful step! iteration = " << i << ", error before = " << error_sum << ", error next = " << error_sum_next << " , lambda = " << lambda_ << " , time consumption = " << (clock() - time_stt) / (double)CLOCKS_PER_SEC << "s" << endl;
                    fprintf(ff, "successful step! iteration = %d, error before = %lf, error next = %lf, lambda = %lf, time consumption = %lfs\n\n", i, error_sum, error_sum_next, lambda_, (clock() - time_stt) / (double)CLOCKS_PER_SEC);
                }else{
                    lambda_ *= 10;
                    total_time_consumption += (clock() - time_stt) / (double)CLOCKS_PER_SEC;
                    cout << "unsuccessful step! iteration = " << i << ", error before = " << error_sum << ", error next = " << error_sum_next << " , lambda = " << lambda_ << " , time consumption = " << (clock() - time_stt) / (double)CLOCKS_PER_SEC << "s" << endl;
                    fprintf(ff, "unsuccessful step! iteration = %d, error before = %lf, error next = %lf, lambda = %lf, time consumption = %lfs\n\n", i, error_sum, error_sum_next, lambda_, (clock() - time_stt) / (double)CLOCKS_PER_SEC);
                }
            }
        }
        cout << "initial error = " << initial_error << " ,final error = " << final_error << " ,error change = " << initial_error - final_error << " ,total time consumption = " << total_time_consumption << "s" << endl;
        fprintf(ff, "initial error = %lf, final error = %lf, error change = %lf, total time consumption = %lfs\n", initial_error, final_error, initial_error - final_error, total_time_consumption);
        fclose(ff);
    }

    void add_errorterms(ReprojectionError *e)
    {
        terms.push_back(e);
    }

    int num_iterations_, num_cameras_, num_points_, num_camera_parameters_;
    double lambda_, initial_error = 0, final_error = 0;
    double *parameter_cameras_;
    double *parameter_points_;
    string method_;
    vector<ReprojectionError *> terms;
};


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

typedef Map<VectorXd> VectorRef;
typedef Map<const VectorXd> ConstVectorRef;

void CameraToAngleAxisAndCenter(double* camera, double* angle_axis, double* center, load_data* f){
    VectorRef angle_axis_ref(angle_axis, 3);
    angle_axis_ref = ConstVectorRef(camera, 3);

    VectorXd inverse_rotation = -angle_axis_ref;
    AngleAxisRotatePoint_byhand(inverse_rotation, camera + f->num_camera_parameters_ - 6, center);

    VectorRef(center, 3) *= -1.0;  
}

void WriteToPLYFile(const char* filename, load_data* f){
    //创建一个文件输出流对象，输出至文件名
    ofstream of(filename);
    //uchar 类型只能保存0-255的数字溢出后会从0开始继续累加
    of << "ply"
       << '\n' << "format ascii 1.0"
       << '\n' << "element vertex " << f->num_cameras_ + f->num_points_
       << '\n' << "property double x"
       << '\n' << "property double y"
       << '\n' << "property double z"
       << '\n' << "property uchar red"
       << '\n' << "property uchar green"
       << '\n' << "property uchar blue"
       << '\n' << "end_header" << std::endl;

    double angle_axis[3];
    double center[3];
    for (int i = 0; i < f->num_cameras_;i++){
        double *camera = f->parameter_cameras() + i * f->num_camera_parameters_;
        CameraToAngleAxisAndCenter(camera, angle_axis, center, f);
        of << center[0] << ' ' << center[1] << ' ' << center[2]
           << " 0 255 0" << '\n';
    }

    const double *points = f->parameter_points();
    for (int i = 0; i < f->num_points_;i++){
        const double *point = points + i * 3;
        for (int j = 0; j < 3;j++){
            of << *(point + j) << ' ';
        }
        of << "255 255 255\n";
    }
    of.close();
}

int main(int argc, char** argv){
    /*if(argc != 2){
        cout << "error input!" << endl;
    }*/

    int num_camera_parameters = 9; //相机参数个数（按照顺序依次：旋转向量、平移向量、相机焦距（fx=fy）、径向畸变系数k1、径向畸变系数k2）

    //load_data(int num_camera_parameters)
    load_data f(num_camera_parameters); //初始化

    if(!f.load_file("/home/vision/Desktop/code_c_c++/my_BA/opt_data/mini_data.txt")){
        cout << "unable to open file!" << endl;
    }

    //LM_SchurOptimization(int num_iterations, int num_cameras, int num_points, int num_camera_parameters, double lambda, double* parameter_cameras, double* parameter_points, string method)
    LM_GN_SchurOptimization opt(50, f.num_cameras_, f.num_points_, num_camera_parameters, 1e-4, f.parameter_cameras(), f.parameter_points(), "LM"); //"GN":lambda取１(1)，"LM":lambda取小于１(1e-5、1e-4)
    for (int i = 0; i < f.num_observations_;i++){
        //ReprojectionError(double camera_id, double fx, double fy, double cx, double cy, double k1, double k2, double point_id, double u, double v)
        //cout << f.camera_index_[i] << endl;
        double fx = *(f.parameter_cameras() + f.camera_index_[i] * num_camera_parameters + 6);
        double fy = fx;
        double cx = 0;
        double cy = 0;
        double k1 = *(f.parameter_cameras() + f.camera_index_[i] * num_camera_parameters + 7);
        double k2 = *(f.parameter_cameras() + f.camera_index_[i] * num_camera_parameters + 8);
        double u = *(f.observations() + 2 * i);
        double v = *(f.observations() + 2 * i + 1);
        ReprojectionError *e = new ReprojectionError(f.camera_index_[i], fx, fy, cx, cy, k1, k2, f.point_index_[i], u ,v);
        opt.add_errorterms(e);
    }

    //开始优化
    opt.optimize();

    FILE* fp = fopen("/home/vision/Desktop/code_c_c++/my_BA/update_parameter/update.txt", "w");
    for (int i = 0; i < f.num_parameters_;i++){
        fprintf(fp, "%.16e\n", *(f.parameters_ + i));
    }
    fclose(fp);

    FILE *fpp = fopen("/home/vision/Desktop/code_c_c++/my_BA/3d_point_update.txt", "w");
    for (int i = 0; i < f.num_points_ * 3;i++){
        fprintf(fpp, "%.16e\n", *(f.parameter_points() + i));
    }
    fclose(fpp);

    WriteToPLYFile("/home/vision/Desktop/code_c_c++/my_BA/3d_point_update.ply", &f);

    return 0;
}