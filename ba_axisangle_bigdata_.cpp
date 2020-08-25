#include<cstdio>
#include<iostream>
#include<fstream>
#include<random>
#include<vector>
#include<string>
#include<cmath>
#include<ctime>
#include<unistd.h>

#include<Eigen/Core>
#include<Eigen/Dense>
#include<unsupported/Eigen/MatrixFunctions>

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
        Matrix3d J = sin(theta) / theta * Matrix3d::Identity() + (1 - sin(theta) / theta) * a * a.transpose() + (1 - cos(theta)) / theta * skew_matrix(a);
        t = camera_in.block(3, 0, 3, 1);
        //Matrix<double,3,1> rou = J.colPivHouseholderQr().solve(t);
        Matrix<double, 3, 1> rou = J.inverse() * t;
        se << rou, fa;
    }

    Matrix<double,2,1> call_error(Matrix<double,3,1> P){
        Matrix<double, 3, 1> P_ = R * P + t;
        double P_x = P_(0, 0) / P_(2, 0);
        double P_y = P_(1, 0) / P_(2, 0);
        double predicted_u = fx_ * P_x + cx_;
        double predicted_v = fy_ * P_y + cy_;
        Matrix<double, 2, 1> error;
        error << (undistorted_u_ - predicted_u), (undistorted_v_ - predicted_v); //观测值减去预测值
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
    LM_GN_SchurOptimization(int num_iterations, int num_cameras, int num_points, int num_camera_parameters, double lambda, double* parameter_cameras, double* parameter_points, string iter_method, string solve_method){
        num_iterations_ = num_iterations;
        num_cameras_ = num_cameras;
        num_points_ = num_points;
        num_camera_parameters_ = num_camera_parameters;
        lambda_ = lambda;
        parameter_cameras_ = parameter_cameras;
        parameter_points_ = parameter_points;
        iter_method_ = iter_method;
        solve_method_ = solve_method;
    }

    Matrix3d skew_matrix(Matrix<double,3,1> d){
        Matrix3d m;
        m << 0, -d(2, 0), d(1, 0), d(2, 0), 0, -d(0, 0), -d(1, 0), d(0, 0), 0;
        return m;
    }

    Matrix<double,6,1> se3_exp_LeftMutiply(Matrix<double,6,1> delta_se3, Matrix<double,6,1> original_se3){
        //返回的是se(3)的形式
        Matrix<double, 3, 1> rou = original_se3.block<3, 1>(0, 0);
        Matrix<double, 3, 1> fa = original_se3.block<3, 1>(3, 0);
        double theta = fa.norm();
        Matrix<double, 3, 1> a = fa / theta;

        Matrix<double, 3, 3> Jl_inverse = (theta / 2) * (1 / tan(theta / 2)) * Matrix3d::Identity() + (1 - (theta / 2) * (1 / tan(theta / 2))) * a * a.transpose() - (theta / 2) * skew_matrix(a);
        Matrix<double, 3, 3> Ql = 0.5 * skew_matrix(rou) + ((theta - sin(theta)) / pow(theta, 3.0)) * (skew_matrix(fa) * skew_matrix(rou) + skew_matrix(rou) * skew_matrix(fa) + skew_matrix(fa) * skew_matrix(rou) * skew_matrix(fa))
                                + ((pow(theta, 2.0) + 2 * cos(theta) - 2) / (2 * pow(theta, 4.0))) * (skew_matrix(fa) * skew_matrix(fa) * skew_matrix(rou) + skew_matrix(rou) * skew_matrix(fa) * skew_matrix(fa) - 3 * skew_matrix(fa) * skew_matrix(rou) * skew_matrix(fa))
                                + ((2 * theta - 3 * sin(theta) + theta * cos(theta)) / (2 * pow(theta, 5.0))) * (skew_matrix(fa) * skew_matrix(rou) * skew_matrix(fa) * skew_matrix(fa) + skew_matrix(fa) * skew_matrix(fa) * skew_matrix(rou) * skew_matrix(fa));

        Matrix<double, 6, 6> Fl_inverse = MatrixXd::Zero(6, 6);
        Fl_inverse.block<3, 3>(0, 0) = Jl_inverse;
        Fl_inverse.block<3, 3>(0, 3) = -Jl_inverse * Ql * Jl_inverse;
        Fl_inverse.block<3, 3>(3, 3) = Jl_inverse;

        Matrix<double, 6, 1> new_se3 = Fl_inverse * delta_se3 + original_se3;

        return new_se3;
    }

    Matrix<double,4,4> se3_exp_ToMatrix(Matrix<double,6,1> se3){
        Matrix<double, 3, 1> fa = se3.block<3, 1>(3, 0);
        Matrix<double, 3, 1> rou = se3.block<3, 1>(0, 0);
        double theta = fa.norm();
        Matrix<double, 3, 1> a = fa / theta;
        Matrix<double, 3, 3> R = cos(theta) * Matrix3d::Identity() + (1 - cos(theta)) * a * a.transpose() + sin(theta) * skew_matrix(a);
        Matrix<double, 3, 3> J = sin(theta) / theta * Matrix3d::Identity() + (1 - sin(theta) / theta) * a * a.transpose() + (1 - cos(theta)) / theta * skew_matrix(a);
        Matrix<double, 3, 1> t = J * rou;

        Matrix<double, 4, 4> exp_se3_matrix;
        exp_se3_matrix = MatrixXd::Zero(4, 4);
        exp_se3_matrix.block<3, 3>(0, 0) = R;
        exp_se3_matrix.block<3, 1>(0, 3) = t;
        exp_se3_matrix(3, 3) = 1;

        return exp_se3_matrix;
    }

    //计算矩阵的无穷范数的函数
    double cal_MatrixInfNorm(MatrixXd K){
        double inf_norm = -1000000000;
        //cout << sqrt(K.size()) << endl;
        for (int i = 0; i < (int)(sqrt(K.size())); i++){
            double row_abs_sum = 0;
            for (int j = 0; j < (int)(sqrt(K.size())); j++){
                row_abs_sum += fabs(K(i, j));
            }
            if(inf_norm < row_abs_sum){
                inf_norm = row_abs_sum;
                //cout << inf_norm << endl;
            }
        }
        return inf_norm;
    }

    //计算矩阵的2范数的函数
    double cal_Matrix2Norm(MatrixXd K){
        EigenSolver<MatrixXd> eigen_solver(K.transpose() * K);
        MatrixXd eigen_values = eigen_solver.pseudoEigenvalueMatrix();
        double max_eigen_value = eigen_values(0, 0);

        for (int i = 0; i < (int)(sqrt(K.size()));i++){
            if(eigen_values(i, i) > max_eigen_value){
                max_eigen_value = eigen_values(i, i);
            }
        }
        //cout << max_eigen_value << endl;
        double two_norm = sqrt(max_eigen_value);
        //cout << two_norm << endl;
        return two_norm;
    }

    void optimize(){
        MatrixXd B(6 * num_cameras_, 6 * num_cameras_); //相机参数的Jacobian矩阵
        vector<Matrix3d> C(num_points_); //储存C每个point(3, 3)的子矩阵
        vector<Matrix3d> C_inverse(num_points_); //储存C每个point(3, 3)的逆子矩阵
        vector<MatrixXd> E(num_points_); //储存E列分每个point(6 * num_cameras_, 3)的逆子矩阵
        vector<MatrixXd> E_T(num_points_); //储存E_T行分每个point(3, 6 * num_cameras_)的逆子矩阵
        MatrixXd v(6 * num_cameras_, 1); //增量方程右边相机部分
        MatrixXd w(3 * num_points_, 1); //增量方程左边3d观测点部分
        MatrixXd parameter_se(6, num_cameras_); //储存每个相机内参李代数形式se(3)

        //FILE *ff = fopen("/home/vision/Desktop/code_c_c++/my_BA/log/log.txt","w");
        cout << "number of cameras: " << num_cameras_ << " , number of points: " << num_points_ << " , number of observations: " << (int)terms.size() << " , iteration method: " << iter_method_ << " , initial lambda: " << lambda_ << " , solve equations method: " << solve_method_ << endl;
        //fprintf(ff, "number of cameras: %d , number of points: %d , number of observations: %d , iteration method: %s , initial lambda: %lf , solve equations method: %s\n\n", num_cameras_, num_points_, (int)terms.size(), iter_method_.c_str(), lambda_, solve_method_.c_str());

        //FILE *fff = fopen("/home/vision/Desktop/code_c_c++/my_BA/result_data/data.txt","w");
        //FILE *ffff = fopen("/home/vision/Desktop/code_c_c++/my_BA/result_data/cond.txt", "w");

        double total_time_consumption = 0;
        int total_iterations = 0;
        int successful_iterations = 0;
        for (int i = 0; i < num_iterations_; i++){
            clock_t time_stt = clock(); //计时
            total_iterations++;

            //赋初值0
            B.setZero(6 * num_cameras_, 6 * num_cameras_);
            for (int j = 0; j < num_points_; j++){
                C[j].setZero(3, 3);
                C_inverse[j].setZero(3, 3);
                E[j].setZero(6 * num_cameras_, 3);
                E_T[j].setZero(3, 6 * num_cameras_);
            }
            v.setZero(6 * num_cameras_, 1);
            w.setZero(3 * num_points_, 1);
            //parameter_se.setZero(6, num_cameras_);
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

                if(i == 0){
                    terms[j]->undistort_points(); //对真实的2d观测去畸变，只需去畸变一次
                }
                
                terms[j]->cal_Rt_se(camera_in); //计算R，t和se
                parameter_se.col(camera_id) = terms[j]->se; //储存se
                Matrix<double, 2, 9> J_EF = terms[j]->cal_FE(P); //计算E，F
                Matrix<double, 2, 6> J_F = J_EF.block<2, 6>(0, 0);
                Matrix<double, 2, 3> J_E = J_EF.block<2, 3>(0, 6);
                Matrix<double, 2, 1> error = terms[j]->call_error(P); //计算误差（观测值减去预测值）

                //fprintf(ff, "camera id = %d , point id = %d , error_u = %lf , error_v = %lf\n", terms[j]->camera_id_, terms[j]->point_id_, error(0, 0), error(1, 0));
                error_sum += 0.5 * error.squaredNorm();

                Matrix<double, 6, 6> J_FTJ_F = J_F.transpose() * J_F;
                Matrix<double, 6, 6> D_FTF_D = MatrixXd(J_FTJ_F.diagonal().asDiagonal());
                Matrix<double, 3, 3> J_ETJ_E = J_E.transpose() * J_E;
                Matrix<double, 3, 3> D_ETE_D = MatrixXd(J_ETJ_E.diagonal().asDiagonal());

                if(iter_method_ == "LM"){
                    B.block<6, 6>(camera_id * 6, camera_id * 6) += J_FTJ_F + lambda_ * D_FTF_D;
                    C[point_id] += J_ETJ_E + lambda_ * D_ETE_D;
                }

                if(iter_method_ == "LM_TR"){
                    if(i == 0 && flag == true){
                        B.block<6, 6>(camera_id * 6, camera_id * 6) += J_FTJ_F;
                        C[point_id] += J_ETJ_E;
                    }else{
                        B.block<6, 6>(camera_id * 6, camera_id * 6) += J_FTJ_F + lambda_ * D_FTF_D;
                        C[point_id] += J_ETJ_E + lambda_ * D_ETE_D;
                    }
                }

                if(iter_method_ == "GN"){
                    B.block<6, 6>(camera_id * 6, camera_id * 6) += lambda_ * J_FTJ_F;
                    C[point_id] += lambda_ * J_ETJ_E;
                }
                
                E[point_id].block<6, 3>(camera_id * 6, 0) += J_F.transpose() * J_E;
                E_T[point_id].block<3, 6>(0, camera_id * 6) += J_E.transpose() * J_F;
                v.block<6, 1>(camera_id * 6, 0) += -J_F.transpose() * error;
                w.block<3, 1>(point_id * 3, 0) += -J_E.transpose() * error;
            }
            cout << "time1 = " << (clock() - time_stt) / (double)CLOCKS_PER_SEC << "s" << endl;

            if(i == 0){
                initial_error = error_sum;
                final_error = error_sum;

                if(iter_method_ == "LM_TR" && flag == true){
                    v_ = 2;
                    maxa_ = B(0, 0);
                    for (int j = 0; j < num_cameras_; j++){
                        if(maxa_ < B(j, j)){
                            maxa_ = B(j, j);
                        }
                    }

                    for (int j = 0; j < num_points_;j++){
                        for (int k = 0; k < 3;k++){
                            if(maxa_ < C[j](k, k)){
                                maxa_ = C[j](k, k);
                            }
                        }
                    }
                    //cout << maxa_ << endl;
                    lambda_ *= maxa_;
                    //cout << lambda_ << endl;
                    i--;
                    flag = false;
                    continue;
                }
            }

            //计算delta_parameter_cameras
            MatrixXd E_C_inverse_E_T(6 * num_cameras_, 6 * num_cameras_); //储存E * C^(-1) * E^T
            E_C_inverse_E_T.setZero(6 * num_cameras_, 6 * num_cameras_);
            MatrixXd E_C_inverse_w(6 * num_cameras_, 1); //储存E * C^(-1) * w
            E_C_inverse_w.setZero(6 * num_cameras_, 1);

            /*for (int j = 0; j < num_points_; j++){
                C_inverse[j] = C[j].inverse();
                E_C_inverse_E_T += (E[j] * C_inverse[j] * E_T[j]);
                E_C_inverse_w += (E[j] * C_inverse[j] * w.block<3, 1>(3 * j, 0));
            }`*/

            /*for (int j = 0; j < num_points_; j++){
                C_inverse[j] = C[j].inverse();
            }*/
            cout << "time2 = " << (clock() - time_stt) / (double)CLOCKS_PER_SEC << "s" << endl;

            for (int j = 0; j < num_points_; j++){
                C_inverse[j] = C[j].inverse();
                for (int k1 = 0; k1 < num_cameras_;k1++){
                    E[j].block<6, 3>(6 * k1, 0) *= C_inverse[j];
                    for (int k2 = 0; k2 < num_cameras_; k2++){
                        E_C_inverse_E_T.block<6, 6>(6 * k1, 6 * k2) += E[j].block<6, 3>(6 * k1, 0) * E_T[j].block<3, 6>(0, 6 * k2);
                    }
                    E_C_inverse_w.block<6, 1>(6 * k1, 0) += E[j].block<6, 3>(6 * k1, 0) * w.block<3, 1>(3 * j, 0);
                }
            }
            cout << "time3 = " << (clock() - time_stt) / (double)CLOCKS_PER_SEC << "s" << endl;

            MatrixXd delta_parameter_cameras(6 * num_cameras_, 1);

            /*LDL算法*/
            if(solve_method_ == "LDL"){
                delta_parameter_cameras = (B - E_C_inverse_E_T).ldlt().solve(v - E_C_inverse_w);
            }

            /*QR算法*/
            if(solve_method_ == "QR"){
                delta_parameter_cameras = (B - E_C_inverse_E_T).colPivHouseholderQr().solve(v - E_C_inverse_w);
            }

            /*共轭梯度算法*/
            if(solve_method_ == "CG"){
                delta_parameter_cameras.setZero(6 * num_cameras_, 1); //迭代初始值0

                MatrixXd r_k(6 * num_cameras_, 1);
                MatrixXd d_k(6 * num_cameras_, 1);
                r_k.setZero(6 * num_cameras_, 1);
                d_k.setZero(6 * num_cameras_, 1);

                r_k = (B - E_C_inverse_E_T) * delta_parameter_cameras - (v - E_C_inverse_w);
                d_k = -r_k;
                MatrixXd r_1(6 * num_cameras_, 1);
                r_1 = r_k;

                int step = 1;
                while ((r_k.transpose() * r_k)(0, 0) / (r_1.transpose() * r_1)(0, 0) > 1e-8){
                    step++;
                    double a_k = (r_k.transpose() * r_k)(0, 0) / (d_k.transpose() * (B - E_C_inverse_E_T) * d_k)(0, 0);
                    delta_parameter_cameras += a_k * d_k;
                    double r_kTr_k = (r_k.transpose() * r_k)(0, 0);
                    r_k += a_k * (B - E_C_inverse_E_T) * d_k;
                    double b_k = (r_k.transpose() * r_k)(0, 0) / r_kTr_k;
                    d_k = -r_k + b_k * d_k;
                }
                cout << "total step = " << step << endl;
                total_step += step;
            }

            /*预处理共轭梯度算法(Jacobi preconditioner)*/
            if(solve_method_ == "PCG-J"){
                delta_parameter_cameras.setZero(6 * num_cameras_, 1); //迭代初始值0

                MatrixXd r_k(6 * num_cameras_, 1);
                MatrixXd d_k(6 * num_cameras_, 1);
                MatrixXd y_k(6 * num_cameras_, 1);
                r_k.setZero(6 * num_cameras_, 1);
                d_k.setZero(6 * num_cameras_, 1);
                y_k.setZero(6 * num_cameras_, 1);

                MatrixXd P(6 * num_cameras_, 6 * num_cameras_);
                P = MatrixXd((B - E_C_inverse_E_T).diagonal().asDiagonal()); //预处理矩阵P
                MatrixXd P_inverse(6 * num_cameras_, 6 * num_cameras_);
                P_inverse.setZero(6 * num_cameras_, 6 * num_cameras_); //预处理矩阵P的逆
                for (int j = 0; j < 6 * num_cameras_;j++){
                    P_inverse(j, j) = 1.0 / P(j, j);
                }

                /*//计算条件数Cond(A) = ||A^(-1)|| * ||A||，范数这里采用矩阵的无穷范数或者2范数
                //A = K^(-1) (B - E_C_inverse_E_T) K^(-T)
                //P = K * K^T
                MatrixXd K_inverse(6 * num_cameras_, 6 * num_cameras_);
                K_inverse.setZero(6 * num_cameras_, 6 * num_cameras_);
                for (int j = 0; j < 6 * num_cameras_;j++){
                    K_inverse(j, j) = sqrt(P_inverse(j, j));
                }

                MatrixXd A(6 * num_cameras_, 6 * num_cameras_);
                A = K_inverse * (B - E_C_inverse_E_T) * (K_inverse.transpose());

                //double Cond_A = cal_MatrixInfNorm(A.inverse()) * cal_MatrixInfNorm(A); //无穷范数
                double Cond_A = cal_Matrix2Norm(A.inverse()) * cal_Matrix2Norm(A); //2范数
                cout << "Precondition Cond = " << Cond_A << endl; //预处理后的条件数
                double Original_Cond = cal_Matrix2Norm((B - E_C_inverse_E_T).inverse()) * cal_Matrix2Norm((B - E_C_inverse_E_T));
                cout << "Original Cond = " << Original_Cond << endl; //原始的条件数
                fprintf(ffff,"%lf %lf\n", Original_Cond, Cond_A);*/

                r_k = (B - E_C_inverse_E_T) * delta_parameter_cameras - (v - E_C_inverse_w);
                y_k = P_inverse * r_k;
                d_k = -y_k;
                MatrixXd r_1(6 * num_cameras_, 1);
                r_1 = r_k;

                int step = 1;
                while ((r_k.transpose() * r_k)(0, 0) / (r_1.transpose() * r_1)(0, 0) > 1e-8){
                    step++;
                    double a_k = (r_k.transpose() * y_k)(0, 0) / (d_k.transpose() * (B - E_C_inverse_E_T) * d_k)(0, 0);
                    delta_parameter_cameras += a_k * d_k;

                    MatrixXd r_k_(6 * num_cameras_, 1);
                    MatrixXd y_k_(6 * num_cameras_, 1);
                    r_k_ = r_k;
                    y_k_ = y_k;

                    r_k += a_k * (B - E_C_inverse_E_T) * d_k;
                    y_k = P_inverse * r_k;
                    double b_k = (r_k.transpose() * y_k)(0, 0) / (r_k_.transpose() * y_k_)(0, 0);
                    d_k = -y_k + b_k * d_k;
                }
                cout << "total step = " << step << endl;
                total_step += step;
            }

            /*预处理共轭梯度算法(Symmetric Successive Over Relaxation preconditioner)*/
            if(solve_method_ == "PCG-SSOR"){
                delta_parameter_cameras.setZero(6 * num_cameras_, 1); //迭代初始值0

                MatrixXd r_k(6 * num_cameras_, 1);
                MatrixXd d_k(6 * num_cameras_, 1);
                MatrixXd y_k(6 * num_cameras_, 1);
                r_k.setZero(6 * num_cameras_, 1);
                d_k.setZero(6 * num_cameras_, 1);
                y_k.setZero(6 * num_cameras_, 1);

                MatrixXd D(6 * num_cameras_, 6 * num_cameras_);
                D = MatrixXd((B - E_C_inverse_E_T).diagonal().asDiagonal());
                MatrixXd D_inverse(6 * num_cameras_, 6 * num_cameras_);
                D_inverse.setZero(6 * num_cameras_, 6 * num_cameras_);
                for (int j = 0; j < 6 * num_cameras_;j++){
                    D_inverse(j, j) = 1.0 / D(j, j);
                }

                MatrixXd L(6 * num_cameras_, 6 * num_cameras_); //下三角矩阵
                L.setZero(6 * num_cameras_, 6 * num_cameras_);
                for (int j = 0; j < 6 * num_cameras_;j++){
                    for (int k = 0; k < j;k++){
                        L(j, k) = (B - E_C_inverse_E_T)(j, k);
                    }
                }

                MatrixXd P(6 * num_cameras_, 6 * num_cameras_);
                double w = 0.5;
                P = (1.0 / (w * (2 - w))) * (D + w * L) * D_inverse * ((D + w * L).transpose());
                MatrixXd P_inverse(6 * num_cameras_, 6 * num_cameras_);
                P_inverse = P.inverse();

                /*//计算条件数Cond(A) = ||A^(-1)|| * ||A||，范数这里采用矩阵的无穷范数或者2范数
                //A = K^(-1) (B - E_C_inverse_E_T) K^(-T)
                //P = K * K^T
                MatrixXd K_inverse(6 * num_cameras_, 6 * num_cameras_);
                K_inverse.setZero(6 * num_cameras_, 6 * num_cameras_);
                for (int j = 0; j < 6 * num_cameras_;j++){
                    K_inverse(j, j) = sqrt(D(j, j));
                }
                K_inverse *= (1.0 / sqrt(w * (2 - w))) * ((D + w * L).inverse().transpose());

                MatrixXd A(6 * num_cameras_, 6 * num_cameras_);
                A = K_inverse * (B - E_C_inverse_E_T) * (K_inverse.transpose());

                //double Cond_A = cal_MatrixInfNorm(A.inverse()) * cal_MatrixInfNorm(A); //无穷范数
                double Cond_A = cal_Matrix2Norm(A.inverse()) * cal_Matrix2Norm(A); //2范数
                cout << "Precondition Cond = " << Cond_A << endl; //预处理后的条件数
                double Original_Cond = cal_Matrix2Norm((B - E_C_inverse_E_T).inverse()) * cal_Matrix2Norm((B - E_C_inverse_E_T));
                cout << "Original Cond = " << Original_Cond << endl; //原始的条件数
                fprintf(ffff,"%lf %lf\n", Original_Cond, Cond_A);*/

                r_k = (B - E_C_inverse_E_T) * delta_parameter_cameras - (v - E_C_inverse_w);
                y_k = P_inverse * r_k;
                d_k = -y_k;
                MatrixXd r_1(6 * num_cameras_, 1);
                r_1 = r_k;

                int step = 1;
                while ((r_k.transpose() * r_k)(0, 0) / (r_1.transpose() * r_1)(0, 0) > 1e-8){
                    step++;
                    double a_k = (r_k.transpose() * y_k)(0, 0) / (d_k.transpose() * (B - E_C_inverse_E_T) * d_k)(0, 0);
                    delta_parameter_cameras += a_k * d_k;

                    MatrixXd r_k_(6 * num_cameras_, 1);
                    MatrixXd y_k_(6 * num_cameras_, 1);
                    r_k_ = r_k;
                    y_k_ = y_k;

                    r_k += a_k * (B - E_C_inverse_E_T) * d_k;
                    y_k = P_inverse * r_k;
                    double b_k = (r_k.transpose() * y_k)(0, 0) / (r_k_.transpose() * y_k_)(0, 0);
                    d_k = -y_k + b_k * d_k;
                }
                cout << "total step = " << step << endl;
                total_step += step;
            }

            /*预处理共轭梯度算法(Band-Limited Block-Based Preconditioner)*/
            if(solve_method_.substr(0, 6) == "PCG-BW"){
                string N_str = solve_method_.substr(7, (int)solve_method_.size() - 7);
                int N = 0; //(band width) / 2
                for (int j = (int)N_str.size() - 1; j >= 0; j--){
                    N += (N_str[(int)N_str.size() - j - 1] - '0') * (int)pow(10.0, double(j));
                }
                //cout << "N = " << N << endl;

                delta_parameter_cameras.setZero(6 * num_cameras_, 1); //迭代初始值0

                MatrixXd r_k(6 * num_cameras_, 1);
                MatrixXd d_k(6 * num_cameras_, 1);
                MatrixXd y_k(6 * num_cameras_, 1);
                r_k.setZero(6 * num_cameras_, 1);
                d_k.setZero(6 * num_cameras_, 1);
                y_k.setZero(6 * num_cameras_, 1);

                MatrixXd D(6 * num_cameras_, 6 * num_cameras_); //P的对角块矩阵D
                D.setZero(6 * num_cameras_, 6 * num_cameras_);
                for (int j = 0; j < num_cameras_;j++){
                    D.block<6, 6>(j * 6, j * 6) = (B - E_C_inverse_E_T).block<6, 6>(j * 6, j * 6);
                }

                MatrixXd L(6 * num_cameras_, 6 * num_cameras_); //L是P的排除对角线块的下三角块矩阵，P是对称的，P = L^T + L + D
                L.setZero(6 * num_cameras_, 6 * num_cameras_);
                for (int j = 0; j < N - 1;j++){
                    for (int k = 0; k < j;k++){
                        L.block<6, 6>(j * 6, k * 6) = (B - E_C_inverse_E_T).block<6, 6>(j * 6, k * 6);
                    }
                }
                for (int j = N - 1; j < num_cameras_;j++){
                    for (int k = j - N + 1; k < j;k++){
                        L.block<6, 6>(j * 6, k * 6) = (B - E_C_inverse_E_T).block<6, 6>(j * 6, k * 6);
                    }
                }

                MatrixXd P(6 * num_cameras_, 6 * num_cameras_);
                if(N != 0){
                    P = L.transpose() + L + D;
                }else{
                    P = MatrixXd::Identity(6 * num_cameras_, 6 * num_cameras_);
                }

                MatrixXd P_inverse(6 * num_cameras_, 6 * num_cameras_);
                P_inverse = P.inverse();

                /*//计算条件数Cond(A) = ||A^(-1)|| * ||A||，范数这里采用矩阵的无穷范数或者2范数
                //A = K^(-1) (B - E_C_inverse_E_T) K^(-T)
                //P = K * K^T
                MatrixXd K_inverse(6 * num_cameras_, 6 * num_cameras_);
                K_inverse.setZero(6 * num_cameras_, 6 * num_cameras_);
                K_inverse = P.llt().matrixL();
                K_inverse = K_inverse.inverse();

                MatrixXd A(6 * num_cameras_, 6 * num_cameras_);
                A = K_inverse * (B - E_C_inverse_E_T) * (K_inverse.transpose());

                //double Cond_A = cal_MatrixInfNorm(A.inverse()) * cal_MatrixInfNorm(A); //无穷范数
                double Cond_A = cal_Matrix2Norm(A.inverse()) * cal_Matrix2Norm(A); //2范数
                cout << "Precondition Cond = " << Cond_A << endl; //预处理后的条件数
                double Original_Cond = cal_Matrix2Norm((B - E_C_inverse_E_T).inverse()) * cal_Matrix2Norm((B - E_C_inverse_E_T));
                cout << "Original Cond = " << Original_Cond << endl; //原始的条件数
                fprintf(ffff,"%lf %lf\n", Original_Cond, Cond_A);*/

                r_k = (B - E_C_inverse_E_T) * delta_parameter_cameras - (v - E_C_inverse_w);
                y_k = P_inverse * r_k;
                d_k = -y_k;
                MatrixXd r_1(6 * num_cameras_, 1);
                r_1 = r_k;

                //为了防止r_k^T * d_k > 0，即d_k变为上升方向，同时也为了防止相邻两次迭代的梯度偏离正交性较大，故采用再开始的共轭梯度算法(restart CG)进行迭代
                int step = 1;
                while ((r_k.transpose() * r_k)(0, 0) / (r_1.transpose() * r_1)(0, 0) > 1e-8){
                    step++;
                    double a_k = (r_k.transpose() * y_k)(0, 0) / (d_k.transpose() * (B - E_C_inverse_E_T) * d_k)(0, 0);
                    delta_parameter_cameras += a_k * d_k;

                    MatrixXd r_k_(6 * num_cameras_, 1);
                    MatrixXd y_k_(6 * num_cameras_, 1);
                    r_k_ = r_k;
                    y_k_ = y_k;

                    r_k += a_k * (B - E_C_inverse_E_T) * d_k;

                    //相邻两次迭代的梯度偏离正交性较大时restart
                    /*if((r_k.transpose() * y_k_)(0, 0) / r_k.squaredNorm() >= 0.1){
                        r_k = (B - E_C_inverse_E_T) * delta_parameter_cameras - (v - E_C_inverse_w);
                        y_k = P_inverse * r_k;
                        d_k = -y_k;
                        step--;
                        continue;
                    }*/

                    y_k = P_inverse * r_k;
                    double b_k = (r_k.transpose() * y_k)(0, 0) / (r_k_.transpose() * y_k_)(0, 0);
                    d_k = -y_k + b_k * d_k;

                    //if r_k^T * d_k > 0, then restart
                    /*if((d_k.transpose() * y_k)(0, 0) > 0){
                        r_k = (B - E_C_inverse_E_T) * delta_parameter_cameras - (v - E_C_inverse_w);
                        y_k = P_inverse * r_k;
                        d_k = -y_k;
                        step--;
                        continue;
                    }*/
                    /*if(i == 14){
                        //cout << d_k.transpose()*y_k << endl;
                        cout << (r_k.transpose() * r_k)(0, 0) / (r_1.transpose() * r_1)(0, 0) << endl;
                        //cout << (r_k.transpose() * r_k)(0, 0) << endl;
                        sleep(1);
                    }*/
                }
                cout << "total step = " << step << endl;
                total_step += step;
            }

            //计算delta_parameter_points
            MatrixXd E_T_delta_parameter_cameras(3 * num_points_, 1);
            E_T_delta_parameter_cameras.setZero(3 * num_points_, 1);

            /*for (int j = 0; j < num_points_; j++){
                E_T_delta_parameter_cameras.block<3, 1>(3 * j, 0) = E_T[j] * delta_parameter_cameras;
            }*/

            for (int j = 0; j < num_points_;j++){
                for (int k = 0; k < num_cameras_;k++){
                    E_T_delta_parameter_cameras.block<3, 1>(3 * j, 0) += E_T[j].block<3, 6>(0, 6 * k) * delta_parameter_cameras.block<6, 1>(6 * k, 0);
                }
            }
            cout << "time4 = " << (clock() - time_stt) / (double)CLOCKS_PER_SEC << "s" << endl;

            MatrixXd delta_parameter_points(3 * num_points_, 1);
            delta_parameter_points.setZero(3 * num_points_, 1);

            for (int j = 0; j < num_points_; j++){
                delta_parameter_points.block<3, 1>(3 * j, 0) = C_inverse[j] * (w - E_T_delta_parameter_cameras).block<3, 1>(3 * j, 0);
            }
            cout << "time5 = " << (clock() - time_stt) / (double)CLOCKS_PER_SEC << "s" << endl;

            //当增量的2范数小于1e-8时迭代终止
            if(delta_parameter_points.norm() + delta_parameter_cameras.norm() < 1e-8){
                break;
            }

            MatrixXd parameter_cameras_new(6, num_cameras_); //临时储存更新后的fa和t值
            parameter_cameras_new.setZero(6, num_cameras_);

            for (int j = 0; j < num_cameras_; j++){
                //Sophus::SE3 SE3_updated = Sophus::SE3::exp(delta_parameter_cameras.block(6 * j, 0, 6, 1)) * Sophus::SE3::exp(parameter_se.col(j));
                //parameter_cameras_new.col(j).block(0, 0, 3, 1) = SE3_updated.log().block(3, 0, 3, 1);
                //parameter_cameras_new.col(j).block(3, 0, 3, 1) = SE3_updated.matrix().block(0, 3, 3, 1);
                Matrix<double, 6, 1> SE3_updated = se3_exp_LeftMutiply(delta_parameter_cameras.block<6, 1>(6 * j, 0), parameter_se.col(j));
                Matrix<double, 4, 4> SE3_updated_ToMatrix = se3_exp_ToMatrix(SE3_updated);
                parameter_cameras_new.col(j).block<3, 1>(0, 0) = SE3_updated.block<3, 1>(3, 0);
                parameter_cameras_new.col(j).block<3, 1>(3, 0) = SE3_updated_ToMatrix.block<3, 1>(0, 3);
            }
            cout << "time6 = " << (clock() - time_stt) / (double)CLOCKS_PER_SEC << "s" << endl;

            double error_sum_next = 0;
            for (int j = 0; j < (int)terms.size();j++){
                int camera_id = terms[j]->camera_id_;
                int point_id = terms[j]->point_id_;

                Matrix<double, 6, 1> camera_in = parameter_cameras_new.col(camera_id);
                Matrix<double, 3, 1> P;
                for (int k = 0; k < 3;k++){
                    P(k, 0) = *(parameter_points_ + point_id * 3 + k) + delta_parameter_points(3 * point_id + k, 0);
                }

                terms[j]->cal_Rt_se(camera_in);
                Matrix<double, 2, 1> error = terms[j]->call_error(P);
                error_sum_next += 0.5 * error.squaredNorm();
            }
            cout << "time7 = " << (clock() - time_stt) / (double)CLOCKS_PER_SEC << "s" << endl;

            //LM method 更新参数
            if(iter_method_ == "LM"){
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
                    successful_iterations++;
                    total_time_consumption += (clock() - time_stt) / (double)CLOCKS_PER_SEC;
                    cout << "successful step! iteration = " << i << " , error before = " << error_sum << " , error next = " << error_sum_next << " , error change = " << error_sum - error_sum_next << " , lambda = " << lambda_ << " , time consumption = " << (clock() - time_stt) / (double)CLOCKS_PER_SEC << "s" << endl;
                    //fprintf(ff, "successful step! iteration = %d , error before = %lf , error next = %lf , error change = %lf , lambda = %lf , time consumption = %lfs\n\n", i, error_sum, error_sum_next, error_sum - error_sum_next, lambda_, (clock() - time_stt) / (double)CLOCKS_PER_SEC);
                    //fprintf(fff, "%lf %lf %lf %lf\n", error_sum, error_sum_next, error_sum - error_sum_next, (clock() - time_stt) / (double)CLOCKS_PER_SEC);
                }else{
                    lambda_ *= 10;
                    total_time_consumption += (clock() - time_stt) / (double)CLOCKS_PER_SEC;
                    cout << "unsuccessful step! iteration = " << i << " , error before = " << error_sum << " , error next = " << error_sum_next << " , error change = " << error_sum - error_sum_next << " , lambda = " << lambda_ << " , time consumption = " << (clock() - time_stt) / (double)CLOCKS_PER_SEC << "s" << endl;
                    //fprintf(ff, "unsuccessful step! iteration = %d , error before = %lf , error next = %lf , error change = %lf , lambda = %lf , time consumption = %lfs\n\n", i, error_sum, error_sum_next, error_sum - error_sum_next, lambda_, (clock() - time_stt) / (double)CLOCKS_PER_SEC);
                }
            }

            //LM_TR method 更新参数
            if(iter_method_ == "LM_TR"){
                double h = (delta_parameter_cameras.transpose() * (delta_parameter_cameras * lambda_ + v))(0, 0) + (delta_parameter_points.transpose() * (delta_parameter_points * lambda_ + w))(0, 0);
                double rou = 2 * (error_sum - error_sum_next) / h; 
                //cout << "rou = " << rou << endl;
                //cout << "1 - pow(2 * rou - 1, 3.0) = " << 1 - pow(2 * rou - 1, 3.0) << endl;

                if(rou > 0){
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
                    lambda_ *= max((double)(1.0 / 3.0), (double)(1 - pow(2 * rou - 1, 3.0)));
                    v_ = 2;
                    successful_iterations++;
                    total_time_consumption += (clock() - time_stt) / (double)CLOCKS_PER_SEC;
                    cout << "successful step! iteration = " << i << " , error before = " << error_sum << " , error next = " << error_sum_next << " , error change = " << error_sum - error_sum_next << " , lambda = " << lambda_ << " , time consumption = " << (clock() - time_stt) / (double)CLOCKS_PER_SEC << "s" << endl;
                    //fprintf(ff, "successful step! iteration = %d , error before = %lf , error next = %lf , error change = %lf , lambda = %lf , time consumption = %lfs\n\n", i, error_sum, error_sum_next, error_sum - error_sum_next, lambda_, (clock() - time_stt) / (double)CLOCKS_PER_SEC);
                    //fprintf(fff, "%lf %lf %lf %lf\n", error_sum, error_sum_next, error_sum - error_sum_next, (clock() - time_stt) / (double)CLOCKS_PER_SEC);
                }else{
                    lambda_ *= v_;
                    v_ *= 2;
                    total_time_consumption += (clock() - time_stt) / (double)CLOCKS_PER_SEC;
                    cout << "unsuccessful step! iteration = " << i << " , error before = " << error_sum << " , error next = " << error_sum_next << " , error change = " << error_sum - error_sum_next << " , lambda = " << lambda_ << " , time consumption = " << (clock() - time_stt) / (double)CLOCKS_PER_SEC << "s" << endl;
                    //fprintf(ff, "unsuccessful step! iteration = %d , error before = %lf , error next = %lf , error change = %lf , lambda = %lf , time consumption = %lfs\n\n", i, error_sum, error_sum_next, error_sum - error_sum_next, lambda_, (clock() - time_stt) / (double)CLOCKS_PER_SEC);
                }
            }

            //GN method 更新参数
            if(iter_method_ == "GN"){
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
                    successful_iterations++;
                    total_time_consumption += (clock() - time_stt) / (double)CLOCKS_PER_SEC;
                    cout << "successful step! iteration = " << i << " , error before = " << error_sum << " , error next = " << error_sum_next << " , error change = " << error_sum - error_sum_next << " , lambda = " << lambda_ << " , time consumption = " << (clock() - time_stt) / (double)CLOCKS_PER_SEC << "s" << endl;
                    //fprintf(ff, "successful step! iteration = %d , error before = %lf , error next = %lf , error change = %lf , lambda = %lf , time consumption = %lfs\n\n", i, error_sum, error_sum_next, error_sum - error_sum_next, lambda_, (clock() - time_stt) / (double)CLOCKS_PER_SEC);
                    //fprintf(fff, "%lf %lf %lf %lf\n", error_sum, error_sum_next, error_sum - error_sum_next, (clock() - time_stt) / (double)CLOCKS_PER_SEC);
                }else{
                    lambda_ *= 10;
                    total_time_consumption += (clock() - time_stt) / (double)CLOCKS_PER_SEC;
                    cout << "unsuccessful step! iteration = " << i << " , error before = " << error_sum << " , error next = " << error_sum_next << " , error change = " << error_sum - error_sum_next << " , lambda = " << lambda_ << " , time consumption = " << (clock() - time_stt) / (double)CLOCKS_PER_SEC << "s" << endl;
                    //fprintf(ff, "unsuccessful step! iteration = %d , error before = %lf , error next = %lf , error change = %lf , lambda = %lf , time consumption = %lfs\n\n", i, error_sum, error_sum_next, error_sum - error_sum_next, lambda_, (clock() - time_stt) / (double)CLOCKS_PER_SEC);
                }
            }

            //当总的重投影误差大于等于0且小于1e-8迭代终止
            if(error_sum - error_sum_next >= 0 && error_sum - error_sum_next < 1e-8){
                break;
            }
        }
        cout << "initial error = " << initial_error << " , final error = " << final_error << " , successful iterations/total iterations = " << successful_iterations << "/" << total_iterations << " , total error change = " << initial_error - final_error << " , total time consumption = " << total_time_consumption << "s" << " , average steps = " << double(total_step) / double(total_iterations) << endl;
        //fprintf(ff, "initial error = %lf , final error = %lf , successful iterations/total iterations = %d/%d , total error change = %lf , total time consumption = %lfs , average steps = %lf\n", initial_error, final_error, successful_iterations, total_iterations, initial_error - final_error, total_time_consumption, double(total_step) / double(total_iterations));
        //fclose(ff);
        //fclose(fff);
        //fclose(ffff);
    }

    void add_errorterms(ReprojectionError *e)
    {
        terms.push_back(e);
    }

    int num_iterations_, num_cameras_, num_points_, num_camera_parameters_;
    double v_ = 0, maxa_ = 0; bool flag = true; //LM_TR
    int total_step = 0; //PCG or CG, average step (total_step / total_iterations)
    double lambda_, initial_error = 0, final_error = 0;
    double *parameter_cameras_;
    double *parameter_points_;
    string iter_method_;
    string solve_method_;
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

    if(!f.load_file("/home/vision/Desktop/code_c_c++/my_BA/opt_data/big_data.txt")){
        cout << "unable to open file!" << endl;
    }

    //LM_SchurOptimization(int num_iterations, int num_cameras, int num_points, int num_camera_parameters, double lambda, double* parameter_cameras, double* parameter_points, string iter_method, string solve_method)

    // iter_method:
    // "GN":lambda取2
    // "LM":lambda取1e-3
    // "LM_TR":lambda取1e-13(Trust Region Method)

    // solve_method:
    // "LDL":LDL分解算法 
    // "QR":QR分解算法
    // "CG":共轭梯度算法
    // "PCG-J":预处理共轭梯度算法(Jacobi preconditioner)
    // "PCG-SSOR":预处理共轭梯度算法(Symmetric Successive Over Relaxation preconditioner)
    // "PCG-BW-N":预处理共轭梯度算法(Band-Limited Block-Based Preconditioner), 2 * N = band width (N取0到number of cameras)
    LM_GN_SchurOptimization opt(3000, f.num_cameras_, f.num_points_, num_camera_parameters, 1e-3, f.parameter_cameras(), f.parameter_points(), "LM", "LDL");

    for (int i = 0; i < f.num_observations_;i++){
        //ReprojectionError(double camera_id, double fx, double fy, double cx, double cy, double k1, double k2, double point_id, double u, double v)
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