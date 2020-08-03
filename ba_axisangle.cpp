#include<iostream>
#include<random>
#include<vector>
#include<string>
#include<cmath>
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
        double r = P_x * P_x + P_y * P_y;
        double distortion = 1.0 + r * (k1_ + k2_ * r);
        double predicted_u = fx_ * P_x * distortion + cx_;
        double predicted_v = fy_ * P_y * distortion + cy_;
        Matrix<double, 2, 1> error;
        error << u_ - predicted_u, v_ - predicted_v; //观测值减去预测值
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
    double fx_, fy_, cx_, cy_, u_, v_, k1_, k2_;
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

    void optimize(){
        //给H11(B)，H22(C)，H12(E)，H21(E_T)，g1(v)，g2(w)设置初始值0
        MatrixXd B(6 * num_cameras_, 6 * num_cameras_);
        B = MatrixXd::Zero(6 * num_cameras_, 6 * num_cameras_);
        MatrixXd C(3 * num_points_, 3 * num_points_);
        C = MatrixXd::Zero(3 * num_points_, 3 * num_points_);
        MatrixXd E(6 * num_cameras_, 3 * num_points_);
        E = MatrixXd::Zero(6 * num_cameras_, 3 * num_points_);
        MatrixXd E_T(3 * num_points_, 6 * num_cameras_);
        E_T = MatrixXd::Zero(3 * num_points_, 6 * num_cameras_);
        MatrixXd v(6 * num_cameras_, 1);
        v = MatrixXd::Zero(6 * num_cameras_, 1);
        MatrixXd w(3 * num_points_, 1);
        w = MatrixXd::Zero(3 * num_points_, 1);
        MatrixXd parameter_se(6, num_cameras_);
        parameter_se = MatrixXd::Zero(6, num_cameras_);

        /*if(method_ == "GN"){
            lambda_ = 0;
        }*/

        FILE *ff = fopen("/home/vision/Desktop/code_c_c++/my_BA/log/log.txt","w");
        cout << "number of cameras: " << num_cameras_ << " ,number of points: " << num_points_ << " ,number of observations: " << (int)terms.size() << " ,method: " << method_ << " ,initial lambda: " << lambda_ << endl;
        fprintf(ff, "number of cameras: %d ,number of points: %d ,number of observations: %d, method: %s\n\n", num_cameras_, num_points_, (int)terms.size(), method_.c_str());

        for (int i = 0; i < num_iterations_; i++){
            double error_sum = 0;
            for (int j = 0; j < (int)terms.size(); j++)
            {
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

                terms[j]->cal_Rt_se(camera_in); //计算R，t和se
                parameter_se.col(camera_id) = terms[j]->se; //储存se
                //cout << parameter_se.col(camera_id) << endl << endl;
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
                    C.block(point_id * 3, point_id * 3, 3, 3) += J_ETJ_E + lambda_ * D_ETE_D;
                }

                if(method_ == "GN"){
                    B.block(camera_id * 6, camera_id * 6, 6, 6) += lambda_ * J_FTJ_F;
                    C.block(point_id * 3, point_id * 3, 3, 3) += lambda_ * J_ETJ_E;
                }
                
                E.block(camera_id * 6, point_id * 3, 6, 3) += J_F.transpose() * J_E;
                E_T.block(point_id * 3, camera_id * 6, 3, 6) += J_E.transpose() * J_F;
                v.block(camera_id * 6, 0, 6, 1) += -J_F.transpose() * error;
                w.block(point_id * 3, 0, 3, 1) += -J_E.transpose() * error;
            }

            if(i == 0){
                initial_error = error_sum;
                final_error = error_sum;
            }

            MatrixXd C_inverse(3 * num_points_, 3 * num_points_);
            //C_inverse = C.colPivHouseholderQr().solve(MatrixXd::Identity(3 * num_points_, 3 * num_points_));
            //对C进行分块矩阵再赋值，避免大矩阵求逆
            for (int j = 0; j < num_points_;j++){
                C_inverse.block(j * 3, j * 3, 3, 3) = C.block(j * 3, j * 3, 3, 3).inverse();
            }

            MatrixXd delta_parameter_cameras(6 * num_cameras_, 1);
            delta_parameter_cameras = (B - E * C_inverse * E_T).colPivHouseholderQr().solve(v - E * C_inverse * w);
            MatrixXd delta_parameter_points(3 * num_points_, 1);
            delta_parameter_points = (C_inverse * (w - E_T * delta_parameter_cameras));

            if(delta_parameter_points.norm() + delta_parameter_cameras.norm() < 1e-10){
                final_error = error_sum;
                break;
            }

            MatrixXd parameter_cameras_new(6, num_cameras_); //临时储存更新后的fa和t值
            parameter_cameras_new = MatrixXd::Zero(6, num_cameras_);

            for (int j = 0; j < num_cameras_; j++){
                parameter_se.col(j) += delta_parameter_cameras.block(j * 6, 0, 6, 1);
                Matrix<double, 3, 1> fa = parameter_se.col(j).block(3, 0, 3, 1);
                double theta = fa.norm();
                Matrix<double, 3, 1> a = fa / theta;
                Matrix3d skew_a;
                skew_a << 0, -a(2, 0), a(1, 0), a(2, 0), 0, -a(0, 0), -a(1, 0), a(0, 0), 0;
                Matrix3d J = sin(theta) / theta * Matrix3d::Identity() + (1 - sin(theta) / theta) * a * a.transpose() + (1 - cos(theta)) / theta * skew_a;
                Matrix<double,3,1> t = J * parameter_se.col(j).block(0, 0, 3, 1);
                parameter_cameras_new.col(j).block(0, 0, 3, 1) = fa;
                parameter_cameras_new.col(j).block(3, 0, 3, 1) = t;
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

                terms[j]->cal_Rt_se(camera_in);
                Matrix<double, 2, 1> error = terms[j]->call_error(P);
                error_sum_next += 0.5 * error.squaredNorm();
            }

            //LM method 更新参数 lambda_ < 1
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
                    cout << "successful step! iteration = " << i << ", error before = " << error_sum << ", error next = " << error_sum_next << " , lambda = " << lambda_ << endl;
                    fprintf(ff, "successful step! iteration = %d, error before = %lf, error next = %lf, lambda = %lf\n\n", i, error_sum, error_sum_next, lambda_);
                }else{
                    lambda_ *= 10;
                    cout << "unsuccessful step! iteration = " << i << ", error before = " << error_sum << ", error next = " << error_sum_next << " , lambda = " << lambda_ << endl;
                    fprintf(ff, "unsuccessful step! iteration = %d, error before = %lf, error next = %lf, lambda = %lf\n\n", i, error_sum, error_sum_next, lambda_);
                }
            }

            //GN method 更新参数 lambda_ > 1
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
                    cout << "successful step! iteration = " << i << ", error before = " << error_sum << ", error next = " << error_sum_next << " , lambda = " << lambda_ << endl;
                    fprintf(ff, "successful step! iteration = %d, error before = %lf, error next = %lf, lambda = %lf\n\n", i, error_sum, error_sum_next, lambda_);
                }else{
                    lambda_ *= 10;
                    cout << "unsuccessful step! iteration = " << i << ", error before = " << error_sum << ", error next = " << error_sum_next << " , lambda = " << lambda_ << endl;
                    fprintf(ff, "unsuccessful step! iteration = %d, error before = %lf, error next = %lf, lambda = %lf\n\n", i, error_sum, error_sum_next, lambda_);
                }
            }
        }
        cout << "initial error = " << initial_error << " ,final error = " << final_error << " ,error change = " << initial_error - final_error << endl;
        fprintf(ff, "initial error = %lf, final error = %lf, error change = %lf\n", initial_error, final_error, initial_error - final_error);
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

int main(int argc, char** argv)
{
    /*if(argc != 2){
        cout << "error input!" << endl;
    }*/

    int num_camera_parameters = 9; //相机参数个数（按照顺序依次：旋转向量、平移向量、相机焦距（fx=fy）、径向畸变系数k1、径向畸变系数k2）

    //load_data(int num_camera_parameters)
    load_data f(num_camera_parameters); //初始化

    if(!f.load_file("/home/vision/Desktop/code_c_c++/my_BA/opt_data/mini_data_2.txt")){
        cout << "unable to open file!" << endl;
    }

    //LM_SchurOptimization(int num_iterations, int num_cameras, int num_points, int num_camera_parameters, double lambda, double* parameter_cameras, double* parameter_points, string method)
    LM_GN_SchurOptimization opt(200, f.num_cameras_, f.num_points_, num_camera_parameters, 5, f.parameter_cameras(), f.parameter_points(), "GN"); //"GN":lambda取大于１的(5、10)，"LM":lambda取小于１(1e-5、1e-4)
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

    return 0;
}