#include<iostream>
#include<random>
#include<vector>
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
            fscanf_change(fp, "%d", point_index_ + i);
            fscanf_change(fp, "%d", camera_index_ + i);
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
        Matrix<double, 3, 1> a = fa / theta;
        Matrix3d J = sin(theta) / theta * Matrix3d::Identity() + (1 - sin(theta) / theta) * a * a.transpose() + (1 - cos(theta)) / theta * skew_matrix(a);
        t = camera_in.block(0, 3, 3, 1);
        Matrix<double,3,1> rou = J.colPivHouseholderQr().solve(t);
        se << rou, fa;
    }

    Matrix<double,2,1> call_error(Matrix<double,3,1> P){
        Matrix<double, 3, 1> P_ = R * P + t;
        double P_x = P(0, 0) / P(2, 0);
        double P_y = P(1, 0) / P(2, 0);
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

class LM_SchurOptimization
{
public:
    LM_SchurOptimization(int num_iterations, int num_cameras, int num_points, int num_camera_parameters, double lambda, double* parameter_cameras, double* parameter_points){
        num_iterations_ = num_iterations;
        num_cameras_ = num_cameras;
        num_points_ = num_points;
        num_camera_parameters_ = num_camera_parameters;
        lambda_ = lambda;
        parameter_cameras_ = parameter_cameras;
        parameter_points_ = parameter_points;
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
        MatrixXd parameter_se(num_cameras_, 6);
        parameter_se = MatrixXd::Zero(6, num_cameras_);

        for (int i = 0; i < num_iterations_;i++){
            double error_sum = 0;
            for (int j = 0; j < (int)terms.size(); j++)
            {
                int camera_id = terms[j]->camera_id_;
                int point_id = terms[j]->point_id_;

                Matrix<double, 6, 1> camera_in;
                for (int k = 0; k < 6;k++){
                    camera_in(k, 0) = *(parameter_cameras_ + camera_id * 9 + k);
                }
                Matrix<double, 3, 1> P;
                for (int k = 0; k < 3;k++){
                    P(k, 0) = *(parameter_points_ + point_id * 3 + k);
                }

                terms[j]->cal_Rt_se(camera_in); //计算R，t和se
                parameter_se.col(camera_id) = terms[j]->se; //储存se
                Matrix<double, 2, 9> EF = terms[j]->cal_FE(P); //计算E，F
                Matrix<double, 2, 6> F = EF.block(0, 0, 2, 6);
                Matrix<double, 2, 3> E = EF.block(0, 6, 2, 3);
                Matrix<double, 2, 1> error = terms[j]->call_error(P); //计算误差（观测值减去预测值）

                error_sum += 0.5 * error.squaredNorm();

                B.block(camera_id * 6, camera_id * 6, 6, 6) += F.transpose() * F + lambda_ * MatrixXd((F.transpose() * F).diagonal().asDiagonal());
                C.block(point_id * 3, point_id * 3, 3, 3) += E.transpose() * E + lambda_ * MatrixXd((E.transpose() * E).diagonal().asDiagonal());
                E.block(camera_id * 6, point_id * 3, 6, 3) += F.transpose() * E;
                E_T.block(point_id * 3, camera_id * 6, 3, 6) += E.transpose() * F;
                v.block(camera_id * 6, 0, 6, 1) += -F.transpose() * error;
                w.block(point_id * 3, 0, 3, 1) += -E.transpose() * error;
            }

            MatrixXd C_inverse(3 * num_points_, 3 * num_points_);
            C_inverse = C.inverse();
            MatrixXd delta_parameter_cameras(6 * num_cameras_, 1);
            delta_parameter_cameras = (B - E * C_inverse * E_T).colPivHouseholderQr().solve(v - E * C_inverse * w);
            MatrixXd delta_parameter_points(3 * num_points_, 1);
            delta_parameter_points = C_inverse * (w - E_T * delta_parameter_cameras);

            MatrixXd parameter_cameras_new(6, num_cameras_); //临时储存更新后的fa和t值
            parameter_cameras_new = MatrixXd::Zero(num_cameras_, 6);

            for (int j = 0; j < num_cameras_; j++)
            {
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

            //更新参数
            if(error_sum_next < error_sum){
                for (int j = 0; j < num_cameras_;j++){
                    for (int k = 0; k < 6;k++){
                        *(parameter_cameras_ + j * num_camera_parameters_ + k) = parameter_cameras_new(k, j);
                    }
                }
                for (int j = 0; j < num_points_;j++){
                    for (int k = 0; k < 3;k++){
                        *(parameter_points_ + 3 * num_points_ + k) += delta_parameter_points(3 * j + k, 0);
                    }
                }
                lambda_ /= 10;
                error_sum = error_sum_next;
            }else{
                lambda_ *= 10;
            }

            cout << "iteration = " << i << ", error = " << error_sum << " , lambda = " << lambda_ << endl;
        }
    }

    void add_errorterms(ReprojectionError *e)
    {
        terms.push_back(e);
    }

    int num_iterations_, num_cameras_, num_points_, num_camera_parameters_;
    double lambda_;
    double *parameter_cameras_;
    double *parameter_points_;
    vector<ReprojectionError *> terms;
};

int main(int argc, char** argv)
{
    if(argc != 2){
        cout << "error input!" << endl;
    }

    int num_camera_parameters = 9; //相机参数个数
    //load_data(int num_camera_parameters)
    load_data f(num_camera_parameters); //初始化

    if(!f.load_file(argv[1])){
        cout << "unable to open file!" << endl;
    }

    //LM_SchurOptimization(int num_iterations, int num_cameras, int num_points, int num_camera_parameters, double lambda, double* parameter_cameras, double* parameter_points)
    LM_SchurOptimization opt(100, f.num_cameras_, f.num_points_, num_camera_parameters, 1e-4, f.parameter_cameras(), f.parameter_points());
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
    }

    //开始优化
    opt.optimize();

    FILE* fp = fopen("update.txt", "w");
    for (int i = 0; i < f.num_parameters_;i++){
        fprintf(fp, "%.16e\n", *(f.parameters_ + i));
    }
    fclose(fp);

    return 0;
}