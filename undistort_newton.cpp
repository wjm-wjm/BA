#include<bits/stdc++.h>
#include<Eigen/Core>
#include<Eigen/Dense>
#include<ctime>
#include<opencv2/opencv.hpp>
#include<opencv2/highgui.hpp>

using namespace std;
using namespace cv;
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


Matrix<double,2,1> undistort_points(Matrix<double,2,1> u_v, double cx, double cy, double fx, double fy, double k1, double k2){
    Matrix<double, 2, 1> x1_y1, x2_y2, delta_x1_y1;
    double r;
    delta_x1_y1(0, 0) = 1000;
    delta_x1_y1(1, 0) = 1000;
    x1_y1(0, 0) = (u_v(0, 0) - cx) / fx;
    x1_y1(1, 0) = (u_v(1, 0) - cy) / fy;
    int step = 0;
    while (delta_x1_y1.norm() >= 1e-5){
        //cout << x1_y1 << endl;
        //cout << endl;
        r = x1_y1.squaredNorm();
        x2_y2(0, 0) = x1_y1(0, 0) * (1.0 + r * (k1 + k2 * r));
        x2_y2(1, 0) = x1_y1(1, 0) * (1.0 + r * (k1 + k2 * r));

        Matrix<double, 2, 1> residual;
        residual(0, 0) = fx * x2_y2(0, 0) + cx - u_v(0, 0);
        residual(1, 0) = fy * x2_y2(1, 0) + cy - u_v(1, 0);

        Matrix<double, 2, 2> J_k;
        J_k(0, 0) = fx * (1 + 3 * k1 * pow(x1_y1(0, 0), 2.0) + k1 * pow(x1_y1(1, 0), 2.0) + 5 * k2 * pow(x1_y1(0, 0), 4) + 6 * k2 * pow(x1_y1(0, 0), 2.0) * pow(x1_y1(1, 0), 2.0) + k2 * pow(x1_y1(1, 0), 4.0));
        J_k(0, 1) = fx * (2 * k1 * x1_y1(0, 0) * x1_y1(1, 0) + 4 * k2 * pow(x1_y1(0, 0), 3.0) * x1_y1(1, 0) + 4 * k2 * x1_y1(0, 0) * pow(x1_y1(1, 0), 3.0));
        J_k(1, 0) = fy * (2 * k1 * x1_y1(0, 0) * x1_y1(1, 0) + 4 * k2 * pow(x1_y1(0, 0), 3.0) * x1_y1(1, 0) + 4 * k2 * x1_y1(0, 0) * pow(x1_y1(1, 0), 3.0));
        J_k(1, 1) = fy * (1 + 3 * k1 * pow(x1_y1(1, 0), 2.0) + k1 * pow(x1_y1(0, 0), 2.0) + 5 * k2 * pow(x1_y1(1, 0), 4) + 6 * k2 * pow(x1_y1(0, 0), 2.0) * pow(x1_y1(1, 0), 2.0) + k2 * pow(x1_y1(0, 0), 4.0));

        delta_x1_y1 = J_k.colPivHouseholderQr().solve(-residual);
        x1_y1 += delta_x1_y1;
        //cout << delta_x1_y1 << endl;
        //cout << endl;
        //cout << "residual = " << residual << endl;
        //cout << endl;
        step++;
    }
    cout << "step = " << step << endl;
    return x1_y1;
}

int main(){

    int num_camera_parameters = 9;

    load_data f(num_camera_parameters);
    f.load_file("/home/vision/Desktop/code_c_c++/my_BA/opt_data/big_data.txt");

    /*for (int i = 0; i < f.num_observations_;i++){
        clock_t time_stt = clock(); //计时

        double fx = *(f.parameter_cameras() + f.camera_index_[i] * num_camera_parameters + 6);
        double fy = fx;
        double cx = 0;
        double cy = 0;
        double k1 = *(f.parameter_cameras() + f.camera_index_[i] * num_camera_parameters + 7);
        double k2 = *(f.parameter_cameras() + f.camera_index_[i] * num_camera_parameters + 8);
        double u = *(f.observations() + 2 * i);
        double v = *(f.observations() + 2 * i + 1);
        Matrix<double, 2, 1> u_v;
        u_v << u, v;
        Matrix<double, 2, 1> x1_y1;
        x1_y1 = undistort_points(u_v, cx, cy, fx, fy, k1, k2);
        Matrix<double, 2, 1> new_u_v;
        new_u_v(0, 0) = fx * x1_y1(0, 0) + cx;
        new_u_v(1, 0) = fy * x1_y1(1, 0) + cy;
        double r = x1_y1.squaredNorm();
        double distortion = 1.0 + r * (k1 + k2 * r);
        Matrix<double, 2, 1> distorted_new_u_v;
        distorted_new_u_v(0, 0) = fx * x1_y1(0, 0) * distortion + cx;
        distorted_new_u_v(1, 0) = fy * x1_y1(1, 0) * distortion + cy;
        cout << "distorted u v = " << u_v << endl;
        cout << "distort by hand u v = " << distorted_new_u_v << endl;
        cout << "undistortrd u v = " << new_u_v << endl;
        cout << "time consumption = " << (clock() - time_stt) / (double)CLOCKS_PER_SEC << "s" << endl;
        cout << endl;
    }*/

    Matrix<double, 2, 1> u_v;
    u_v << 385.99000000000001, -387.12;
    double fx = 1510.7711540717191;
    double fy = 1510.7711540717191;
    double cx = 0;
    double cy = 0;
    double k1 = -0.49972197200096624;
    double k2 = 0.51099729734344146;
    Matrix<double, 2, 1> x1_y1;
    x1_y1 = undistort_points(u_v, cx, cy, fx, fy, k1, k2);
    cout << x1_y1 << endl;
    Matrix<double, 2, 1> new_u_v;
    new_u_v(0, 0) = fx * x1_y1(0, 0) + cx;
    new_u_v(1, 0) = fy * x1_y1(1, 0) + cy;
    double r = x1_y1.squaredNorm();
    double distortion = 1.0 + r * (k1 + k2 * r);
    Matrix<double, 2, 1> distorted_new_u_v;
    distorted_new_u_v(0, 0) = fx * x1_y1(0, 0) * distortion + cx;
    distorted_new_u_v(1, 0) = fy * x1_y1(1, 0) * distortion + cy;
    cout << "distorted u v = " << u_v << endl;
    cout << "My: undistorted u v = " << new_u_v << endl;
    cout << "distort by hand u v = " << distorted_new_u_v << endl;

    vector<Point2f> undistort_p, distort_p;
    distort_p.push_back(Point2f(u_v(0, 0), u_v(1, 0)));
    double a = distort_p[0].x;
    //cout << a << endl;
    Mat K = (Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
    //cout << K << endl;
    vector<double> D;
    D.push_back(k1);
    D.push_back(k2);
    D.push_back(0);
    D.push_back(0);
    cv::undistortPoints(distort_p, undistort_p, K, D);
    cout << "OpenCV:undistortPoints undistorted u v = " << endl;
    cout << fx * undistort_p[0].x + cx << endl;
    cout << fy * undistort_p[0].y + cy << endl;
    cout << undistort_p << endl;

    return 0;
}