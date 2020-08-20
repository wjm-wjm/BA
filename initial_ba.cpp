#include<bits/stdc++.h>
#include<fstream>

#include<Eigen/Core>
#include<Eigen/Dense>

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

int main(){
    int num_camera_parameters = 9; 
    
    load_data f(num_camera_parameters);
    f.load_file("/home/vision/Desktop/code_c_c++/my_BA/raw_data/big_data.txt");

    WriteToPLYFile("/home/vision/Desktop/code_c_c++/my_BA/3d_point_original.ply", &f);

    return 0;
}