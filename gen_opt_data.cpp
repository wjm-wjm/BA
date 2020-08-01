#include <cmath>
#include <cstdio>
#include <iostream>

#include "ceres/ceres.h"
#include "ceres/rotation.h"

// Read a Bundle Adjustment in the Large dataset.
class BALProblem {
 public:
  ~BALProblem() {
    delete[] point_index_;
    delete[] camera_index_;
    delete[] observations_;
    delete[] parameters_;
  }

  int num_observations()       const { return num_observations_;               }
  const double* observations() const { return observations_;                   }
  double* mutable_cameras()          { return parameters_;                     }
  double* mutable_points()           { return parameters_  + 9 * num_cameras_; }

  double* mutable_camera_for_observation(int i) {
    return mutable_cameras() + camera_index_[i] * 9; //返回的是对应编号i的camera参数的首地址
  }
  double* mutable_point_for_observation(int i) {
    return mutable_points() + point_index_[i] * 3; //camera参数结束后是观测点世界坐标系下的坐标，返回的是对应编号i的观测点坐标的首地址
  }

  bool LoadFile(const char* filename) {
    FILE* fptr = fopen(filename, "r");
    if (fptr == NULL) {
      return false;
    };

    FscanfOrDie(fptr, "%d", &num_cameras_);
    FscanfOrDie(fptr, "%d", &num_points_);
    FscanfOrDie(fptr, "%d", &num_observations_);

    point_index_ = new int[num_observations_];
    camera_index_ = new int[num_observations_];
    observations_ = new double[2 * num_observations_];

    num_parameters_ = 9 * num_cameras_ + 3 * num_points_;
    parameters_ = new double[num_parameters_];

    for (int i = 0; i < num_observations_; ++i) {
      FscanfOrDie(fptr, "%d", camera_index_ + i);
      FscanfOrDie(fptr, "%d", point_index_ + i);
      for (int j = 0; j < 2; ++j) {
        FscanfOrDie(fptr, "%lf", observations_ + 2*i + j);
      }
    }

    for (int i = 0; i < num_parameters_; ++i) {
      FscanfOrDie(fptr, "%lf", parameters_ + i);
    }
    return true;
  }

 //private:
  template<typename T>
  void FscanfOrDie(FILE *fptr, const char *format, T *value) {
    int num_scanned = fscanf(fptr, format, value);
    if (num_scanned != 1) {
      LOG(FATAL) << "Invalid UW data file.";
    }
  }

  int num_cameras_;
  int num_points_;
  int num_observations_;
  int num_parameters_;

  int* point_index_;
  int* camera_index_;
  double* observations_;
  double* parameters_;
};

// Templated pinhole camera model for used with Ceres.  The camera is
// parameterized using 9 parameters: 3 for rotation, 3 for translation, 1 for
// focal length and 2 for radial distortion. The principal point is not modeled
// (i.e. it is assumed be located at the image center).
struct SnavelyReprojectionError {
  SnavelyReprojectionError(double observed_x, double observed_y)
      : observed_x(observed_x), observed_y(observed_y) {}

  template <typename T>
  bool operator()(const T* const camera,
                  const T* const point,
                  T* residuals) const { //对SnavelyReprojectionError()进行函数重载
    // camera[0,1,2] are the angle-axis rotation 采用的是旋转向量（Axis-Angle），其方向与旋转轴一致，长度等于旋转角
    T p[3];
    ceres::AngleAxisRotatePoint(camera, point, p); //只去camera前三位作为旋转向量，p存储result，point应该是某个点世界坐标系下的坐标【转换到相机坐标系】

    // camera[3,4,5] are the translation.
    p[0] += camera[3];
    p[1] += camera[4];
    p[2] += camera[5];

    // Compute the center of distortion. The sign change comes from 计算畸变中心
    // the camera model that Noah Snavely's Bundler assumes, whereby
    // the camera coordinate system has a negative z axis.
    // 归一化平面上的像素坐标，z轴是negative的
    T xp = - p[0] / p[2];
    T yp = - p[1] / p[2];

    // Apply second and fourth order radial distortion.
    const T& l1 = camera[7];
    const T& l2 = camera[8];
    T r2 = xp*xp + yp*yp;//将[x,y]写成极坐标形式[r,theta]，这里r2=r
    T distortion = 1.0 + r2  * (l1 + l2  * r2);//两个参数k1、k2的径向畸变

    // Compute final projected point position.
    const T& focal = camera[6];
    T predicted_x = focal * distortion * xp;
    T predicted_y = focal * distortion * yp; //fx=fy=focal，cx=cy=0

    // The error is the difference between the predicted and observed position.
    residuals[0] = predicted_x - observed_x; //residuals也就是error
    residuals[1] = predicted_y - observed_y;

    return true;
  }

  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction* Create(const double observed_x,
                                     const double observed_y) {
    return (new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 2, 9, 3>(
                new SnavelyReprojectionError(observed_x, observed_y)));
  }

  double observed_x;
  double observed_y;
};

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  if (argc != 2) {
    std::cerr << "usage: simple_bundle_adjuster <bal_problem>\n";
    return 1;
  }

  BALProblem bal_problem;
  if (!bal_problem.LoadFile(argv[1])) {
    std::cerr << "ERROR: unable to open file " << argv[1] << "\n";
    return 1;
  }

  const double* observations = bal_problem.observations();

  // Create residuals for each observation in the bundle adjustment problem. The
  // parameters for cameras and points are added automatically.
  ceres::Problem problem;
  for (int i = 0; i < bal_problem.num_observations(); ++i) {
    // Each Residual block takes a point and a camera as input and outputs a 2
    // dimensional residual. Internally, the cost function stores the observed
    // image location and compares the reprojection against the observation.

    ceres::CostFunction* cost_function =
        SnavelyReprojectionError::Create(observations[2 * i + 0],
                                         observations[2 * i + 1]);
    problem.AddResidualBlock(cost_function,
                             NULL /* squared loss */,
                             bal_problem.mutable_camera_for_observation(i),
                             bal_problem.mutable_point_for_observation(i));
  }

  // Make Ceres automatically detect the bundle structure. Note that the
  // standard solver, SPARSE_NORMAL_CHOLESKY, also works fine but it is slower
  // for standard bundle adjustment problems.
  // 配置并运行求解器
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_SCHUR; //配置增量方程的解法schur
  options.minimizer_progress_to_stdout = true; //输出到cout

  ceres::Solver::Summary summary; //优化信息
  ceres::Solve(options, &problem, &summary); //求解
  std::cout << summary.FullReport() << "\n"; //输出优化的全部信息

  FILE *f1 = fopen("/home/vision/Desktop/code_c_c++/my_BA/raw_data/big_data.txt", "r");
  //FILE *f1 = fopen("/home/vision/Desktop/code_c_c++/my_BA/raw_data/mini_data.txt", "r");
  //FILE *f1 = fopen("/home/vision/Desktop/code_c_c++/my_BA/raw_data/mini_data_2.txt", "r");

  FILE *f2 = fopen("/home/vision/Desktop/code_c_c++/my_BA/opt_data/big_data.txt", "w");
  //FILE *f2 = fopen("/home/vision/Desktop/code_c_c++/my_BA/opt_data/mini_data.txt", "w");
  //FILE *f2 = fopen("/home/vision/Desktop/code_c_c++/my_BA/opt_data/mini_data_2.txt", "w");

  FILE *f3 = fopen("/home/vision/Desktop/code_c_c++/my_BA/ceres_opt_data/big_data.txt", "w");
  //FILE *f3 = fopen("/home/vision/Desktop/code_c_c++/my_BA/ceres_opt_data/mini_data.txt", "w");
  //FILE *f3 = fopen("/home/vision/Desktop/code_c_c++/my_BA/ceres_opt_data/mini_data_2.txt", "w");

  int n_camera, n_point, n_observation;
  fscanf(f1, "%d", &n_camera);
  fscanf(f1, "%d", &n_point);
  fscanf(f1, "%d", &n_observation);
  fprintf(f2, "%d %d %d\n", n_camera, n_point, n_observation);
  fprintf(f3, "%d %d %d\n", n_camera, n_point, n_observation);

  for (int i = 0; i < n_observation;i++){
      int c_id, p_id;
      double u, v;
      fscanf(f1, "%d%d%lf%lf", &c_id, &p_id, &u, &v);
      fprintf(f2, "%d %d %.6e %.6e\n", c_id, p_id, -u, -v);
      fprintf(f3, "%d %d %.6e %.6e\n", c_id, p_id, u, v);
  }

  for (int i = 0; i < n_camera;i++){
      for (int j = 0; j < 9;j++){
          double c;
          fscanf(f1, "%lf", &c);
          if(j==6 || j==7 || j==8){
              fprintf(f2, "%.16e\n", *(bal_problem.parameters_ + i * 9 + j));
              fprintf(f3, "%.16e\n", *(bal_problem.parameters_ + i * 9 + j));
              continue;
          }
          fprintf(f2, "%.16e\n", c);
          fprintf(f3, "%.16e\n", c);
      }
  }

  for (int i = 0; i < n_point;i++){
      for (int j = 0; j < 3;j++){
          double p;
          fscanf(f1, "%lf", &p);
          fprintf(f2, "%.16e\n", p);
          fprintf(f3, "%.16e\n", p);
      }
  }

  fclose(f1);
  fclose(f2);
  fclose(f3);

  return 0;
}