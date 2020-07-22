#include <iostream>
#include<random>
#include<vector>
#include<Eigen/Core>
#include<Eigen/Dense>
using namespace std;
using namespace Eigen;

class cal_error_Jacobian
{
public:
    cal_error_Jacobian(double x,double y){
        x_ = x;
        y_ = y;
    }

    double cal_error(double m,double c){
        return (exp(m * x_ + c) - y_);
    }

    Matrix<double,1,2> cal_Jacobian(double m,double c){
        Matrix<double, 1, 2> J;
        J[0] = x_ * exp(m * x_ + c);
        J[1] = exp(m * x_ + c);
        return J;
    }
    double x_, y_;
};

/*class LMoptimization
{
public:
    //设置初值
    void setinitialvalue(double m,double c){
        m_ = m;
        c_ = c;
    }
    
    //优化算法主体
    void optimize(int num_iter){
        double lambda = 1e-4;
        for (int i = 0; i < num_iter; i++){
            //构造雅可比和函数值向量，将每一项作为一行
            MatrixXd Jacobian(errorTerms.size(), 2);
            MatrixXd error(errorTerms.size(), 1);
            for (int k = 0; k < (int)errorTerms.size();k++){
                error[k] = errorTerms[k]->cal_error(m_, c_);
                Jacobian.row(k) = errorTerms[k]->cal_Jacobian(m_, c_);
            }
            //构造增量方程
            Matrix2d JTJ = Jacobian.transpose() * Jacobian;
            Matrix2d A = JTJ + lambda * Matrix2d(JTJ.diagonal().asDiagonal());
            Matrix<double, 2, 1> b = -Jacobian * error;
            //求解增量方程Ax=b;
            Matrix<double, 2, 1> delta = A.colPivHouseholderQr().solve(b);
        }
    }

    double m_, c_;
    vector<cal_error_Jacobian *> errorTerms;
};*/

int main()
{
    //用来生成高斯噪声
    std::default_random_engine e;
    std::normal_distribution<double> n(0, 0.2);
    //用来存储数据点
    std::vector<std::pair<double, double>> data;
    //生成数据点，假设x的范围是0-5
    for(double x = 0; x < 5; x += 0.05)
    {
        double y = exp(0.3*x + 0.1)+n(e);
        data.push_back(std::make_pair(x, y));
    }

    /*MatrixXd a(5, 2),b(3,3);
    a.row(1) = MatrixXd::Random(1, 2);
    b = MatrixXd::Zero(3, 3);
    cout << a << endl << b << endl;*/

    //cout << cal_error_Jacobian(2, 2).cal_Jacobian(1, 2) << endl;

    Matrix<double, 2, 1> a;
    a << 3, 4;
    cout << a.norm() << endl;
    //cout << a.diagonal()<< endl;

    return 0;
}