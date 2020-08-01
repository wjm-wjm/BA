# BA
日期7.24：  
前面几天把《视觉slam14讲》里与bundle adjustment相关的内容看了一遍，公式推导了一下，然后还熟悉了一下vs code的编程环境（之前没有用过），自己又按照书上的流程写了一部分代码，并且安装了Cerse solver。根据学长的建议后面打算先去看看Ceres是怎么实现的。

日期7.27：  
把BA实现的流程大致写了一下。把problem-16-22106-pre.txt预处理了一下，将其中的focal和畸变系数，换成了ceres优化以后的。方便后面实现BA优化。

日期7.29:  
初步编写完成，还在调试。

日期8.1:  
数据有三种：big_data.txt(cameras:16 points:22106 observations:83718)、mini_data.txt(cameras:16 points:1000 observations:8037)、mini_data_2.txt(cameras:16 points:6 observations:52)。  
gen_raw_data.cpp产生的数据在raw_data文件夹中（将三种数据从原始数据（在original_data文件夹中的problem-16-22106-pre.txt）提取出来）。  
gen_opt_data.cpp产生的数据在opt_data文件夹中（将raw_data文件夹的数据中相机的内参数改为用ceres优化好的）。  
gen_realtrue_data.cpp产生的数据在realtrue_data文件夹中（直接用ceres优化好的全部数据）。  
ceres_opt_data文件夹是ceres_ba.cpp测试用的数据（因为snavely用的模型的观测数据，z轴是向后的不是向前的，所以得把u、v都取反）。  
test文件夹中是之前的那个test.cpp（用LM算法拟合曲线y=e^(mx+c)）。  
其中gen_opt_data.cpp、gen_realtrue_data.cpp与ceres_ba.cpp需要cmake tool编译（见CMakeLists.txt）。  
bx_axisangle.cpp可以运行但还在调整。
