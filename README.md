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

日期8.2:  
ba_axisangle_bigdata.cpp可以跑big_data.txt的数据（通过分块矩阵乘法进行实现的）  
log文件夹中log_mini_data_8_2.txt已经log_big_data_8_2.txt由于文件过大，我放在谷歌云盘上了（https://drive.google.com/drive/folders/1hTAIIY_rsX1jE9l8h2rGUbwIR6i5Tp7V?usp=sharing）。 

日期8.3:  
ba_axisangle.cpp与ba_axisangle_bigdata.cpp可以运行高斯牛顿法，但好像更容易收敛到鞍点（"GN":lambda取大于１的(5、10)，"LM":lambda取小于１(1e-5、1e-4)），同时输出增加了initial error、final error以及error change，log文件夹以及上述谷歌云盘的分享链接有上传例子。

日期8.10:  
ba_axisangle_bigdata_.cpp 解决了一些以前没注意到的问题:  
1. 将真实的2d观测先进行了undistort处理（采用牛顿法进行迭代），这样后面计算Jacobian矩阵时会比较方便一点，不用考虑distortion对Jacobian矩阵的影响（之前一直没注意到distortion会对Jacobian矩阵产生影响，于是只对预测得到的2d观测做了distort处理）。  
2. 之前将增量方程得到的相机位姿的增量直接加到se(3)上，后来发现对相机位姿se(3)的Jacobian矩阵是扰动模型的形式不能直接加，要对增量与原始的se(3)先做exp变换再左乘，再log变换回来，对此我一开始采用了sophus库，后来参照T.Barfoot,"State estimation for robotics: A matrix lie group approach," 2016中的7.76(b)、7.86(b)、7.95(b)的公式自己实现了一下，和sophus相比略微有些小误差（可能是精度问题）。  
![图1](https://github.com/wjm-wjm/BA/blob/master/image/2020-08-10%2023-11-03%20%E7%9A%84%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE.png)  
3. 最后测试big_data.txt（也就是Ceres example中的problem-16-22106-pre.txt）最终的平均重投影误差为438280/83718=5.235194343，Ceres example的simple_bundle_adjuster的平均重投影误差为18033.92/83718=0.215412695，误差还是有点大。（浅色的点是我的结果，深色的点是Ceres的结果）  
![图2](https://github.com/wjm-wjm/BA/blob/master/image/2020-08-10%2023-36-15%20%E7%9A%84%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE.png)  
![图3](https://github.com/wjm-wjm/BA/blob/master/image/2020-08-11%2014-00-46%20%E7%9A%84%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE.png)  
![图4](https://github.com/wjm-wjm/BA/blob/master/image/2020-08-11%2014-03-24%20%E7%9A%84%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE.png)  
4. 然后我还编写了了PCG-J以及PCG-SSOR算法，还在调试。

日期8.12:  
找到了error降不下来的问题了，我之前每次迭代的时候忘记给所有的储存数据的矩阵更新初值了，导致只有第一次优化是正确的，后面就优化不下去了。下面是改进的结果（好很多）:  
![图5](https://github.com/wjm-wjm/BA/blob/master/image/2020-08-13%2000-11-40%20%E7%9A%84%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE.png)  
依旧是浅色的点是我的结果(平均重投影误差为19089.3/83718=0.228019064)，深色的点是Ceres的结果(平均重投影误差为18033.92/83718=0.215412695)，从图中可以看出几乎完全重叠，和之前相比墙也变得很平整:  
![图6](https://github.com/wjm-wjm/BA/blob/master/image/2020-08-13%2000-06-04%20%E7%9A%84%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE.png)  
![图7](https://github.com/wjm-wjm/BA/blob/master/image/2020-08-13%2000-06-25%20%E7%9A%84%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE.png)  
