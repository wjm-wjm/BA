# 声明要求的cmake最低版本
cmake_minimum_required( VERSION 2.8 )

# 声明一个cmake工程
project( ceres_opencv_sophus )

find_package( Ceres REQUIRED )
find_package( OpenCV REQUIRED )
find_package( Sophus REQUIRED )

include_directories( ${CERES_INCLUDE_DIRS} )
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories( ${Sophus_INCLUDE_DIRS} )

# 添加一个可执行程序
# 语法:add_executable( 程序名 源代码文件 )
add_executable( gen_opt_data gen_opt_data.cpp )
add_executable( gen_realtrue_data gen_realtrue_data.cpp )
add_executable( ceres_ba ceres_ba.cpp )
add_executable( undistort_newton undistort_newton.cpp )
add_executable( AngleAxisRotatePoint_byhand AngleAxisRotatePoint_byhand.cpp )
add_executable( se3_exp_log_byhand se3_exp_log_byhand.cpp )
add_executable( ba_axisangle_bigdata_ ba_axisangle_bigdata_.cpp )
add_executable( ba_axisangle ba_axisangle.cpp )

target_link_libraries( gen_opt_data ${CERES_LIBRARIES} )
target_link_libraries( gen_realtrue_data ${CERES_LIBRARIES} )
target_link_libraries( ceres_ba ${CERES_LIBRARIES} )
target_link_libraries( AngleAxisRotatePoint_byhand ${CERES_LIBRARIES} )

target_link_libraries( undistort_newton ${OpenCV_LIBS} )

target_link_libraries( se3_exp_log_byhand ${Sophus_LIBRARIES} )

