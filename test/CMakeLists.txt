# 声明要求的cmake最低版本
cmake_minimum_required( VERSION 2.8 )

# 声明一个cmake工程
project( ba_sophus_ )

find_package( Sophus REQUIRED )
include_directories( ${Sophus_INCLUDE_DIRS} )

# 添加一个可执行程序
# 语法:add_executable( 程序名 源代码文件 )
add_executable( test3_ba_sophus test3_ba_sophus.cpp )
add_executable( ba_axisangle_test ba_axisangle_test.cpp )
target_link_libraries( test3_ba_sophus ${Sophus_LIBRARIES} )
target_link_libraries( ba_axisangle_test ${Sophus_LIBRARIES} )
