# 声明要求的cmake最低版本
cmake_minimum_required( VERSION 2.8 )

# 声明一个cmake工程
project( ceres_ )

find_package(Ceres REQUIRED)

include_directories( ${CERES_INCLUDE_DIRS} )

# 添加一个可执行程序
# 语法:add_executable( 程序名 源代码文件 )
add_executable( gen_opt_data gen_opt_data.cpp )
add_executable( gen_realtrue_data gen_realtrue_data.cpp )
add_executable( ceres_ba ceres_ba.cpp )

target_link_libraries( gen_opt_data ${CERES_LIBRARIES} )
target_link_libraries( gen_realtrue_data ${CERES_LIBRARIES} )
target_link_libraries( ceres_ba ${CERES_LIBRARIES} )
