cmake_minimum_required(VERSION 3.13)
project(3D_Frontiers)

set(CMAKE_CXX_STANDARD 14)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)

add_executable(3D_Frontiers 3d_frontiers.cpp 3d_frontiers.h)
target_link_libraries (3D_Frontiers Eigen3::Eigen)