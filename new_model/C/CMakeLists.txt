cmake_minimum_required(VERSION 3.3.2)

project(C_check)

add_definitions("-g")
include_directories(/local-scratch/Xilinx/Vitis_HLS/2020.2/include)
add_executable(C_check src/host.cpp src/lc_mgvf.cpp)