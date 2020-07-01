cmake_minimum_required(VERSION 3.0.0)
project(imslic_project VERSION 0.1.0)

set(CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED true)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mtune=native")

add_subdirectory(npy_array)
add_subdirectory(thread_pool)

add_executable(imslic_project main.cpp)
