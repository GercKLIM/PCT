cmake_minimum_required(VERSION 3.27)
project(Lab_2)

#set(CMAKE_CXX_STANDARD)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -fopenmp")

add_executable(Lab_2 main.cpp
        Helmholtz_Solver.cpp
        Helmholtz_Solver.h
)