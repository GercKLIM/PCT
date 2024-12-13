cmake_minimum_required(VERSION 3.18)  # Минимальная версия для поддержки CUDA
project(Lab5 LANGUAGES CXX CUDA)

# "-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc" надо добавить в Cmake options
# Cтандарт C++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Архитектура CUDA
set(CMAKE_CUDA_ARCHITECTURES 86)

include_directories(include)
include_directories(../INPUT)
include_directories(../OUTPUT)

# Исполняемая цель
add_executable(Lab5

        main.cpp

        src/functions.cpp      # Реализация функций импорта-экспорта
        src/cuda-functions.cu  # Реализация вычислений на CUDA
        src/runge-kutta.cu     # Реализация метода Рунге-Кутты

        include/N-body-problem-cuda.cuh

        #json

        include/json/json-forwards.h
        include/json/json.h
        include/json/jsoncpp.cpp

)

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
link_directories(${CUDA_LIBRARIES})

