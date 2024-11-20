cmake_minimum_required(VERSION 3.27)
project(Lab_3)

set(CMAKE_CXX_STANDARD 17)

# Подключаем OMP
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -fopenmp")

# Найдем MPI
find_package(MPI REQUIRED)
# Укажем, что проект использует MPI
include_directories(${MPI_INCLUDE_PATH})

add_executable(lab3_MPI
        main1.cpp
        Helmholtz_Solver.cpp
        Helmholtz_Solver.h
)

add_executable(Main_MPI
        main.cpp
)

# Свяжем с MPIMain
#target_link_libraries(Main_MPI ${MPI_LIBRARIES})
target_link_libraries(Main_MPI PRIVATE MPI::MPI_CXX)
target_link_libraries(lab3_MPI PRIVATE MPI::MPI_CXX)


# Включите вывод команды для MPI
#enable_testing()
# Добавьте тест для MPI с использованием mpiexec
#add_test(
#        NAME RunMPI
#        COMMAND mpiexec -n 4 $<TARGET_FILE:Main_MPI>
#)
