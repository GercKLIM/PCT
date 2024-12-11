cmake_minimum_required(VERSION 3.27)
project(Lab_3)

set(CMAKE_CXX_STANDARD 17)

# Подключаем OMP
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -fopenmp")

# Найдем MPI
find_package(MPI REQUIRED)

# Добавим директорию MPI
include_directories(${MPI_INCLUDE_PATH})

# Добавим директорию header's файлов
include_directories(include)


# Общая цель - компоновка под библиотеку
add_executable(Lab3

        main.cpp

        ### HEADERS
        include/Helmholtz_Solver.h

        ### REALIZATIONS
        src/functions.cpp
        src/Helmholtz_Solver_OMP.cpp
        src/Helmholtz_Solver_MPI.cpp

        #json

        include/json/json-forwards.h
        include/json/json.h
        include/json/jsoncpp.cpp

        ### TESTS
        ##=tests/test_omp.cpp
        #tests/test_mpi.cpp
)

# Свяжем с MPIMain
# target_link_libraries(Helmholtz_Solver ${MPI_LIBRARIES})
target_link_libraries(Lab3 PUBLIC MPI::MPI_CXX)


# Добавляем библиотеку для работы с JSON файлами
#include(FetchContent)
#FetchContent_Declare(
#        json
#        GIT_REPOSITORY https://github.com/nlohmann/json.git
#        GIT_TAG v3.11.2 # Замените на последнюю стабильную версию
#)
#FetchContent_MakeAvailable(json)
#target_link_libraries(Helmholtz_Solver PRIVATE nlohmann_json::nlohmann_json)


# Добавляем библиотеку Логов
#FetchContent_Declare(
#        Logs
#        GIT_REPOSITORY https://github.com/GercKLIM/Logs.git
#        GIT_TAG main
#)
#FetchContent_MakeAvailable(Logs)
#target_link_libraries(Helmholtz_Solver PRIVATE Logs)


