cmake_minimum_required(VERSION 3.10)
project(LAB_4)

set(CMAKE_CXX_STANDARD 17)

# Подключаем OMP
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -fopenmp")

# Найдем MPI
find_package(MPI REQUIRED)
# Укажем, что проект использует MPI
include_directories(${MPI_INCLUDE_PATH})

add_executable(LAB_4
        main.cpp

        include/N-BodyProblem.h

        src/functions.cpp
        src/N-BodyProblem_MPI.cpp
)

# Свяжем с MPIMain
#target_link_libraries(Main_MPI ${MPI_LIBRARIES})
target_link_libraries(LAB_4 PRIVATE MPI::MPI_CXX)


# Добавляем библиотеку для работы с JSON файлами
include(FetchContent)
FetchContent_Declare(
        json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG v3.11.2 # Замените на последнюю стабильную версию
)
FetchContent_MakeAvailable(json)
target_link_libraries(LAB_4 PRIVATE nlohmann_json::nlohmann_json)