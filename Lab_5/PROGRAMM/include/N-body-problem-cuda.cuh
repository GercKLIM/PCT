#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <random>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <array>
#include <string>
#include <iomanip>
#include <stdio.h>

#include <nlohmann/json.hpp> // Для работы с json файлами



// Используемый тип данных для вычислений
typedef float mytype;



// Гравитационная постоянная и параметр стабилизации
constexpr mytype G = -6.67e-11;
constexpr mytype eps = 1e-6;
constexpr int BS = 32; // Размер блока для CUDA



/**
 * @brief Считывает входные данные из файла.
 * @param path Путь к файлу.
 * @param global_m Вектор масс.
 * @param global_r Вектор начальных координат.
 * @param global_v Вектор начальных скоростей.
 * @param N Число тел.
 */
bool read(const std::string& path, std::vector<mytype>& global_m, std::vector<mytype>& global_r,
          std::vector<mytype>& global_v, int& N);


/**
 * @brief Записывает текущие координаты тела в файл.
 * @param path Базовый путь к файлу.
 * @param r Вектор координат тела.
 * @param t Текущее время.
 * @param number Номер тела.
 */
__host__ bool write(const std::string& path, const std::vector<mytype>& r, mytype t, int number);


/**
 * @brief Очищает файлы вывода перед началом записи.
 * @param path Базовый путь к файлам.
 * @param N Число тел.
 */
__host__ void clear_files(const std::string& path, int N);



/** Функция получения параметров из .json файла */
template <typename type>
bool input_json(const std::string& filename, const std::string& name, type& pts) {
    std::ifstream config_file(filename);
    if (!config_file.is_open()) {
        std::cerr << "[LOG]: ERROR: Unable to open " << filename << std::endl;
        return false;
    }
    nlohmann::json config;
    try {
        config_file >> config;
    } catch (const nlohmann::json::parse_error& e) {
        std::cerr << "[LOG]: ERROR: Parsing error: " << e.what() << "\n";
        return false;
    }
    try {
        pts = config.at(name).get<type>();
    } catch (const nlohmann::json::out_of_range& e) {
        std::cerr << "[LOG]: ERROR: Key \"" << name << "\" not found: " << e.what() << "\n";
        return false;
    }
    return true;
}


/**
 * @brief Вычисляет квадрат нормы вектора для ускорения.
 * @param x1 Компонента x.
 * @param x2 Компонента y.
 * @param x3 Компонента z.
 * @return Куб нормы вектора.
 */
__device__ mytype norm(mytype x1, mytype x2, mytype x3);


/**
 * @brief Основной CUDA-ядро для вычисления производных координат и скоростей.
 * @param kr Вектор изменения координат.
 * @param kv Вектор изменения скоростей.
 * @param device_m Массы тел на устройстве.
 * @param device_r Координаты тел на устройстве.
 * @param device_v Скорости тел на устройстве.
 * @param N Число тел.
 */
__global__ void f(mytype* kr, mytype* kv, mytype* device_m, mytype* device_r, mytype* device_v, int N);


/**
 * @brief Обновляет координаты тел с использованием данных Runge-Kutta.
 * @param device_r Исходные координаты.
 * @param kr Изменения координат.
 * @param tau Шаг интегрирования.
 * @param temp_device_r Обновленные координаты.
 * @param N Число тел.
 */
__global__ void add(mytype* device_r, mytype* kr, mytype tau, mytype* temp_device_r, int N);


/**
 * @brief Итоговая коррекция координат и скоростей.
 * @param device_r Координаты тел.
 * @param device_v Скорости тел.
 * @param tau Шаг интегрирования.
 * @param kr1..kr4 Вектора изменений координат.
 * @param kv1..kv4 Вектора изменений скоростей.
 * @param N Число тел.
 */
__global__ void summarize(mytype* device_r, mytype* device_v, mytype tau,
                          mytype* kr1, mytype* kv1,
                          mytype* kr2, mytype* kv2,
                          mytype* kr3, mytype* kv3,
                          mytype* kr4, mytype* kv4, int N);


/**
 * @brief Метод Рунге-Кутта для решения задачи о движении N тел с использованием CUDA.
 * @param path Путь к файлу для вывода данных.
 * @param global_m Массы тел.
 * @param global_r Начальные координаты тел.
 * @param global_v Начальные скорости тел.
 * @param tau Шаг интегрирования.
 * @param T Время интегрирования.
 * @param output Флаг записи результата.
 * @return Среднее время выполнения одного шага.
 */
float Runge_Kutta(const std::string& path, const std::vector<mytype>& global_m,
                  std::vector<mytype>& global_r, std::vector<mytype>& global_v,
                  mytype tau, mytype T, bool output);
