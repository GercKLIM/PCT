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

#include <json/json.h> // Для работы с json файлами



// Используемый тип данных для вычислений
typedef double mytype;
typedef double3 mytype3;
//typedef float mytype;
//typedef float3 mytype3;



// Гравитационная постоянная и параметр стабилизации
constexpr mytype G = -6.67e-11;
constexpr mytype eps = 1e-6;
constexpr int BS = 128; // Размер блока для CUDA



__device__ float3 operator+(const float3& a, const float3& b);

__device__ float3 operator-(const float3& a, const float3& b);

__device__ float3 operator*(const float3& a, float b);


__device__ double3 operator+(const double3& a, const double3& b);

__device__ double3 operator-(const double3& a, const double3& b);

__device__ double3 operator*(const double3& a, float b);


/**
 * @brief Считывает входные данные из файла.
 * @param path Путь к файлу.
 * @param global_m Вектор масс.
 * @param global_r Вектор начальных координат.
 * @param global_v Вектор начальных скоростей.
 * @param N Число тел.
 */
bool read(const std::string& path, std::vector<mytype>& global_m,
                                  std::vector<mytype3>& global_r,
                                  std::vector<mytype3>& global_v, int& N);


/**
 * @brief Записывает текущие координаты тела в файл.
 * @param path Базовый путь к файлу.
 * @param r Вектор координат тела.
 * @param t Текущее время.
 * @param number Номер тела.
 */
__host__ bool write(const std::string& path, const mytype3& r, mytype t, int number);


/**
 * @brief Очищает файлы вывода перед началом записи.
 * @param path Базовый путь к файлам.
 * @param N Число тел.
 */
__host__ void clear_files(const std::string& path, int N);



/** Функция получения параметров из .json файла */
bool input_parametres(const std::string& filename, std::string& test_filename, std::string& output_filename,
                      mytype& T, mytype&tau, mytype& EPS, bool& output, int& max_iterations);


///**
// * @brief Вычисляет квадрат нормы вектора для ускорения.
// * @param x1 Компонента x.
// * @param x2 Компонента y.
// * @param x3 Компонента z.
// * @return Куб нормы вектора.
// */
//__device__ mytype norm(const mytype3& v);

/**
 * @brief Вычисляет квадрат нормы вектора для ускорения.
 * @param v Вектор типа mytype3.
 * @return Куб нормы вектора.
 */
__device__ float norm(const float3& v);


/**
 * @brief Вычисляет квадрат нормы вектора для ускорения.
 * @param v Вектор типа mytype3.
 * @return Куб нормы вектора.
 */
__device__ double norm(const double3& v);


/**
 * @brief Основной CUDA-ядро для вычисления производных координат и скоростей.
 * @param kr Вектор изменения координат.
 * @param kv Вектор изменения скоростей.
 * @param device_m Массы тел на устройстве.
 * @param device_r Координаты тел на устройстве.
 * @param device_v Скорости тел на устройстве.
 * @param N Число тел.
 */
__global__ void f(mytype3* kr, mytype3* kv, mytype* device_m, mytype3* device_r, mytype3* device_v, int N);



/**
 * @brief Обновляет координаты тел с использованием данных Runge-Kutta.
 * @param device_r Исходные координаты.
 * @param kr Изменения координат.
 * @param tau Шаг интегрирования.
 * @param temp_device_r Обновленные координаты.
 * @param N Число тел.
 */
__global__ void
add(mytype3 *device_r, mytype3 *device_v, mytype3 *kr, mytype3 *kv, mytype3 *temp_device_r, mytype3 *temp_device_v,
    int N, mytype tau);



/**
 * @brief Итоговая коррекция координат и скоростей.
 * @param device_r Координаты тел.
 * @param device_v Скорости тел.
 * @param tau Шаг интегрирования.
 * @param kr1..kr4 Вектора изменений координат.
 * @param kv1..kv4 Вектора изменений скоростей.
 * @param N Число тел.
 */
__global__ void summarize(mytype3* device_r, mytype3* device_v, mytype tau,
                          mytype3* kr1, mytype3* kv1,
                          mytype3* kr2, mytype3* kv2,
                          mytype3* kr3, mytype3* kv3,
                          mytype3* kr4, mytype3* kv4, int N);


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
                  std::vector<mytype3>& global_r, std::vector<mytype3>& global_v,
                  mytype tau, mytype T, bool output);
