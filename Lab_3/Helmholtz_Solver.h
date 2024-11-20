/* --------------------------------------------------------- */
/* ### ОБЪЯВЛЕНИЕ ФУНКЦИЙ РЕШЕНИЯ УРАВНЕНИЯ ГЕЛЬМГОЛЬЦА  ### */
/* --------------------------------------------------------- */


#pragma once
#include <iostream>
#include <vector> // Для доступа к типу std::vector
#include "omp.h"  // Для распараллеливания вычислений
#include <ctime>  // Для инициализации генератора
#include <fstream>// Для работы с файлами
#include <cmath>  // Для математических функций
#include <functional> // Для передачи функторов
#include <iomanip> // Для setprecision

#include "mpi.h"

/** Функция Нормы разности векторов
 * (NOD - Norm Of Difference)
 * @param size - Длина вектора
 * @param h    - Шаг по узлам
 * @param A    - Первый вектор
 * @param B    - Второй вектор
 * @return     Норму разности векторов
 */
double NOD(const int& size, const double& h, const std::vector<double>& A, const std::vector<double>& B);


/* Структура для определения выходной информации о работе метода */
struct MethodResultInfo {
    int iterations = 0;     // Итоговое кол-во итераций
    double time = 0;        // Время работы алгоритма
    double norm_iter = 100; // Норма приближений решений на последней итерации
    std::vector<double> Y;  // Итоговое решение
    std::vector<double> Yp; // Предыдущее итоговому приближенное решение
};



/** Метод решения двумерного уравнения Гельмгольца методом Якоби
 * @param y - массив решения
 * @param f - функция правой части
 * @param k - коэффициент в уравнении
 * @param N - число разбиений
 * @param max_num_iterations - Макс. кол-во итераций
 * @return  структуру с информацией о работе метода
 */
MethodResultInfo Method_Jacobi(std::vector<double>& y, std::function<double(double, double)>&f, const double& k, const int& N,
                   const double& eps, const int& max_num_iterations = 1000);

MethodResultInfo Method_Jacobi_P2P(std::vector<double>& y, std::function<double(double, double)>&f, const double& k, const int& N,
                               const double& eps, const int& max_num_iterations = 1000);

MethodResultInfo Method_Jacobi_SIMULT(std::vector<double>& y, std::function<double(double, double)>&f, const double& k, const int& N,
                               const double& eps, const int& max_num_iterations = 1000);

MethodResultInfo Method_Jacobi_NOBLOCK(std::vector<double>& y, std::function<double(double, double)>&f, const double& k, const int& N,
                               const double& eps, const int& max_num_iterations = 1000);


/** Метод Зейделя (красно-черных итераций)
 *  решения двумерного уравнения Гельмгольца
 * @param y - массив решения
 * @param f - функция правой части
 * @param k - коэффициент в уравнении
 * @param N - число разбиений
 * @param max_num_iterations - Макс. кол-во итераций
 * @return  структуру с информацией о работе метода
 */
MethodResultInfo Method_Zeidel(std::vector<double>& y, std::function<double(double, double)>&f, const double& k,
                               const int& N, const double& eps, const int& max_num_iterations);

/** Функция для проверки корректности решения уравнения
 * @param N - Кол-во узлов в решении
 * @param y - Вектор численного решения
 * @param True_sol_func - Функция точного решения
 * @return Норму разности точного и численного решения
 */
double test_sol(const int& N, const std::vector<double>& y, std::function<double(double, double)>& True_sol_func);
