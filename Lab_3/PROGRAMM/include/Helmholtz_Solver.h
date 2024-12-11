
/* --------------------------------------------------------- */
/* ### ОБЪЯВЛЕНИЕ ФУНКЦИЙ РЕШЕНИЯ УРАВНЕНИЯ ГЕЛЬМГОЛЬЦА  ### */
/* --------------------------------------------------------- */


#pragma once

/* ### Стандартные библиотеки ###*/
#include <iostream>
#include <vector>     // Для доступа к типу std::vector
#include "omp.h"      // Для распараллеливания вычислений
#include <ctime>      // Для инициализации генератора
#include <fstream>    // Для работы с файлами
#include <cmath>      // Для математических функций
#include <functional> // Для передачи функторов
#include <iomanip>    // Для setprecision

/* ### Прикрученные библиотеки ###*/
#include "mpi.h"      // Для распараллеливания между процессорами
#include "omp.h"      // Для распараллеливания вычислений

/* ### Импортированные библиотеки ###*/
#include <json/json.h> // Для работы с json файлами


/* ### ПОБОЧНЫЕ ФУНКЦИИ ### */





/** Функция Нормы разности векторов
 * (NOD - Norm Of Difference)
 * @param size - Длина вектора
 * @param h    - Шаг по узлам
 * @param A    - Первый вектор
 * @param B    - Второй вектор
 * @return     Норму разности векторов
 */
double NOD(const int& size, const double& h, const std::vector<double>& A, const std::vector<double>& B);



/** Функция получения параметров из .json файла
 *
 *
 * @param filename         - Имя .json файла для импорта данных
 * @param NP               - Кол-во потоков
 * @param N                - Кол-во узлов
 * @param K                - Коэф. уравнения K
 * @param max_iterations   - Ограничение кол-ва итераций
 * @param EPS              - Допустимая погрешность
 * @param test_name        - название тестового набора параметров
 * @return
 */
bool input_parametres(const std::string& filename, int& N, double& K, int& max_iterations, double& EPS,
                      std::string& test_name);



/* Структура для определения выходной информации о работе метода */
struct MethodResultInfo {

    int NP = 1;                    // Число потоков, используемое методом
    int iterations = 0;            // Итоговое кол-во итераций
    double time = 0;               // Время работы алгоритма
    double norm_iter = 100;        // Норма приближений решений на последней итерации
    std::vector<double> Y;         // Итоговое решение
    std::vector<double> Yp;        // Предыдущее итоговому приближенное решение
    std::string method_name = " "; // Название используемого метода
    double norm_sol = 100;         // Норма разности с точным решением
};

/* Вывод результатов метода */
void print_MethodResultInfo(const MethodResultInfo& MR);


/* Вывод результатов метода в файл */
void print_MethodResultInfoFile(const MethodResultInfo& MR, std::ofstream& Fout);


/** Функция для проверки корректности решения уравнения
 * @param N - Кол-во узлов в решении
 * @param y - Вектор численного решения
 * @param True_sol_func - Функция точного решения
 * @return Норму разности точного и численного решения
 */
double test_sol(const int& N, const std::vector<double>& y, std::function<double(double, double)>& True_sol_func);



/** Функция раздача работы */
void Work_Distribution(int NP, int N, int& str_local, int& nums_local,
                       std::vector<int>& str_per_proc, std::vector<int>& nums_start);



/* ### МЕТОДЫ РЕШЕНИЯ С OMP ### */




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



/* ### МЕТОДЫ РЕШЕНИЯ С MPI ### */


/** Функция решения уравнения Гельмогольца
 *  методом Якоби
 *  с импольщованием пересылов Send Recv
 *
 * @param result         - Структура результатов алгоритма
 * @param f              - Функция правой части
 * @param K              - Коэф. К уравнения
 * @param N              - Кол-во узлов
 * @param eps            - Точность
 * @param max_iterations - Ограничение итераций
 */
void Method_Jacobi_P2P(MethodResultInfo& result, std::function<double(double, double)>&f, const double& K, const int& N,
                       const double& eps, const int& max_iterations);


/** Функция решения уравнения Гельмогольца
 *  методом Якоби
 *  с импольщованием пересылов SendRecv
 *
 * @param result         - Структура результатов алгоритма
 * @param f              - Функция правой части
 * @param K              - Коэф. К уравнения
 * @param N              - Кол-во узлов
 * @param eps            - Точность
 * @param max_iterations - Ограничение итераций
 */
void Method_Jacobi_SIMULT(MethodResultInfo& result, std::function<double(double, double)>&f, const double& K, const int& N,
                          const double& eps, const int& max_iterations);


/** Функция решения уравнения Гельмогольца
 *  методом Якоби
 *  с импольщованием пересылов ISend IRecv
 *
 * @param result         - Структура результатов алгоритма
 * @param f              - Функция правой части
 * @param K              - Коэф. К уравнения
 * @param N              - Кол-во узлов
 * @param eps            - Точность
 * @param max_iterations - Ограничение итераций
 */
void Method_Jacobi_NOBLOCK(MethodResultInfo& result, std::function<double(double, double)>&f, const double& K, const int& N,
                           const double& eps, const int& max_iterations);


/** Функция решения уравнения Гельмогольца
 *  методом Зейделя
 *  с импольщованием пересылов Send Recv
 *
 * @param result         - Структура результатов алгоритма
 * @param f              - Функция правой части
 * @param K              - Коэф. К уравнения
 * @param N              - Кол-во узлов
 * @param eps            - Точность
 * @param max_iterations - Ограничение итераций
 */
void Method_Zeidel_P2P(MethodResultInfo& result, std::function<double(double, double)>&f, const double& K, const int& N,
                       const double& eps, const int& max_iterations);


/** Функция решения уравнения Гельмогольца
 *  методом Зейделя
 *  с импольщованием пересылов SendRecv
 *
 * @param result         - Структура результатов алгоритма
 * @param f              - Функция правой части
 * @param K              - Коэф. К уравнения
 * @param N              - Кол-во узлов
 * @param eps            - Точность
 * @param max_iterations - Ограничение итераций
 */
void Method_Zeidel_SIMULT(MethodResultInfo& result, std::function<double(double, double)>&f, const double& K, const int& N,
                          const double& eps, const int& max_iterations);


/** Функция решения уравнения Гельмогольца
 *  методом Зейделя
 *  с импольщованием пересылов ISend IRecv
 *
 * @param result         - Структура результатов алгоритма
 * @param f              - Функция правой части
 * @param K              - Коэф. К уравнения
 * @param N              - Кол-во узлов
 * @param eps            - Точность
 * @param max_iterations - Ограничение итераций
 */
void Method_Zeidel_NOBLOCK(MethodResultInfo& result, std::function<double(double, double)>&f, const double& K, const int& N,
                           const double& eps, const int& max_iterations);