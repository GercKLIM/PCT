/* ### ЛАБОРАТОРНАЯ РАБОТА №1 ###
 *
 *  ЗАДАНИЕ:
 *
 *  Разработать программу для решения двумерного уравнения Гельмгольца
 *
 *  -D[u[x, y], x, x] - D[u[x, y], y, y] + k^2 u = f(x, y)
 *
 *  в квадратной области (x, y) in [0, 1]x[0, 1]
 *
 *  c граничными условиями:
 *  u(x, 0) = u(x, 1) = u(0, y) = u(1, y) = 0
 *
 *  и правой частью
 *  f(x, y) = 2 sin(pi y) + k^2 (1 - x) x sin(pi y) + pi^2 (1 - x) x sin(pi y)
 *
 *  Для численного решения уравнения на прямоугольной равномерной сетке использовать
 *  конечно-разностную схему «крест» второго порядка аппроксимации по обеим независимым переменным.
 *  Полученную систему линейных алгебраических уравнений решить итерационным
 *  методом Якоби и Зейделя (точнее, «красно-черных» итераций).
 *  Обеспечить вывод на экран количества итераций до сходимости и погрешности.
 *
 *  Точное решение задачи: u(x, y) = (1 - x) x sin(pi y)
 *
 *  С использование OpenMP
 * */

#include "Helmholtz_Solver.h"
#include <iostream>
#include <iomanip> // Для setprecision
#include <vector> // Для доступа к типу std::vector
#include "omp.h"  // Для распараллеливания вычислений
#include <ctime>  // Для инициализации генератора
#include <fstream>// Для работы с файлами
#include <cmath>  // Для математических функций



/* -------------------------------- */
/* ### ТЕСТИРОВАНИЕ АЛГОРИТМА  ### */
/* -------------------------------- */

const double PI = 3.14159265358979;
const double EPS = 1e-7;

/* Ограничение кол-ва итераций */
const int MAX_ITERATION = 100;

/* Коэффициент k */
double k = 20;

/* Кол-во узлов в одном направлении */
int N = 10;

/* Правая часть f */
//double f(double x, double y) {
//    return (2 * sin(M_PI * y) + k * k * (1 - x) * x * sin(M_PI * y)
//                    + M_PI * M_PI * (1 - x) * x * sin(M_PI * y));
//}



/* Точное решение задачи u(x, y) */
double TRUE_SOL(double x, double y) {
    return ((1 - x) * x * sin(PI * y));
}

void test() {
    //Method_Jacobi(std::vector<double>& y, std::function<double(double, double)>&f, const double& k, const int& N,
    //const double& eps, const int& max_num_iterations = 1000)

    //double PI = 3.14159265358979;
    std::function<double(double, double)> f = ([&](double x, double y){
        return (2 * sin(PI * y) + k * k * (1 - x) * x * sin(PI * y)
                + PI * PI * (1 - x) * x * sin(PI * y));
    });

    std::vector<double> y(N * N, 0.0);


    // Инициализация решения правой частью
    double h = 1 / (N - 1);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            y[i * N + j] = f(i * h, j * h);
        }
    }

    Method_Jacobi(y, f, k, N, EPS, MAX_ITERATION);

    std::vector<double> y_true(N*N, 0.0);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            y_true[i * N + j] = TRUE_SOL(i * h, j * h);
        }
    }

    double norm_res = DifferentNorm(N*N, h, y_true, y);
    printf("Norm = %f \n", norm_res);
    std::cout << "std::scientific: " << std::scientific << norm_res << '\n';


    std::vector<double> NT(N * N, 0.0);
    std::cout << "Different " << std::scientific /*<< DifferentNorm(N*N, h, y_true, NT)*/<< DifferentNorm(N*N, h, y, NT) << std::endl;


}


int main() {



    test();
    std::cout << "Complete!" << std::endl;
    return 0;
}
