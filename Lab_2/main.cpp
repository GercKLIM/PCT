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



/* -------------------------------- */
/* ### ТЕСТИРОВАНИЕ АЛГОРИТМА  ### */
/* -------------------------------- */



void test() {

    /* Числовая константа Пи */
    const double PI = 3.14159265358979;

    /* Кол-во потоков */
    int NUM_THREADS = 1;
    omp_set_num_threads(NUM_THREADS);

    /* Точность алгоритмов */
    const double EPS = 1e-7;

    /* Ограничение кол-ва итераций */
    const int MAX_ITERATION = 10000;

    /* Коэффициент k */
    double k = 20;

    /* Кол-во узлов в одном направлении */
    int N = 100;

    /* Правая часть*/
    std::function<double(double, double)> f = ([&](double x, double y){
        return (2 * sin(PI * y) + k * k * (1 - x) * x * sin(PI * y)
                + PI * PI * (1 - x) * x * sin(PI * y));
    });

    /* Точное решение */
    std::function<double(double, double)> TRUE_SOL = ([&](double x, double y){
        return ((1 - x) * x * sin(PI * y));
    });


    /* Численное решение задачи 1) МЕТОД ЯКОБИ */
    std::vector<double> y(N * N, 0.0);
    MethodResultInfo MJ = Method_Jacobi(y, f, k, N, EPS, MAX_ITERATION);

    std::cout << "<----------------------------------->" << std::endl;
    std::cout << " ### METHOD JACOBI ### "   << std::endl<< std::endl;
    std::cout << "Norm   = " << test_sol(N, y, TRUE_SOL) << std::endl;
    std::cout << "Iter   = " << MJ.iterations            << std::endl;
    std::cout << "Time   = " << MJ.time                  << std::endl;
    std::cout << "|Y-Yp| = " << MJ.norm_iter             << std::endl;
    std::cout<<"CPU: "<<omp_get_num_threads()<<" threads"<< std::endl;
    std::cout << "<----------------------------------->" << std::endl;

    /* Численное решение задачи 2) МЕТОД ЗЕЙДЕЛЯ */
    std::vector<double> y2(N * N, 0.0);
    MethodResultInfo MZ = Method_Zeidel(y2, f, k, N, EPS, MAX_ITERATION);

    std::cout << "<----------------------------------->"  << std::endl;
    std::cout << " ### METHOD ZEIDEL ### "   << std::endl << std::endl;
    std::cout << "Norm   = " << test_sol(N, y2, TRUE_SOL) << std::endl;
    std::cout << "Iter   = " << MZ.iterations             << std::endl;
    std::cout << "Time   = " << MZ.time                   << std::endl;
    std::cout << "|Y-Yp| = " << MZ.norm_iter              << std::endl;
    std::cout <<"CPU: "<<omp_get_num_threads()<<" threads"<< std::endl;
    std::cout << "<----------------------------------->"  << std::endl;

}

void speadup_test() {

    /* Числовая константа Пи */
    const double PI = 3.14159265358979;

    /* Кол-во потоков */
    int NUM_THREADS = 1;
    omp_set_num_threads(NUM_THREADS);

    /* Точность алгоритмов */
    const double EPS = 1e-7;

    /* Ограничение кол-ва итераций */
    const int MAX_ITERATION = 10000;

    /* Коэффициент k */
    double k = 20;

    /* Кол-во узлов в одном направлении */
    int N = 100;

    /* Правая часть*/
    std::function<double(double, double)> f = ([&](double x, double y){
        return (2 * sin(PI * y) + k * k * (1 - x) * x * sin(PI * y)
                + PI * PI * (1 - x) * x * sin(PI * y));
    });

    /* Точное решение */
    std::function<double(double, double)> TRUE_SOL = ([&](double x, double y){
        return ((1 - x) * x * sin(PI * y));
    });


    /* Численное решение задачи 1) МЕТОД ЯКОБИ */
    std::vector<double> y(N * N, 0.0);
    std::vector<double> y_copy(y);

    /* Численное решение задачи 1) МЕТОД ЯКОБИ */
    y = y_copy;
    MethodResultInfo MJ = Method_Jacobi(y, f, k, N, EPS, MAX_ITERATION);

    /* Численное решение задачи 2) МЕТОД ЗЕЙДЕЛЯ */
    y = y_copy;
    MethodResultInfo MZ = Method_Jacobi(y, f, k, N, EPS, MAX_ITERATION);


    std::ofstream file1, file2;
    int MAX_TREADS = omp_get_max_threads();


    file1.open(("output_method_1.txt"));
    omp_set_num_threads(1);
    MJ = Method_Jacobi(y, f, k, N, EPS, MAX_ITERATION);
    file1 << std::to_string(1) << " " << std::to_string(MJ.time) << std::endl;
    for (int i = 2; i <= MAX_TREADS; ++i) {
        omp_set_num_threads(i);
        y = y_copy;
        MJ = Method_Jacobi(y, f, k, N, EPS, MAX_ITERATION);

        file1 << std::to_string(i) << " " << std::to_string(MJ.time) << std::endl;
    }
    file1.close();


    file2.open(("output_method_2.txt"));
    omp_set_num_threads(1);
    y = y_copy;
    MZ = Method_Zeidel(y, f, k, N, EPS, MAX_ITERATION);
    file2 << std::to_string(1) << " " << std::to_string(MJ.time) << std::endl;

    for (int i = 2; i <= MAX_TREADS; ++i) {
        omp_set_num_threads(i);
        y = y_copy;
        MZ = Method_Zeidel(y, f, k, N, EPS, MAX_ITERATION);
        file2 << std::to_string(i) << " " << std::to_string(MJ.time) << std::endl;
    }
    file2.close();
}


int main() {



    test();
    std::cout << "Complete!" << std::endl;
    return 0;
}
