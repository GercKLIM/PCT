/* ------------------------------------ */
/* ### ТЕСТИРОВАНИЕ АЛГОРИТМА C OMP ### */
/* ------------------------------------ */

#include "../INCLUDE/Helmholtz_Solver.h"
#include <iostream>



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

//void speadup_test() {
//
//    /* Числовая константа Пи */
//    const double PI = 3.14159265358979;
//
//    /* Точность алгоритмов */
//    const double EPS = 1e-7;
//
//    /* Ограничение кол-ва итераций */
//    const int MAX_ITERATION = 10000;
//
//    /* Коэффициент k */
//    double k = 20;
//
//    /* Кол-во узлов в одном направлении */
//    int N = 100;
//
//    /* Правая часть*/
//    std::function<double(double, double)> f = ([&](double x, double y){
//        return (2 * sin(PI * y) + k * k * (1 - x) * x * sin(PI * y)
//                + PI * PI * (1 - x) * x * sin(PI * y));
//    });
//
//    /* Точное решение */
//    std::function<double(double, double)> TRUE_SOL = ([&](double x, double y){
//        return ((1 - x) * x * sin(PI * y));
//    });
//
//
//    /* Численное решение задачи 1) МЕТОД ЯКОБИ */
//    std::vector<double> y(N * N, 0.0);
//    std::vector<double> y_copy(y);
//
//    std::ofstream file1, file2;
//    int MAX_TREADS = omp_get_max_threads();
//
//
//    /* Численное решение задачи 1) МЕТОД ЯКОБИ */
//    std::cout << "<----------------------------------->" << std::endl;
//    std::cout << " ### METHOD JACOBI ### "   << std::endl<< std::endl;
//
//    file1.open(("output/output_method_1.txt"));
//    omp_set_num_threads(1);
//    y = y_copy;
//    MethodResultInfo MJ = Method_Jacobi(y, f, k, N, EPS, MAX_ITERATION);
//    std::cout << "Threads = " << 1 << ", Iter = " << MJ.iterations << ", Time   = " << MJ.time << std::endl;
//
//    file1 << std::to_string(1) << " " << std::to_string(MJ.time) << std::endl;
//    for (int i = 2; i <= MAX_TREADS; ++i) {
//        omp_set_num_threads(i);
//        y = y_copy;
//        MJ = Method_Jacobi(y, f, k, N, EPS, MAX_ITERATION);
//        file1 << std::to_string(i) << " " << std::to_string(MJ.time) << std::endl;
//        std::cout << "Threads = " << i << ", Iter = " << MJ.iterations << ", Time   = " << MJ.time << std::endl;
//    }
//    file1.close();
//
//
//
//
//
//    std::cout << "<----------------------------------->" << std::endl;
//    std::cout << " ### METHOD ZEIDEL ### "   << std::endl<< std::endl;
//    omp_set_num_threads(1);
//    y = y_copy;
//    MethodResultInfo MZ = Method_Zeidel(y, f, k, N, EPS, MAX_ITERATION);
//    file2 << std::to_string(1) << " " << std::to_string(MZ.time) << std::endl;
//    std::cout << "Threads = " << 1 << ", Iter = " << MZ.iterations << ", Time   = " << MZ.time << std::endl;
//
//    for (int i = 2; i <= MAX_TREADS; ++i) {
//        omp_set_num_threads(i);
//        y = y_copy;
//        MZ = Method_Zeidel(y, f, k, N, EPS, MAX_ITERATION);
//        file2 << std::to_string(i) << " " << std::to_string(MZ.time) << std::endl;
//        std::cout << "Threads = " << i << ", Iter = " << MZ.iterations << ", Time   = " << MZ.time << std::endl;
//    }
//    file2.close();
//}
