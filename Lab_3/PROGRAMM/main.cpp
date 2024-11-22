///* ### ЛАБОРАТОРНАЯ РАБОТА №3 ###
// *
// *  ЗАДАНИЕ:
// *
// *  Разработать программу для решения двумерного уравнения Гельмгольца
// *
// *  -D[u[x, y], x, x] - D[u[x, y], y, y] + k^2 u = f(x, y)
// *
// *  в квадратной области (x, y) in [0, 1]x[0, 1]
// *
// *  c граничными условиями:
// *  u(x, 0) = u(x, 1) = u(0, y) = u(1, y) = 0
// *
// *  и правой частью
// *  f(x, y) = 2 sin(pi y) + k^2 (1 - x) x sin(pi y) + pi^2 (1 - x) x sin(pi y)
// *
// *  Для численного решения уравнения на прямоугольной равномерной сетке использовать
// *  конечно-разностную схему «крест» второго порядка аппроксимации по обеим независимым переменным.
// *  Полученную систему линейных алгебраических уравнений решить итерационным
// *  методом Якоби и Зейделя (точнее, «красно-черных» итераций).
// *  Обеспечить вывод на экран количества итераций до сходимости и погрешности.
// *
// *  Точное решение задачи: u(x, y) = (1 - x) x sin(pi y)
// *
// *  С использованием MPI
// * */
//
//#include "include/Helmholtz_Solver.h"
//#include <iostream>
//
///* Числовая константа Пи */
//const double PI = 3.14159265358979;
//
//
////void testMPI(int argc, char** argv){
////
////    int k = 0;
////    int ID = 0;
////    MPI_Init(&argc, &argv);
////
////    k = 5;
////    //MPI_Comm_size(MPI_COMM_WORLD, &NP); // Получаем кол-во исполняемых программ
////    MPI_Comm_rank(MPI_COMM_WORLD, &ID); // Получаем номер исполняемой программы
////    std::cout << ID << std::endl;
////
////    MPI_Finalize();
////
////    std::cout << k << std::endl;
////}
//
//void testMPI(int argc, char** argv) {
//    int ID = 0;
//    MPI_Init(&argc, &argv);
//    MPI_Comm_rank(MPI_COMM_WORLD, &ID);
//
//    // Каждый процесс выведет свой ранг
//    std::cout << "Hello from process " << ID << std::endl;
//
//    MPI_Finalize();
//
//    // Чтобы убедиться, что программа не зависает
//    std::cout << "Process " << ID << " finalized." << std::endl;
//}
//
//
//
//
//int main(int argc, char** argv) {
//
////    const std::string TEST_FILENAME = "../../INPUT/input_parametres_1.json";
////
////    /* Начальные параметры программы */
////    int NP = 1;                  // Кол-во потоков
////    int N = 1;                   // Кол-во разбиений
////    double K = 1;                // Коэф. уравнения K
////    int MAX_ITERATIONS = 1;      // Ограничение кол-ва итераций
////    double EPS = 1;              // Допустимая погрешность
////    std::string TEST_NAME = " "; // Назавние Теста
////
////    /* Получаем параметры из внешнего файла */
////    input_parametres(TEST_FILENAME, NP, N, K, MAX_ITERATIONS, EPS, TEST_NAME);
////
////
////    /* Определение задачи */
////
////    /* Правая часть */
////    std::function<double(double, double)> f = ([&](double x, double y){
////        return (2 * sin(PI * y) + K * K * (1 - x) * x * sin(PI * y)
////                + PI * PI * (1 - x) * x * sin(PI * y));
////    });
////
////    /* Точное решение */
////    std::function<double(double, double)> TRUE_SOL = ([&](double x, double y){
////        return ((1 - x) * x * sin(PI * y));
////    });
//
//
//    testMPI(argc,argv);
//    /* Решение методом Якоби Send Recv */
////    std::vector<double> y(N * N, 0.0);
////    MethodResultInfo MJ = Method_Jacobi_P2P(argc, argv, f, K, N, EPS, MAX_ITERATIONS);
////
////    std::cout << "<----------------------------------->" << std::endl;
////    std::cout << " ### METHOD JACOBI ### "   << std::endl<< std::endl;
////    std::cout << "Norm   = " << test_sol(N, y, TRUE_SOL) << std::endl;
////    std::cout << "Iter   = " << MJ.iterations            << std::endl;
////    std::cout << "Time   = " << MJ.time                  << std::endl;
////    std::cout << "|Y-Yp| = " << MJ.norm_iter             << std::endl;
////    //std::cout<<"CPU: "<<omp_get_num_threads()<<" threads"<< std::endl;
////    std::cout << "<----------------------------------->" << std::endl;
//
//    /* Решение методом Якоби SendRecv */
//
//    /* Решение методом Якоби ISend IRecv */
//
//    /* Решение методом Зейделя Send Recv */
//
//    /* Решение методом Зейделя SendRecv */
//
//    /* Решение методом Зейделя ISend IRecv */
//
//    std::cout << Logs::LOG_SUCCESS << "Complete!" << std::endl;
//    return EXIT_SUCCESS;
//}

#include <mpi.h>
#include <iostream>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::cout << "Hello from process " << rank << " of " << size << std::endl;

    MPI_Finalize();
    return 0;
}