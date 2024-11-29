/* ### ЛАБОРАТОРНАЯ РАБОТА №3 ###
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
 *  С использованием MPI
 * */

#include "include/Helmholtz_Solver.h"
#include <iostream>


void test_cout(int argc, char** argv){

    /* Путь к файлу с тестовыми параметрами */
    const std::string TEST_FILENAME = "../INPUT/input_parametres_1.json";


    /* Начальные параметры программы */
    int ID = 0;                  // ID Процесса
    int NP = 1;                  // Общее число процессов
    int N = 1;                   // Кол-во разбиений
    double K = 1;                // Коэф. уравнения K
    int MAX_ITERATIONS = 1;      // Ограничение кол-ва итераций
    double EPS = 1;              // Допустимая погрешность
    std::string TEST_NAME = " "; // Назавние Теста

    /* Получаем параметры из внешнего файла */
    input_parametres(TEST_FILENAME, N, K, MAX_ITERATIONS, EPS, TEST_NAME);
//    if ((input_parametres(TEST_FILENAME, N, K, MAX_ITERATIONS, EPS, TEST_NAME)) and (ID == 0)) {
//        std::cout << Logs::LOG_SUCCESS << "Parametres import process was SUCCESSFULLY." << std::endl;
//    }

    /* Запускаем MPI */
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &NP); // Получаем общее число процессов
    MPI_Comm_rank(MPI_COMM_WORLD, &ID); // Получаем ID текущего процесса


    /* Определение задачи */

    /* Правая часть */
    std::function<double(double, double)> f = ([&](double x, double y){
        return (2 * sin(M_PI * y) + K * K * (1 - x) * x * sin(M_PI * y)
                + M_PI * M_PI * (1 - x) * x * sin(M_PI * y));
    });

    /* Точное решение */
    std::function<double(double, double)> TRUE_SOL = ([&](double x, double y){
        return ((1 - x) * x * sin(M_PI * y));
    });



    /* РЕШЕНИЕ УРАВНЕНИЯ */



    /* Решение методом Якоби Send Recv */
    MethodResultInfo MJ1;
    Method_Jacobi_P2P(MJ1, f, K, N, EPS, MAX_ITERATIONS);
    if (ID ==0 ) {
        MJ1.norm_sol = test_sol(N, MJ1.Y, TRUE_SOL);
        print_MethodResultInfo(MJ1);
    }


    /* Решение методом Якоби SendRecv */
    MethodResultInfo MJ2;
    Method_Jacobi_SIMULT(MJ2, f, K, N, EPS, MAX_ITERATIONS);
    if (ID ==0 ) {
        MJ2.norm_sol = test_sol(N, MJ2.Y, TRUE_SOL);
        print_MethodResultInfo(MJ2);
    }


    /* Решение методом Якоби ISend IRecv */
    MethodResultInfo MJ3;
    Method_Jacobi_NOBLOCK(MJ3, f, K, N, EPS, MAX_ITERATIONS);
    if (ID ==0 ) {
        MJ3.norm_sol = test_sol(N, MJ3.Y, TRUE_SOL);
        print_MethodResultInfo(MJ3);
    }


    /* Решение методом Зейделя Send Recv */
    MethodResultInfo MZ1;
    Method_Zeidel_P2P(MZ1, f, K, N, EPS, MAX_ITERATIONS);
    if (ID ==0 ) {
        MZ1.norm_sol = test_sol(N, MZ1.Y, TRUE_SOL);
        print_MethodResultInfo(MZ1);
    }


    /* Решение методом Зейделя SendRecv */
    MethodResultInfo MZ2;
    Method_Zeidel_SIMULT(MZ2, f, K, N, EPS, MAX_ITERATIONS);
    if (ID ==0 ) {
        MZ2.norm_sol = test_sol(N, MZ2.Y, TRUE_SOL);
        print_MethodResultInfo(MZ2);
    }


    /* Решение методом Зейделя ISend IRecv */
    MethodResultInfo MZ3;
    Method_Zeidel_NOBLOCK(MZ3, f, K, N, EPS, MAX_ITERATIONS);
    if (ID == 0) {
        MZ3.norm_sol = test_sol(N, MZ3.Y, TRUE_SOL);
        print_MethodResultInfo(MZ3);
    }


    /* КОНЕЦ */
    if (ID == 0) {
        std::cout << Logs::LOG_SUCCESS << "Complete!" << std::endl;
    }

    MPI_Finalize();

}

void test_fout(int argc, char** argv){
    /* Путь к файлу с тестовыми параметрами */
    const std::string TEST_FILENAME = "/nethome/student/FS21/FS2-x1/Klim_and_Shaman/LAB_3/INPUT/input_parametres_1.json";


    /* Начальные параметры программы */
    int ID = 0;                  // ID Процесса
    int NP = 1;                  // Общее число процессов
    int N = 1;                   // Кол-во разбиений
    double K = 1;                // Коэф. уравнения K
    int MAX_ITERATIONS = 1;      // Ограничение кол-ва итераций
    double EPS = 1;              // Допустимая погрешность
    std::string TEST_NAME = " "; // Назавние Теста

    /* Получаем параметры из внешнего файла */
    input_parametres(TEST_FILENAME, N, K, MAX_ITERATIONS, EPS, TEST_NAME);

    /* Запускаем MPI */
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &NP); // Получаем общее число процессов
    MPI_Comm_rank(MPI_COMM_WORLD, &ID); // Получаем ID текущего процесса



    /* Файл для вывода */
    const std::string OUT_FILENAME = "/nethome/student/FS21/FS2-x1/Klim_and_Shaman/LAB_3/OUTPUT/output_N=" + std::to_string(N) + "_K=" + std::to_string((int)K) + "_NP=" + std::to_string(NP) + ".txt";

    std::ofstream Fout(OUT_FILENAME);

    if (!Fout) {
        std::cout << "File " <<  OUT_FILENAME << " is NOT open. " << std::endl;
    }




    /* Определение задачи */

    /* Правая часть */
    std::function<double(double, double)> f = ([&](double x, double y){
        return (2 * sin(M_PI * y) + K * K * (1 - x) * x * sin(M_PI * y)
                + M_PI * M_PI * (1 - x) * x * sin(M_PI * y));
    });

    /* Точное решение */
    std::function<double(double, double)> TRUE_SOL = ([&](double x, double y){
        return ((1 - x) * x * sin(M_PI * y));
    });



    /* РЕШЕНИЕ УРАВНЕНИЯ */



    /* Решение методом Якоби Send Recv */
    MethodResultInfo MJ1;
    Method_Jacobi_P2P(MJ1, f, K, N, EPS, MAX_ITERATIONS);
    if (ID ==0 ) {
        MJ1.norm_sol = test_sol(N, MJ1.Y, TRUE_SOL);
        print_MethodResultInfoFile(MJ1, Fout);
    }


    /* Решение методом Якоби SendRecv */
    MethodResultInfo MJ2;
    Method_Jacobi_SIMULT(MJ2, f, K, N, EPS, MAX_ITERATIONS);
    if (ID ==0 ) {
        MJ2.norm_sol = test_sol(N, MJ2.Y, TRUE_SOL);
        print_MethodResultInfoFile(MJ2, Fout);
    }


    /* Решение методом Якоби ISend IRecv */
    MethodResultInfo MJ3;
    Method_Jacobi_NOBLOCK(MJ3, f, K, N, EPS, MAX_ITERATIONS);
    if (ID ==0 ) {
        MJ3.norm_sol = test_sol(N, MJ3.Y, TRUE_SOL);
        print_MethodResultInfoFile(MJ3, Fout);
    }


    /* Решение методом Зейделя Send Recv */
    MethodResultInfo MZ1;
    Method_Zeidel_P2P(MZ1, f, K, N, EPS, MAX_ITERATIONS);
    if (ID ==0 ) {
        MZ1.norm_sol = test_sol(N, MZ1.Y, TRUE_SOL);
        print_MethodResultInfoFile(MZ1, Fout);
    }


    /* Решение методом Зейделя SendRecv */
    MethodResultInfo MZ2;
    Method_Zeidel_SIMULT(MZ2, f, K, N, EPS, MAX_ITERATIONS);
    if (ID ==0 ) {
        MZ2.norm_sol = test_sol(N, MZ2.Y, TRUE_SOL);
        print_MethodResultInfoFile(MZ2, Fout);
    }


    /* Решение методом Зейделя ISend IRecv */
    MethodResultInfo MZ3;
    Method_Zeidel_NOBLOCK(MZ3, f, K, N, EPS, MAX_ITERATIONS);
    if (ID == 0) {
        MZ3.norm_sol = test_sol(N, MZ3.Y, TRUE_SOL);
        print_MethodResultInfoFile(MZ3, Fout);
    }


    /* КОНЕЦ */
    if (ID == 0) {
        std::cout << Logs::LOG_SUCCESS << "Complete!" << std::endl;
    }

    Fout.close();
    MPI_Finalize();
}



int main(int argc, char** argv) {

    test_cout(argc, argv);
    return EXIT_SUCCESS;
}
