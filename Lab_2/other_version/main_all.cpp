/* ### ЛАБОРАТОРНАЯ РАБОТА №2 ###
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

#include <iostream>
#include <vector> // Для доступа к типу std::vector
#include "omp.h"  // Для распараллеливания вычислений
#include <fstream>// Для работы с файлами
#include <cmath>  // Для математических функций
#include <functional> // Для передачи функторов
#include <iomanip> // Для setprecision
#include <string>
#include <cassert>
#include <sstream>
#include <utility>
#include <format>

/* --------------------------------------------------------- */
/* ### РЕАЛИЗАЦИЯ ФУНКЦИЙ РЕШЕНИЯ УРАВНЕНИЯ ГЕЛЬМГОЛЬЦА  ### */
/* --------------------------------------------------------- */


template<typename T>
struct scientificNumberType
{
    explicit scientificNumberType(T number, int decimalPlaces) : number(number), decimalPlaces(decimalPlaces) {}

    T number;
    int decimalPlaces;
};

template<typename T>
scientificNumberType<T> scientificNumber(T t, int decimalPlaces)
{
    return scientificNumberType<T>(t, decimalPlaces);
}

template<typename T>
std::ostream& operator<<(std::ostream& os, const scientificNumberType<T>& n)
{
    double numberDouble = n.number;

    int eToThe = 0;
    for(; numberDouble > 9; ++eToThe)
    {
        numberDouble /= 10;
    }

    // memorize old state
    std::ios oldState(nullptr);
    oldState.copyfmt(os);

    os << std::fixed << std::setprecision(n.decimalPlaces) << numberDouble << "e" << eToThe;

    // restore state
    os.copyfmt(oldState);

    return os;
}


/* Структура для определения выходной информации о работе метода */
struct MethodResultInfo {
    int iterations = 0;     // Итоговое кол-во итераций
    double time = 0;        // Время работы алгоритма
    double norm_iter = 100; // Норма приближений решений на последней итерации
    std::vector<double> Y;  // Итоговое решение
    std::vector<double> Yp; // Предыдущее итоговому приближенное решение
};


/** Функция Нормы разности векторов
 * (NOD - Norm Of Difference)
 * @param size - Длина вектора
 * @param h    - Шаг по узлам
 * @param A    - Первый вектор
 * @param B    - Второй вектор
 * @return     Норму разности векторов
 */
double NOD(const int& size, const double& h, const std::vector<double>& A, const std::vector<double>& B) {
    double sum = 0.0;
    double tmp = 0.;

#pragma omp parallel for default(none) shared(size,A,B) private(tmp) reduction(+:sum)
    for (int i = 0; i < size; ++i){
        tmp = A[i] - B[i];
        sum += tmp * tmp;
    }
    return sqrt(sum * h);
}


/** Метод Якоби
 *  решения двумерного уравнения Гельмгольца
 * @param y - массив решения
 * @param f - функция правой части
 * @param k - коэффициент в уравнении
 * @param N - число разбиений
 * @param max_num_iterations - Макс. кол-во итераций
 * @return  структуру с информацией о работе метода
 */
MethodResultInfo Method_Jacobi(std::vector<double>& y, std::function<double(double, double)>&f, const double& k,
                               const int& N, const double& eps, const int& max_num_iterations) {

    int true_iterations = max_num_iterations; // Число итераций
    double h = 1. / (N - 1);                  // Шаг
    std::vector<double> yp(y);                // Предыдущее приближение решения

    /* Инициализация решения правой частью */
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            y[i * N + j] = f(i * h, j * h);
        }
    }

    double h_sqr = h * h;
    double mult = 1. / (4 + k * k * h_sqr);

    /* Заполнение Граничных Условий (ГУ) */
    double u0 = 0.; // Значение на краях
    for (int i = 0; i < N; ++i) {
        y[i] = u0;
    }

    for (int i = 0; i < N * N; i += N) {
        y[i] = u0;
    }

    for (int i = N * (N -1); i < N * N; ++i) {
        y[i] = u0;
    }

    for (int i = N -1; i < N * N; i += N) {
        y[i] = u0;
    }

    /* Итерационный процесс */
    double time_start = omp_get_wtime();
    for (int iterations = 1; iterations <= max_num_iterations; ++iterations) {

        y.swap(yp);

#pragma omp parallel for default(shared)
        for (int i = 1; i < N - 1; ++i) {
            for (int j = 1; j < N - 1; ++j) {
                /* Результирующее соотношение */
                y[i * N + j] = mult * (yp[(i + 1) * N + j] + yp[(i - 1) * N + j] +
                                       yp[i * N + (j + 1)] + yp[i * N + (j - 1)] +
                                       h_sqr * f(h * i, h * j));
            }
        }

        if (NOD(N * N, h, y, yp) < eps) {
            true_iterations = iterations;
            break;
        }
    }
    double time_end = omp_get_wtime();

    /* Упаковка результата работы алгоритма */
    MethodResultInfo info;
    info.iterations = true_iterations;
    info.norm_iter = NOD(N * N, h, y, yp);
    info.time = time_end - time_start;
    info.Y =  y;
    info.Yp = yp;
    return info;
}


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
                               const int& N, const double& eps, const int& max_num_iterations) {

    int true_iterations = max_num_iterations; // Число итераций
    double h = 1. / (N - 1);                  // Шаг
    std::vector<double> yp(y);                // Предыдущее приближение решения

    /* Инициализация решения правой частью */
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            y[i * N + j] = f(i * h, j * h);
        }
    }

    double h_sqr = h * h;
    double mult = 1. / (4 + k * k * h_sqr);

    /* Заполнение Граничных Условий (ГУ) */
    double u0 = 0.; // Значение на краях
    for (int i = 0; i < N; ++i) {
        y[i] = u0;
    }

    for (int i = 0; i < N * N; i += N) {
        y[i] = u0;
    }

    for (int i = N * (N -1); i < N * N; ++i) {
        y[i] = u0;
    }

    for (int i = N -1; i < N * N; i += N) {
        y[i] = u0;
    }

    double time_start = omp_get_wtime();
    for (int iterations = 1; iterations <= max_num_iterations; ++iterations) {

        std::swap(yp, y);

        // По красным узлам
#pragma omp parallel for default(shared)
        for (int i = 1; i < N - 1; ++i) {
            //#pragma omp parallel for default(none) private(i) shared(mult, h_sqr, h, f, yp, y, N)
            for (int j = 1 + (i + 1) % 2; j < N - 1; j += 2) {
                y[i * N + j] = (yp[(i + 1) * N + j]
                                + yp[(i - 1) * N + j]
                                + yp[i * N + (j + 1)]
                                + yp[i * N + (j - 1)]
                                + h_sqr * f(i * h, j * h)) * mult;
            }
        }

        // По чёрным узлам
#pragma omp parallel for default(shared)
        for (int i = 1; i < N - 1; ++i) {
            for (int j = 1 + i % 2; j < N - 1; j += 2) {
                y[i * N + j] = (y[(i + 1) * N + j]
                                + y[(i - 1) * N + j]
                                + y[i * N + (j + 1)]
                                + y[i * N + (j - 1)]
                                + h_sqr * f(i * h, j * h)) * mult;
            }
        }

        if (NOD(N*N, h, yp, y) < eps) {
            true_iterations = iterations;
            break;
        }
    }
    double time_end = omp_get_wtime();

    /* Упаковка результата работы алгоритма */
    MethodResultInfo info;
    info.iterations = true_iterations;
    info.norm_iter = NOD(N * N, h, y, yp);
    info.time = time_end - time_start;
    info.Y =  y;
    info.Yp = yp;
    return info;
}


/** Функция для проверки корректности решения уравнения
 * @param N - Кол-во узлов в решении
 * @param y - Вектор численного решения
 * @param True_sol_func - Функция точного решения
 * @return Норму разности точного и численного решения
 */
double test_sol(const int& N, const std::vector<double>& y, std::function<double(double, double)>& True_sol_func) {

    double h = 1. / (N - 1);
    std::vector<double> y_true(N*N, 0.0);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            y_true[i * N + j] = True_sol_func(i * h, j * h);
        }
    }

    double norm_res = NOD(N*N, h, y, y_true);
    return norm_res;
}


/* -------------------------------- */
/* ### СЧИТЫВАНИЕ ПАРАМЕТРОВ ИЗ ФАЙЛА  ### */
/* -------------------------------- */

void setValsFromFile(std::string fileName, double& k, int& N, double& eps, int& nThreads, int& maxIts){
    std::ifstream file(fileName);
    assert(file.is_open());
    std::string tmp_line;
    std::getline(file, tmp_line);
    std::istringstream ss(tmp_line);
    ss >> k >> N >> eps >> nThreads >> maxIts;
    printf("Values of the test: k = %f, N = %d, eps = %f, nThreads = %d, maxIts = %d \n",
           k, N, eps, nThreads, maxIts);
}


/* -------------------------------- */
/* ### ТЕСТИРОВАНИЕ АЛГОРИТМА  ### */
/* -------------------------------- */



void test(std::string filename) {

    /* Числовая константа Пи */
    const double PI = 3.14159265358979;

    /* Кол-во потоков */
    int NUM_THREADS = 1;


    /* Точность алгоритмов */
    double EPS = 1e-7;

    /* Ограничение кол-ва итераций */
    int MAX_ITERATION = 10000;

    /* Коэффициент k */
    double k = 20;

    /* Кол-во узлов в одном направлении */
    int N = 100;

    setValsFromFile(filename, k, N, EPS, NUM_THREADS, MAX_ITERATION);

    //omp_set_dynamic(0);
    omp_set_num_threads(NUM_THREADS);

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
    std::cout<<"CPU: "<<NUM_THREADS<<" threads"<< std::endl;
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
    std::cout <<"CPU: "<<NUM_THREADS<<" threads"<< std::endl;
    std::cout << "<----------------------------------->"  << std::endl;

}

void speadup_test() {

    /* Числовая константа Пи */
    const double PI = 3.14159265358979;

    /* Точность алгоритмов */
    /*const*/ double EPS = 1e-7;

    /* Ограничение кол-ва итераций */
    /*const*/ int MAX_ITERATION = 10000;

    /* Коэффициент k */
    /*const*/ double k = 20;

    /* Кол-во узлов в одном направлении */
    int N = 100;

    int NUM_THREADS = 1;


    setValsFromFile("input.txt", k, N, EPS, NUM_THREADS, MAX_ITERATION);

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

    std::ofstream file1, file2;
    int MAX_TREADS = omp_get_max_threads();

    double time_threads1;

    /* Численное решение задачи 1) МЕТОД ЯКОБИ */
    std::cout << "<----------------------------------->" << std::endl;
    std::cout << " ### METHOD JACOBI ### "   << std::endl<< std::endl;



    //std::cout << std::scientific;
    //std::cout.precision(3);//точность 4 цифры



    file1.open(("output_method_1.txt"));
    omp_set_num_threads(1);
    y = y_copy;
    MethodResultInfo MJ = Method_Jacobi(y, f, k, N, EPS, MAX_ITERATION);
    std::cout << "Threads = " << 1 << ", Iter = " << MJ.iterations << ", Norm = "  << std::scientific << test_sol(N, y, TRUE_SOL) << ", Time   = " << MJ.time << std::endl;
    file1 << "Threads = " << 1 << ", Iter = " << MJ.iterations << ", Norm = " << std::scientific << test_sol(N, y, TRUE_SOL) << ", Time   = " << MJ.time << std::endl;

    time_threads1 = MJ.time;

    for (int i = 2; i <= MAX_TREADS; i+=2) {
        omp_set_num_threads(i);
        y = y_copy;
        MJ = Method_Jacobi(y, f, k, N, EPS, MAX_ITERATION);
        //file1 << std::to_string(i) << " " << std::to_string(MJ.time) << std::endl;

        //std::cout << "Threads = " << i << ", Iter = " << MJ.iterations << ", Time   = " << MJ.time << std::endl;
        std::cout << "Threads = " << i << ", Iter = " << MJ.iterations << ", Norm = " << std::scientific << test_sol(N, y, TRUE_SOL) <<
                  ", Time   = " << MJ.time << ", speadup = " << time_threads1 / MJ.time << std::endl;

        //file1 << "Threads = " + std::to_string(i) << ", Iter = " + std::to_string(MZ.iterations) << ", Time = " + std::to_string(MJ.time) << std::endl;
        file1 << "Threads = " << i << ", Iter = " << MJ.iterations << ", Norm = " << std::scientific << test_sol(N, y, TRUE_SOL) <<
              ", Time   = " << MJ.time << ", speadup = " << time_threads1 / MJ.time << std::endl;

    }
    file1.close();





    std::cout << "<----------------------------------->" << std::endl;
    std::cout << " ### METHOD ZEIDEL ### "   << std::endl<< std::endl;
    file2.open(("output_method_2.txt"));
    omp_set_num_threads(1);
    y = y_copy;
    MethodResultInfo MZ = Method_Zeidel(y, f, k, N, EPS, MAX_ITERATION);

    file2 << "Threads = " << 1 << ", Iter = " << MZ.iterations << ", Norm = " << std::scientific << test_sol(N, y, TRUE_SOL) <<
          ", Time   = " << MZ.time << std::endl;

    std::cout << "Threads = " << 1 << ", Iter = " << MZ.iterations << ", Norm = " << std::scientific << test_sol(N, y, TRUE_SOL) << ", Time   = " << MZ.time << std::endl;

    time_threads1 = MZ.time;
    for (int i = 2; i <= MAX_TREADS; i += 2) {
        omp_set_num_threads(i);
        y = y_copy;
        MZ = Method_Zeidel(y, f, k, N, EPS, MAX_ITERATION);

        file2 << std::to_string(i) << " " << std::to_string(MZ.time) << std::endl;
        std::cout << "Threads = " << i << ", Iter = " << MZ.iterations << ", Norm = " << std::scientific << test_sol(N, y, TRUE_SOL) <<
                  ", Time   = " << MZ.time << ", speadup = " << time_threads1 / MZ.time<< std::endl;


        file2 << "Threads = " << i << ", Iter = " << MZ.iterations << ", Norm = " << std::scientific << test_sol(N, y, TRUE_SOL) <<
              ", Time   = " << MZ.time << std::endl;
    }
    file2.close();
}


int main() {
    std::cout << std::scientific;
    speadup_test();
    //test("input1.txt");
    //test("input4.txt");

    std::cout << "Complete!" << std::endl;
    return 0;
}