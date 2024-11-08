/* --------------------------------------------------------- */
/* ### РЕАЛИЗАЦИЯ ФУНКЦИЙ РЕШЕНИЯ УРАВНЕНИЯ ГЕЛЬМГОЛЬЦА  ### */
/* --------------------------------------------------------- */

#include "Helmholtz_Solver.h"



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


