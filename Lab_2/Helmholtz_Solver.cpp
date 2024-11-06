/* --------------------------------------------------------- */
/* ### РЕАЛИЗАЦИЯ ФУНКЦИЙ РЕШЕНИЯ УРАВНЕНИЯ ГЕЛЬМГОЛЬЦА  ### */
/* --------------------------------------------------------- */

#include "Helmholtz_Solver.h"


/* Функция нормы разности векторов */
double DifferentNorm(int size, std::vector<double> A, std::vector<double> B) {
    double sum = 0;
    double tmp;

    #pragma omp parallel for default(none) shared(size,A,B) private(tmp) reduction(+:sum)
    for (int i = 0; i < size; ++i){
        tmp = A[i] - B[i];
        sum += tmp * tmp;
    }
    return sqrt(sum);
}

//  метод Якоби
//struct MethodResultInfo {
//    int iterations = 0;
//    std::vector<double> Y;
//    std::vector<double> Yp;
//
//};

/** Метод решения двумерного уравнения Гельмгольца методом Якоби
 * @param y - массив решения
 * @param f - функция правой части
 * @param k - коэффициент в уравнении
 * @param N - число разбиений
 * @param max_num_iterations - Макс. кол-во итераций
 */
void Method_Jacobi(std::vector<double>& y, std::function<double(double, double)>&f, const double& k, const int& N,
                   const double& eps, const int& max_num_iterations) {

    int true_iterations = max_num_iterations; // Число итераций
    double h = 1. / (N - 1);  // Шаг

    double h_sqr = h * h;
    double mult = 1. / (4 + k * k * h_sqr);

    std::vector<double> yp(y);

    for (int iterations = 1; iterations <= max_num_iterations; ++iterations) {
        //std::swap(y, yp);
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

        if (DifferentNorm(N, y, yp) < eps) {
            true_iterations = iterations;
            break;
        }
    }
}


