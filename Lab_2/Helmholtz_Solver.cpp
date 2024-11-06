/* --------------------------------------------------------- */
/* ### РЕАЛИЗАЦИЯ ФУНКЦИЙ РЕШЕНИЯ УРАВНЕНИЯ ГЕЛЬМГОЛЬЦА  ### */
/* --------------------------------------------------------- */

#include "Helmholtz_Solver.h"




double DifferentNorm(const int& size, const double& h, const std::vector<double>& A, const std::vector<double>& B) {
    double sum = 0.0;
    double tmp = 0.;

    #pragma omp parallel for default(none) shared(size,A,B) private(tmp) reduction(+:sum)
    for (int i = 0; i < size; ++i){
        tmp = A[i] - B[i];
        sum += tmp * tmp;
    }
    std::cout << "std::scientific: " << std::scientific << sqrt(sum * h) << '\n';
    return sqrt(sum * h);
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

    // ГУ
    double u0 = 0.;
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

        if (DifferentNorm(N * N, h, y, yp) < eps) {
            true_iterations = iterations;
            break;
        }
    }
    std::cout << "Iter: " << true_iterations << std::endl;
}


