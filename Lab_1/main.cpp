/* ### ЛАБОРАТОРНАЯ РАБОТА №1 ###
 *
 *  ЗАДАНИЕ: Реализация блочного алгоритма LU-разложения матрицы
 *  Требуется реализовать неблочный (традиционный) алгоритм -
 *  - в приложенной книге алгоритм 2.3 (а также 2.4),
 *  а также его блочный вариант - алгоритм 2.10.
 *
 *  Требования и уточнения:
 * 1) выбор главного элемента производить не нужно;
 * 2) получаемые в ходе разложения элементы матриц L и U записывать
 *    прямо поверх уже обработанных элементов матрицы A.
 *    Новое место в памяти под L и U не выделять;
 * 3) выполнить предварительное копирование матрицы A в отдельную переменную
 *    для последующей проверки правильности полученного разложения A = L*U;
 * 4) в блочном алгоритме выделить память под блок размера b x b,
 *    а также полосы под ним и справа от него (можно объединить блок с вертикальной полосой).
 *    Выделение памяти производить 1 раз, в дальнейшем просто использовать меньший объем.
 *
 *    !Все матрицы хранить в одномерных массивах по строкам;!
 *
 * 5) сравнить время последовательной и параллельной реализаций (с различным числом потоков),
 *    а также блочной и неблочной версий (определить наиболее эффективный размер блока).
 *    Измерять время собственно для выполняемого алгоритма
 *    (без подготовки данных и последующей проверки результата).
 *
 * */


#include <iostream>
#include <vector> // Для доступа к типу std::vector
#include "omp.h"  // Для распараллеливания вычислений
#include <random> // Для генерации случайных чисел
#include <ctime>  // Для инициализации генератора

/* Константа допустимой погрешности */
const double EPS = 1e-7;

/* Функция вывода матрицы */
void print(std::vector<double> matrix, int size = 0){

    for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; ++j) {
                std::cout << matrix[i * size + j] << " ";
            }
            std::cout << std::endl;
        }
    std::cout << std::endl;
}

///* Функция умножения матриц (обычный алгоритм, для матриц различной размерности) */
//std::vector<double> matrix_multiply(const std::vector<double>& A, const std::vector<double>& B, int m, int n, int p) {
//    // Результирующая матрица C размером m x p
//    std::vector<double> C(m * p, 0.0);
//
//    for (int i = 0; i < m; ++i) {
//        for (int j = 0; j < p; ++j) {
//            double sum = 0.0;
//            for (int k = 0; k < n; ++k) {
//                sum += A[i * n + k] * B[k * p + j];
//            }
//            C[i * p + j] = sum;
//        }
//    }
//
//    return C;
//}

/* Функция умножения матриц (обычный алгоритм) */
std::vector<double> matrix_multiply(const std::vector<double>& A, const std::vector<double>& B, int n) {
    // Результирующая матрица C размером m x p
    std::vector<double> C(n * n, 0.0);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double sum = 0.0;
            for (int k = 0; k < n; ++k) {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }

    return C;
}


/* Функция умножения матриц (блочный алгоритм) */
std::vector<double> matrix_multiply_block(const std::vector<double>& A, const std::vector<double>& B, int n, int block_size) {
    std::vector<double> C(n * n, 0.0); // Результирующая матрица

    // Проходим по блокам
    for (int i = 0; i < n; i += block_size) {
        for (int j = 0; j < n; j += block_size) {
            for (int k = 0; k < n; k += block_size) {

                // Перемножаем блоки
                for (int ii = i; ii < std::min(i + block_size, n); ++ii) {
                    for (int jj = j; jj < std::min(j + block_size, n); ++jj) {
                        double sum = 0.0;
                        for (int kk = k; kk < std::min(k + block_size, n); ++kk) {
                            sum += A[ii * n + kk] * B[kk * n + jj];
                        }
                        C[ii * n + jj] = C[ii * n + jj] + sum;
                    }
                }
            }
        }
    }
    return C;
}

/* Функция обратного хода метода Гаусса */
void reverse_Gauss(const std::vector<double>& A, const size_t n, std::vector<double>& b)
{
    for (int k = 0; k < n; ++k)
        for (int i = k + 1; i < n; ++i)
            b[i] -= A[i * n + k] * b[k];
}


/* Функция - Реализация "Традиционного алгоритма"(2.3-2.4) LU-разложения матрицы
 *
 *  A - входная матрица в виде одномерного массива по строкам
 *  A_size - длина этого вектора
 *  n - размерность матрицы
 *  use_omp - вкл/выкл распараллеливания
 *
 * */
void LU_Decomposition(std::vector<double>& A, const int& A_size, const int& n, bool use_omp = false){

    for (int i = 0; i < n-1; i++){

        #pragma omp parallel for if (use_omp)
        for (int j = i + 1; j < n; j++){

            A[j * n + i] /= A[i * n + i];

            for (int k = i + 1; k < n; k++){
                A[j * n + k] -= A[j * n + i] * A[i * n + k];
            }
        }
    }
}


/* Функция - Реализация "Блочного алгоритма"(2.10) LU-разложения матрицы */
void LU_decomposition_block(std::vector<double>& A, const int& A_size, const int& n, const int& block_size, bool use_omp = false) {


}


/* Функция - Создание случайного вектора размера n */
std::vector<double> randvec(int n) {

    // Границы для случайных чисел
    double lower_bound = 0.0;
    double upper_bound = 10.0;

    // Создание вектора
    std::vector<double> random_numbers;

    // Инициализация генератора случайных чисел
    std::mt19937 generator(static_cast<unsigned int>(std::time(0))); // Mersenne Twister
    std::uniform_real_distribution<double> distribution(lower_bound, upper_bound);

    // Заполнение вектора случайными числами
    for (int i = 0; i < n; ++i) {
        random_numbers.push_back(distribution(generator));
    }

    return random_numbers;
}


/* Функция - Тестирование алгоритма на корректность результата */
bool test_result(){

    // Пример
    std::vector<double> A = {2, 3, 1, 4, 7, -1, -2, -3, -4};
    std::vector<double> A_copy(A);
    int n = 3;

    LU_Decomposition(A, n * n, n);

    // Получаем L
    std::vector<double> L(n * n, 0.0); // Инициализируем нулями
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j) {
                L[i * n + j] = 1.0; // Диагональные элементы равны 1
            } else if (i > j) {
                L[i * n + j] = A[i * n + j]; // Элементы ниже диагонали
            }
        }
    }

    // Получаем U
    std::vector<double> U(n * n, 0.0); // Инициализируем нулями
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i <= j) {
                U[i * n + j] = A[i * n + j]; // Элементы на и выше диагонали
            }
        }
    }

    // A_new = L * U;
    std::vector<double> A_new = matrix_multiply(L, U, n);

    // Сравниваем исходную матрицу с восстановленной
    for (int i = 0; i < A.size(); i++) {
        if (abs(A_new[i] - A_copy[i]) > EPS) {
            std::cout << "FALSE" << std::endl;
            return false;
        }
    }
    std::cout << "TRUE" << std::endl;
    return true;
}

/* Функция - Тестирование алгоритма на время выполнения
 * n - размерность матрицы */
void time_test(int n = 1024){

    //int n = 1024; // Размерность матрицы

    std::vector<double> A = randvec(n * n); // Случайная матрица, на которой тестируем
    std::vector<double> A_copy(A);          // Копия A

    double time_start = 0,  // Время старта отсчета
           time_end = 0,    // Время конца отсчета
           time_res = 0;    // Итоговое время


    /* Один поток */

    // Замеряем время
    time_start = omp_get_wtime();
    LU_Decomposition(A_copy, n * n, n, false);
    time_end = omp_get_wtime();
    time_res = time_end - time_start;
    std::cout << "(1 threads, n = " << n << ") Speed time is " << time_res << " (sec)" << std::endl;


    /* Многа потоков */

    //int num_of_threads = omp_get_max_threads(); // Максимальное возможное кол-во потоков

    for (int i = 2; i <= 4; i = i + 2){

        // Берем изначальный пример
        A_copy = A;

        // Устанавливаем кол-во потоков
        omp_set_num_threads(i);

        // Замеряем время
        time_start = omp_get_wtime();
        LU_Decomposition(A_copy, n * n, n, true);
        time_end = omp_get_wtime();
        time_res = time_end - time_start;
        std::cout << "(" << i << " threads, n = " << n << ") Speed time is " << time_res << " (sec)"<< std::endl;
    }
}


int main(){
    time_test(1024);
    //time_test(2048);
    //test_result();


    return EXIT_SUCCESS;
}
