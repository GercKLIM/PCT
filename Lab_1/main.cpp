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
#include <cmath>

/* Константа допустимой погрешности */
const double EPS = 1e-7;

/* Функция вывода матрицы произвольной */
void print(std::vector<double> matrix, int m, int n){

    for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; ++j) {
                std::cout << matrix[i * n + j] << " ";
            }
            std::cout << std::endl;
        }
    std::cout << std::endl;
}

void print(std::vector<double> matrix, int n){

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; ++j) {
            std::cout << matrix[i * n + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}


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


/* Функция - Реализация "Традиционного алгоритма"(2.3-2.4) LU-разложения матрицы
 *
 *  A - входная матрица в виде одномерного массива по строкам
 *  n - размерность матрицы
 *  use_omp - вкл/выкл распараллеливания
 *
 * */
void LU_Decomposition(std::vector<double>& A, const int& n, bool if_omp_use = false){

    for (int i = 0; i < n-1; i++){

        #pragma omp parallel for if (if_omp_use)
        for (int j = i + 1; j < n; j++){

            A[j * n + i] /= A[i * n + i];

            for (int k = i + 1; k < n; k++){
                A[j * n + k] -= A[j * n + i] * A[i * n + k];
            }
        }
    }
}


/* Функция - Реализация "Блочного алгоритма"(2.10) LU-разложения матрицы
 *
 *  A - входная матрица в виде одномерного массива по строкам
 *  n - размерность матрицы
 *  if_use_omp - вкл/выкл распараллеливания
 *
 * */
void LU_decomposition_block(std::vector<double>& A, const int& n, const int& block_size, const bool& if_omp_use = false) {


        std::vector<double> L22(block_size * block_size, 0.0);       // L22
        std::vector<double> L32(block_size * (n - block_size), 0.0); // L32
        std::vector<double> U23(block_size * (n - block_size), 0.0); // U23

        for (int i = 0; i < n; i += block_size) {

/*        */#pragma omp parallel for default(none) shared(A, L22, i, block_size, n) collapse(2) if(if_omp_use) // заполнение блока
            for (int j = 0; j < block_size; j++) {
                for (int k = 0; k < block_size; k++) {
                    L22[j * block_size + k] = A[n * (j + i) + k + i];
                }
            }

            LU_Decomposition(L22, block_size, if_omp_use);




/*        */#pragma omp parallel for default(none) shared(A, L22, i, n, block_size) collapse(2) if(if_omp_use)// заполнение матрицы блоком
            for (int j = 0; j < block_size; j++) {
                for (int k = 0; k < block_size; k++) {
                    A[n * (j + i) + k + i] = L22[j * block_size + k];
                }
            }

            double sum = 0.0;

/*        */#pragma omp parallel for default(none) shared(L32, A, i, block_size, n) collapse(2) if(if_omp_use)
            for (int j = i; j < n - block_size; j++) {// заполнение L
                for (int k = 0; k < block_size; k++) {
                    L32[j * block_size + k] = A[(j + block_size) * n + k + i];
                }
            }

/*        */#pragma omp parallel for default(none) shared(U23, A, i, block_size, n) collapse(2) if(if_omp_use)
            for (int j = 0; j < block_size; j++) {// заполнение U
                for (int k = i; k < n - block_size; k++) {
                    U23[j * (n - block_size) + k] = A[(j + i) * n + block_size + k];
                }
            }

/*        */#pragma omp parallel for default(none) shared(L32, L22, i, block_size, n) private(sum) if(if_omp_use)
            for (int j = i + block_size; j < n; ++j) {
                L32[(j - block_size) * block_size] = L32[(j - block_size) * block_size] / L22[0];
                for (int k = 1; k < block_size; ++k) {
                    for (int p = 0; p < k; ++p) {
                        sum += L32[(j - block_size) * block_size + p] * L22[p * block_size + k];        // L
                    }

                    L32[(j - block_size) * block_size + k] =
                            (L32[(j - block_size) * block_size + k] - sum) / L22[k * block_size + k];
                    sum = 0.0;
                }
            }

            for (int j = i; j < n - block_size; j++) {// заполнение матрицы элементами L
                for (int k = 0; k < block_size; k++) {
                    A[(j + block_size) * n + k + i] = L32[j * block_size + k];
                }
            }

/*        */#pragma omp parallel for default(none) shared(U23, L22, i, block_size, n) private(sum) collapse(2) if(if_omp_use)
            for (int j = i + block_size; j < n; ++j) {
                for (int k = 1; k < block_size; ++k) {
                    for (int p = 0; p < k; ++p) {
                        sum += U23[p * (n - block_size) + j - block_size] * L22[k * block_size + p];    // U
                    }
                    U23[k * (n - block_size) + j - block_size] = (U23[k * (n - block_size) + j - block_size] - sum);
                    sum = 0.0;
                }
            }

/*        */#pragma omp parallel for default(none) shared(U23, A, i, block_size, n) collapse(2) if(if_omp_use)
            for (int j = 0; j < block_size; j++) {// заполнение U
                for (int k = i; k < n - block_size; k++) {
                    A[(j + i) * n + block_size + k] = U23[j * (n - block_size) + k];
                }
            }

/*        */#pragma omp parallel for default(none) shared(A, L32, U23, i, block_size, n) private(sum) collapse(2) if(if_omp_use)
            for (int j = i + block_size; j < n; ++j) {
                for (int k = i + block_size; k < n; ++k) {
                    for (int p = 0; p < block_size; ++p)
                        sum += L32[(j - block_size) * block_size + p] * U23[p * (n - block_size) + k - block_size];
                    A[j * n + k] -= sum;
                    sum = 0.0;
                }
            }
        }
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
    //std::vector<double> A = {2, 3, 1, 4, 7, -1, -2, -3, -4};
    std::vector<double> A = randvec(12);
    std::vector<double> A_copy(A);
    int n = 3;

    //print(A, m, n);
    LU_Decomposition(A, n * n, n);
    //print(A, m, n);

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


std::vector<double> matrix_dif(std::vector<double> A, std::vector<double> B, int m, int n) {
    std::vector<double> res(m * n, 0.0);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j){
            res[i * n + j] = A[i * n + j] - B[i * n + j];
        }
    }
    //print(res, m, n);
    return res;
}


/* Функция - Тестирование алгоритма на время выполнения
 * n - размерность матрицы */
void time_test_1(){

    int n = 1024; // Размерность матрицы

    std::vector<double> A = randvec(n * n); // Случайная матрица, на которой тестируем
    std::vector<double> A_copy(A);          // Копия A

    double time_start = 0,  // Время старта отсчета
           time_end = 0,    // Время конца отсчета
           time_res = 0;    // Итоговое время


    std::cout << "TRADITIONAL ALGORITHM" << std::endl;
    /* Один поток */

    // Замеряем время
    time_start = omp_get_wtime();
    LU_Decomposition(A_copy, n, false);
    time_end = omp_get_wtime();
    time_res = time_end - time_start;
    std::cout << "(1 threads, n = " << n << ") Speed time is " << time_res << " (sec)" << std::endl;


    /* Многа потоков */

    int num_of_threads = omp_get_max_threads(); // Максимальное возможное кол-во потоков

    for (int i = 2; i <= num_of_threads; i = i + 2){

        // Берем изначальный пример
        A_copy = A;

        // Устанавливаем кол-во потоков
        omp_set_num_threads(i);

        // Замеряем время
        time_start = omp_get_wtime();
        LU_Decomposition(A_copy, n, true);
        time_end = omp_get_wtime();
        time_res = time_end - time_start;
        std::cout << "(" << i << " threads, n = " << n << ") Speed time is " << time_res << " (sec)"<< std::endl;
    }
}


void time_test_2(){

    int n = 1024;       // Размерность матрицы
    int block_size = 32; // Размер блока

    std::vector<double> A = randvec(n * n); // Случайная матрица, на которой тестируем
    std::vector<double> A_copy(A);          // Копия A


    double time_start = 0,  // Время старта отсчета
           time_end = 0,    // Время конца отсчета
           time_res = 0;    // Итоговое время


    std::cout << "BLOCK ALGORITHM" << std::endl;
    /* Один поток */

    // Замеряем время
    time_start = omp_get_wtime();
    LU_decomposition_block(A_copy, n, block_size, false);
    time_end = omp_get_wtime();

    time_res = time_end - time_start;
    std::cout << "(1 threads, n = " << n << ") Speed time is " << time_res << " (sec)" << std::endl;


    /* Многа потоков */

    int num_of_threads = omp_get_max_threads(); // Максимальное возможное кол-во потоков

    for (int i = 2; i <= num_of_threads; i = i + 2) {

        // Берем изначальный пример
        A_copy = A;

        // Устанавливаем кол-во потоков
        omp_set_num_threads(i);

        // Замеряем время
        time_start = omp_get_wtime();
        LU_decomposition_block(A_copy, n, block_size, true);
        time_end = omp_get_wtime();

        time_res = time_end - time_start;
        std::cout << "(" << i << " threads, n = " << n << ") Speed time is " << time_res << " (sec)"<< std::endl;
    }
}


void test(){

    //time_test(1024);
    //time_test(2048);
    //test_result();

    int n = 8;          /* Размерность матрицы */
    int block_size = 4; /* Размерность блока для блочного алгоритма */


    //std::vector<double> A = randvec(n * m);
    std::vector<double> A = randvec(n * n);
    std::vector<double> A_copy(A);
    std::vector<double> A_copy2(A);


    //std::cout << "--------------------------" << std::endl;
    //std::cout << "A =" << std::endl;
    //print(A, n);
    LU_Decomposition(A_copy, n);

    //std::cout << "--------------------------" << std::endl;
    //std::cout << "A_LU1 =" << std::endl;
    //print(A_copy, n);


    LU_decomposition_block(A_copy2, n, block_size);

    //std::cout << "--------------------------" << std::endl;
    //std::cout << "A_LU2 =" << std::endl;
    //print(A_copy2, n);

    //std::cout << "--------------------------" << std::endl;
    //std::cout << "difference =" << std::endl;
    std::vector<double> dif = matrix_dif(A_copy, A_copy2, n, n);

    double norm1 = 0.0;
    for (int i = 0; i < n * n; ++i) {
            norm1 += dif[i];
    }
    std::cout << "Norm1 = " << norm1 << std::endl;
}


int main(){


    time_test_1();
    time_test_2();
    test();
    std::cout << "Complete!" << std::endl;

    return EXIT_SUCCESS;
}
