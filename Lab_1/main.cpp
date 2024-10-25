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
#include <fstream>// Для работы с файлами



/* -------------------------------- */
/* ### ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ  ### */
/* -------------------------------- */



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

/* Функция вывода матрицы квадратной */
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



/* -------------------------- */
/* ### ФУНКЦИИ АЛГОРИТМОВ ### */
/* -------------------------- */



/* Функция - Реализация "Традиционного алгоритма"(2.3-2.4) LU-разложения матрицы
 *
 *  A - входная матрица в виде одномерного массива по строкам
 *  n - размерность матрицы
 *  use_omp - вкл/выкл распараллеливания
 *
 * */
void LU_Decomposition(std::vector<double>& A, const int& n, bool if_omp_use = false){

/**///#pragma omp parallel for default(none) shared(A, n) if (if_omp_use) //schedule(dynamic)
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



/* Функция - Обратный ход метода Гаусса */
void back_gauss(std::vector<double>& U23, std::vector<double>& L22, const int& i, const int& block_size, const int& n, double& sum, const bool& if_omp_use) {

/**/#pragma omp parallel for default(none) shared(U23, L22, i, block_size, n) private(sum) collapse(2) if(if_omp_use)
    for (int j = i + block_size; j < n; ++j) {
        for (int k = 1; k < block_size; ++k) {
            for (int p = 0; p < k; ++p) {
                sum += U23[p * (n - block_size) + j - block_size] * L22[k * block_size + p];
            }
            U23[k * (n - block_size) + j - block_size] = (U23[k * (n - block_size) + j - block_size] - sum);
            sum = 0.0;
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

            /* Заполнение L22 */
/*        */#pragma omp parallel for default(none) shared(A, L22, i, block_size, n) collapse(2) if(if_omp_use) // заполнение блока
            for (int j = 0; j < block_size; j++) {
                for (int k = 0; k < block_size; k++) {
                    L22[j * block_size + k] = A[n * (j + i) + k + i];
                }
            }

            /* LU-разложение блока */
            LU_Decomposition(L22, block_size, if_omp_use);

            /* Записываем разложение в A */
/*        */#pragma omp parallel for default(none) shared(A, L22, i, n, block_size) collapse(2) if(if_omp_use)
            for (int j = 0; j < block_size; j++) {
                for (int k = 0; k < block_size; k++) {
                    A[n * (j + i) + k + i] = L22[j * block_size + k];
                }
            }

            double sum = 0.0;

            /* Заполнение L32 */
/*        */#pragma omp parallel for default(none) shared(L32, A, i, block_size, n) collapse(2) if(if_omp_use)
            for (int j = i; j < n - block_size; j++) {
                for (int k = 0; k < block_size; k++) {
                    L32[j * block_size + k] = A[(j + block_size) * n + k + i];
                }
            }

            /* Заполнение U23 */
/*        */#pragma omp parallel for default(none) shared(U23, A, i, block_size, n) collapse(2) if(if_omp_use)
            for (int j = 0; j < block_size; j++) {
                for (int k = i; k < n - block_size; k++) {
                    U23[j * (n - block_size) + k] = A[(j + i) * n + block_size + k];
                }
            }

            /* Вычисление L32  */
/*        */#pragma omp parallel for default(none) shared(L32, L22, i, block_size, n) private(sum) if(if_omp_use)
            for (int j = i + block_size; j < n; ++j) {
                L32[(j - block_size) * block_size] = L32[(j - block_size) * block_size] / L22[0];
                for (int k = 1; k < block_size; ++k) {
                    for (int p = 0; p < k; ++p) {
                        sum += L32[(j - block_size) * block_size + p] * L22[p * block_size + k];
                    }

                    L32[(j - block_size) * block_size + k] =
                            (L32[(j - block_size) * block_size + k] - sum) / L22[k * block_size + k];
                    sum = 0.0;
                }
            }

            /* Записываем в A */
            for (int j = i; j < n - block_size; j++) {
                for (int k = 0; k < block_size; k++) {
                    A[(j + block_size) * n + k + i] = L32[j * block_size + k];
                }
            }


            /* Обратный ход метода Гаусса */
            back_gauss(U23, L22, i, block_size, n, sum, if_omp_use);


/*        */#pragma omp parallel for default(none) shared(U23, A, i, block_size, n) collapse(2) if(if_omp_use)
            for (int j = 0; j < block_size; j++) {// заполнение U
                for (int k = i; k < n - block_size; k++) {
                    A[(j + i) * n + block_size + k] = U23[j * (n - block_size) + k];
                }
            }

            /* Пункт 3) */
/*        */#pragma omp parallel for default(none) shared(A, L32, U23, i, block_size, n) private(sum) collapse(2) if(if_omp_use)
            for (int j = i + block_size; j < n; ++j) {
                for (int k = i + block_size; k < n; ++k) {
                    for (int p = 0; p < block_size; ++p) {
                        sum += L32[(j - block_size) * block_size + p] * U23[p * (n - block_size) + k - block_size];
                    }
                    A[j * n + k] -= sum;
                    sum = 0.0;
                }
            }
        }
}



/* --------------------------------------- */
/* ### ФУНКЦИИ ТЕСТИРОВАНИЯ АЛГОРИТМОВ ### */
/* --------------------------------------- */



/* Функция генерация случайного вектора размера n */
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

/* Функция матричной разности */
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

/* Функция - Тестирование Традиционного алгоритма на время выполнения */
std::vector<std::vector<double>> time_test_1(int n){

    //int n = 1024; // Размерность матрицы

    std::vector<double> A = randvec(n * n); // Случайная матрица, на которой тестируем
    std::vector<double> A_copy(A);          // Копия A

    std::vector<std::vector<double>> times;
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
    times.push_back({1, time_res});

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
        times.push_back({1.0 * i, time_res});
    }


    return times;
}

/* Функция - Тестирование Блочного алгоритма на время выполнения */
std::vector<std::vector<double>> time_test_2(int n, int block_size){

    //int n = 1024;       // Размерность матрицы
    //int block_size = 64; // Размер блока

    std::vector<double> A = randvec(n * n); // Случайная матрица, на которой тестируем
    std::vector<double> A_copy(A);          // Копия A

    std::vector<std::vector<double>> times;
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
    std::cout << "(1 threads, n = " << n << ", B = " << block_size << ") Speed time is " << time_res << " (sec)" << std::endl;
    times.push_back({1, time_res});

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
        std::cout << "(" << i << " threads, n = " << n << ", B = " << block_size << ") Speed time is " << time_res << " (sec)"<< std::endl;
        times.push_back({1.0 * i, time_res});
    }
    return times;
}

/* Функция - Проверка решения на корректность */
void test(){

    int n = 8;          /* Размерность матрицы */
    int block_size = 4; /* Размерность блока для блочного алгоритма */

    std::vector<double> A = randvec(n * n);
    std::vector<double> A_copy(A);
    std::vector<double> A_copy2(A);

    LU_Decomposition(A_copy, n);
    LU_decomposition_block(A_copy2, n, block_size);

    /* Матричная разность A_copy и A_copy2*/
    std::vector<double> dif(n * n, 0.0);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            dif[i * n + j] = A_copy[i * n + j] - A_copy2[i * n + j];
        }
    }

    double norm1 = 0.0;
    for (int i = 0; i < n * n; ++i) {
            norm1 += dif[i];
    }
    std::cout << "Norm1 = " << norm1 << std::endl;
}

void time_test(){

    int n = 1024;          /* Размерность матрицы */
    int block_size = 128; /* Размерность блока для блочного алгоритма */

    time_test_1(n);
    time_test_2(n, block_size);

}

void make_data_for_grid(){

    std::vector<int> ns = {1024};          /* Размерность матрицы */
    int block_size = 64; /* Размерность блока для блочного алгоритма */
    std::ofstream file1, file2;
    int n;

    for (int ins = 0; ins < ns.size(); ++ins) {

        n = ns[ins];

        file1.open(("output_data_traditional_" + std::to_string(n) + ".txt"));
        std::vector<std::vector<double>> times_traditional = time_test_1(n);
        std::vector<std::vector<double>> times_block = time_test_2(n, block_size);

        for (int i = 0; i < times_traditional.size(); ++i) {
            file1 << times_traditional[i][0];
            file1 << " ";
            file1 << times_traditional[i][1];
            file1 << std::endl;
        }
        file1.close();

        file2.open(("output_data_block_" + std::to_string(n) + ".txt"));
        for (int i = 0; i < times_block.size(); ++i) {
            file2 << times_block[i][0];
            file2 << " ";
            file2 << times_block[i][1];
            file2 << std::endl;
        }
        file2.close();
    }





};


int main(){

    //time_test();
    //test();
    make_data_for_grid();


    std::cout << "Complete!" << std::endl;

    return EXIT_SUCCESS;
}
