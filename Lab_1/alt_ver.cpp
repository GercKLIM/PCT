#include <iostream>
#include <vector> // Для доступа к типу std::vector
#include "omp.h"  // Для распараллеливания вычислений
#include <random> // Для генерации случайных чисел
#include <ctime>  // Для инициализации генератора
#include <cmath>


void LU1(std::vector<double>& A, int m, int n, int i0, int j0, int N)
{
    int min = std::min(m - 1, n);
    for (int i = 0; i < min; ++i) {
        for (int j = i + 1; j < m; ++j)
            A[(j + i0) * N + i + j0] /= A[(i + i0) * N + i + j0];

        if (i < n - 1) {
            for (int j = i + 1; j < m; ++j)
                for (int k = i + 1; k < n; ++k)
                    A[(j + i0) * N + k + j0] -= A[(j + i0) * N + i + j0] * A[(i + i0) * N + k + j0];
        }
    }

}

void FindU23(std::vector<double>& L22, std::vector<double>& U23, int N, int block_size, int i) {
    for (int k = 0; k < N - i - block_size; ++k)
        for (int p = 0; p < block_size; ++p)
            for (int j = p - 1; j >= 0; --j)
                U23[p * (N - i - block_size) + k] -=
                        L22[p * block_size + j] * U23[j * (N - i - block_size) + k];

}

void LU(std::vector<double>& A, int M, int N)
{
    for (int i = 0; i < std::min(N, M - 1); ++i)
    {
        for (int j = i + 1; j < M; ++j)
        {
            A[j * N + i] /= A[i * N + i];
            for (int k = i + 1; k < N; ++k)
            {
                A[j * N + k] -= A[j * N + i] * A[i * N + k];
            }
        }
    }
}

void LU_decomposition_block2(std::vector<double>& A, const int& N, const int& block_size) {
    int num_blocks = N / block_size;

    int counter;

    std::vector<double> column(N * block_size, 0);
    std::vector<double> L22(block_size * block_size, 0);
    std::vector<double> L32(block_size * (N - block_size), 0);
    std::vector<double> U23((N - block_size) * block_size, 0);

    for (int i = 0; i < N - 1; i += block_size) {

        LU1(A, N - i, block_size, i, i, N);

        // L22
        for (int p = i; p < i + block_size; ++p)
            for (int q = i; q < i + block_size; ++q)
            {
                if (q == p)
                    L22[(q - i) * block_size + (p - i)] = 1;
                else
                    L22[(q - i) * block_size + (p - i)] = A[q * N + p];
            }
        // L32
        for (int j = 0 /*i*/; j < N - block_size - i; ++j)
            for (int k = 0; k < block_size; ++k) {
                U23[k * (N - i - block_size) + j] = A[(i + k) * N + j + i + block_size];
                L32[j * block_size + k] = A[(i + j + block_size) * N + k + i];
            }

        FindU23(L22, U23, N, block_size, i);

        // записываем в матрицу
        counter = 0;
        for (int q = i; q < i + block_size; ++q)
            for (int p = i + block_size; p < N; ++p)
            {
                A[q * N + p] = U23[counter];
                counter++;
            }

        for (int p = 0; p < N - i - block_size; ++p)
            for (int k = 0; k < block_size; ++k)
                for (int j = 0; j < N - i - block_size; ++j)
                    A[(p + i + block_size) * N + (j + i + block_size)] -= L32[p * block_size + k] * U23[k * (N - i - block_size) + j];

    }
}

void FillMatrix(std::vector<double>& A, int N)
{
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            A[i * N + j] = rand() / 10000.0;
}

int main(){
    //time_test(1024);
    //time_test(2048);
    //test_result();

    int n = 8,
            m = 8,
            block_size = 4;


    //std::vector<double> A = randvec(n * m);
    std::vector<double> A(n * m, 0);
    FillMatrix(A, n);
    std::vector<double> A_copy(A);
    std::vector<double> A_copy2(A);


    std::cout << "--------------------------" << std::endl;
    std::cout << "A =" << std::endl;
    print(A, m, n);
    LU_Decomposition(A_copy, n * m, m, n);

    std::cout << "--------------------------" << std::endl;
    std::cout << "A_LU1 =" << std::endl;
    print(A_copy, m, n);


    LU_decomposition_block2(A_copy2, n, block_size);

    std::cout << "--------------------------" << std::endl;
    std::cout << "A_LU2 =" << std::endl;
    print(A_copy2, m, n);

    std::cout << "--------------------------" << std::endl;
    std::cout << "difference =" << std::endl;
    matrix_dif(A_copy, A_copy2, m, n);

    return EXIT_SUCCESS;
}