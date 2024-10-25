// lab1.cpp : Этот файл содержит функцию "main". Здесь начинается и заканчивается выполнение программы.
//

#include <iomanip>
#include <iostream>
#include <omp.h>
#include <cmath>

using namespace std;

const int n = 1024;
const int blocksize = 32;


void print(double* matr)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            cout << matr[i * n + j] << " ";
        }
        cout << endl;
    }
}

void printB(double* matr)
{
    for (int i = 0; i < blocksize; i++)
    {
        for (int j = 0; j < blocksize; j++)
        {
            cout << matr[i * blocksize + j] << " ";
        }
        cout << endl;
    }
}

void LUnormalDec(double* matr)
{
    double sum1;
    //#pragma omp parallel for default(none) shared(matr) //schedule(dynamic)
    for (int j = 1; j < n; ++j)
        matr[j * n] = matr[j * n] / matr[0];   //first layer

    for (int i = 1; i < n; ++i)
    {
        //sum1 = 0.0;
        //#pragma omp parallel for default(none) shared(matr, i) private(sum1) //schedule(dynamic)
        for (int j = i; j < n; ++j)
        {
            sum1 = 0.0;
            for (int k = 0; k < i; ++k)
            {
                sum1 += matr[i * n + k] * matr[k * n + j];
            }
            matr[n * i + j] -= sum1; // "U" elements
        }
        //#pragma omp parallel for default(none) shared(matr,i) private(sum1) //schedule(dynamic)
        for (int j = i + 1; j < n; ++j)
        {
            sum1 = 0.0;
            for (int k = 0; k < i; ++k)
            {
                sum1 += matr[j * n + k] * matr[k * n + i];
            }
            matr[n * j + i] = (matr[n * j + i] - sum1) / matr[n * i + i];
        }
    }
}

void mulijk(double* A, double* B, double* C)
{
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
        {
            C[i * n + j] = 0.0;
            for (int k = 0; k < n; ++k)
            {
                C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
}

void LU_check(double* matr, double* matr_for_check)
{
    double* L = new double[n * n];
    double* U = new double[n * n];
    for (int i = 0; i < n; ++i)
    {
        L[i * n + i] = 1.0;
        U[i * n + i] = matr[i * n + i];
        for (int j = 0; j < n; ++j)
        {
            if (i > j)
            {
                L[i * n + j] = matr[i * n + j];
                U[i * n + j] = 0.0;
            }
            if (i < j)
            {
                U[i * n + j] = matr[i * n + j];
                L[i * n + j] = 0.0;
            }
        }
    }
    double* M = new double[n * n];

    mulijk(L, U, M);

    double maxx = -100.0;
    for (int i = 0; i < n * n; ++i)
        if (fabs(matr_for_check[i] - M[i]) > maxx) maxx = fabs(matr_for_check[i] - M[i]);
    cout << "error value: " << maxx << endl;
    delete[] L;
    delete[] U;
    delete[] M;
}

double* MatrixCopy(double* matr)
{
    double* copied_matr = new double[n * n];
    for (int i = 0; i < n * n; ++i)
        copied_matr[i] = matr[i];
    return copied_matr;
}

void LUnormalB(double* matr)
{
    double sum1;
    for (int j = 1; j < blocksize; ++j)
        matr[j * blocksize] = matr[j * blocksize] / matr[0];

    for (int i = 1; i < blocksize; ++i)
    {
        //sum1 = 0.0;
        for (int j = i; j < blocksize; ++j)
        {
            sum1 = 0.0;
            for (int k = 0; k < i; ++k)
            {
                sum1 += matr[i * blocksize + k] * matr[k * blocksize + j];
            }
            matr[blocksize * i + j] -= sum1;
        }
        for (int j = i + 1; j < blocksize; ++j)
        {
            sum1 = 0.0;
            for (int k = 0; k < i; ++k)
            {
                sum1 += matr[j * blocksize + k] * matr[k * blocksize + i];
            }
            matr[blocksize * j + i] = (matr[blocksize * j + i] - sum1) / matr[blocksize * i + i];
        }
    }
}

void LUBlockDec(double* matr)
{
    double* block = new double[blocksize * blocksize];
    for (int i = 0; i < n; i += blocksize)
    {
        for (int j = 0; j < blocksize; j++)
            for (int k = 0; k < blocksize; k++)
                block[j * blocksize + k] = matr[n * (j + i) + k + i];
        LUnormalB(block);
        for (int j = 0; j < blocksize; j++)
            for (int k = 0; k < blocksize; k++)
                matr[n * (j + i) + k + i] = block[j * blocksize + k];
        double sum1 = 0.0;
        for (int j = i + blocksize; j < n; ++j)
        {
            matr[(j)*n + i] = matr[(j)*n + i] / block[0];
            for (int k = 1; k < blocksize; ++k)
            {
                sum1 = 0.0;
                for (int p = 0; p < k; ++p)
                {
                    sum1 = sum1 + matr[(j)*n + p + i] * block[p * blocksize + k];
                }
                matr[(j)*n + k + i] = (matr[(j)*n + k + i] - sum1) / block[k * blocksize + k];
            }
        }
        for (int j = i + blocksize; j < n; ++j)
        {
            for (int k = 1; k < blocksize; ++k)
            {
                sum1 = 0.0;
                for (int p = 0; p < k; ++p)
                {
                    sum1 = sum1 + matr[(p + i) * n + j] * block[(k)*blocksize + p];
                }
                matr[(k + i) * n + j] = (matr[(k + i) * n + j] - sum1);
            }
        }
        for (int j = i + blocksize; j < n; ++j)
        {
            for (int k = i + blocksize; k < n; ++k)
            {
                sum1 = 0.0;
                for (int p = 0; p < blocksize; ++p)
                    sum1 += matr[j * n + p + i] * matr[(p + i) * n + k];
                matr[j * n + k] -= sum1;
            }
        }
    }

}


void LUnormalDecParallel(double* matr)
{
    //omp_set_num_threads(8);
    double sum1;
#pragma omp parallel for shared(matr) ///schedule(dynamic)
    for (int j = 1; j < n; ++j)
        matr[j * n] = matr[j * n] / matr[0];   //first layer

    for (int i = 1; i < n; ++i)
    {
        sum1 = 0.0;
#pragma omp parallel for default(none) shared(matr, i) private(sum1) //schedule(dynamic)
        for (int j = i; j < n; ++j)
        {
            sum1 = 0.0;
            for (int k = 0; k < i; ++k)
            {
                sum1 += matr[i * n + k] * matr[k * n + j];
            }
            matr[n * i + j] -= sum1;
        }
#pragma omp parallel for default(none) shared(matr,i) private(sum1) //collapse(2)
        //schedule(dynamic)
        for (int j = i + 1; j < n; ++j)
        {
            sum1 = 0.0;
            for (int k = 0; k < i; ++k)
            {
                sum1 += matr[j * n + k] * matr[k * n + i];
            }
            matr[n * j + i] = (matr[n * j + i] - sum1) / matr[n * i + i];
            //sum1 = 0.0;
        }
    }
}


void LUBlock_v2(double* matr)
{
    double* block = new double[blocksize * blocksize];
    double* L = new double[(n - blocksize) * blocksize];
    double* U = new double[blocksize * (n - blocksize)];

    // Переменные для накопления общего времени
    double total_time_block = 0.0;
    double total_time_below_block = 0.0;
    double total_time_right_block = 0.0;
    double total_time_remaining_square = 0.0;


    for (int i = 0; i < n; i += blocksize)
    {
//#pragma omp parallel for default(none) shared(matr, block, i) // заполнение блока
        for (int j = 0; j < blocksize; j++)
            for (int k = 0; k < blocksize; k++)
                block[j * blocksize + k] = matr[n * (j + i) + k + i];
        //cout << "Block before LU" << endl;
        //printB(block);
        double t_block_start = omp_get_wtime();
        LUnormalB(block); // LU для блока
        double t_block_time = omp_get_wtime() - t_block_start;

        total_time_block += t_block_time;

//#pragma omp parallel for default(none) shared(matr, block, i) // заполнение матрицы блоком
        for (int j = 0; j < blocksize; j++)
            for (int k = 0; k < blocksize; k++)
                matr[n * (j + i) + k + i] = block[j * blocksize + k];
        //cout << "Block after LU" << endl;
        //printB(block);
        double sum = 0.0;

//#pragma omp parallel for default(none) shared(L, matr, i)
        for (int j = i; j < n - blocksize; j++) // заполнение L
            for (int k = 0; k < blocksize; k++)
                L[j * blocksize + k] = matr[(j + blocksize) * n + k + i];

        double t_right_block_start = omp_get_wtime();
//#pragma omp parallel for default(none) shared(U, matr, i)
        for (int j = 0; j < blocksize; j++) // заполнение U
            for (int k = i; k < n - blocksize; k++)
                U[j * (n - blocksize) + k] = matr[(j + i) * n + blocksize + k];
        double t_right_block_time = omp_get_wtime() - t_right_block_start;
        total_time_right_block += t_right_block_time;


        double t_below_block_start = omp_get_wtime();
//#pragma omp parallel for default(none) shared(L, block, i) private(sum)
        for (int j = i + blocksize; j < n; ++j)
        {
            L[(j - blocksize) * blocksize] = L[(j - blocksize) * blocksize] / block[0];
            for (int k = 1; k < blocksize; ++k)
            {
                for (int p = 0; p < k; ++p)
                    sum += L[(j - blocksize) * blocksize + p] * block[p * blocksize + k];		// L
                L[(j - blocksize) * blocksize + k] = (L[(j - blocksize) * blocksize + k] - sum) / block[k * blocksize + k];
                sum = 0.0;
            }
        }
        double t_below_block_time = omp_get_wtime() - t_below_block_start;  // Конец измерения
        total_time_below_block += t_below_block_time;


//#pragma omp parallel for default(none) shared(L, matr, i)
        for (int j = i; j < n - blocksize; j++) // заполнение матрицы элементами L
            for (int k = 0; k < blocksize; k++)
                matr[(j + blocksize) * n + k + i] = L[j * blocksize + k];

//#pragma omp parallel for default(none) shared(U, block, i) private(sum)
        for (int j = i + blocksize; j < n; ++j)
        {
            for (int k = 1; k < blocksize; ++k)
            {
                for (int p = 0; p < k; ++p)
                {
                    sum += U[p * (n - blocksize) + j - blocksize] * block[k * blocksize + p];	// U
                }
                U[k * (n - blocksize) + j - blocksize] = (U[k * (n - blocksize) + j - blocksize] - sum);
                sum = 0.0;
            }
        }

//#pragma omp parallel for default(none) shared(U, matr, i)
        for (int j = 0; j < blocksize; j++) // заполнение U
            for (int k = i; k < n - blocksize; k++)
                matr[(j + i) * n + blocksize + k] = U[j * (n - blocksize) + k];

        double t_remaining_square_start = omp_get_wtime();
//#pragma omp parallel for default(none) shared(matr, L, U, i) private(sum) //
        for (int j = i + blocksize; j < n; ++j)
        {
            for (int k = i + blocksize; k < n; ++k)
            {
                for (int p = 0; p < blocksize; ++p)
                    sum += L[(j - blocksize) * blocksize + p] * U[p * (n - blocksize) + k - blocksize];
                matr[j * n + k] -= sum;
                sum = 0.0;
            }
        }
        double t_remaining_square_time = omp_get_wtime() - t_remaining_square_start;
        total_time_remaining_square += t_remaining_square_time;
    }


    cout << "Total time for blocks: " << total_time_block << " seconds" << endl;
    cout << "Total time for below blocks: " << total_time_below_block << " seconds" << endl;
    cout << "Total time for right blocks: " << total_time_right_block << " seconds" << endl;
    cout << "Total time for remaining square: " << total_time_remaining_square << " seconds" << endl;

    delete[] block;
    delete[] L;
    delete[] U;
}

void LUBlockParallel_v2(double* matr)
{
    double* block = new double[blocksize * blocksize];
    double* L = new double[(n - blocksize) * blocksize];
    double* U = new double[blocksize * (n - blocksize)];

    // Переменные для накопления общего времени
    double total_time_block = 0.0;
    double total_time_below_block = 0.0;
    double total_time_right_block = 0.0;
    double total_time_remaining_square = 0.0;

    for (int i = 0; i < n; i += blocksize)
    {
#pragma omp parallel for default(none) shared(matr, block, i) collapse(2)// заполнение блока
        for (int j = 0; j < blocksize; j++)
            for (int k = 0; k < blocksize; k++)
                block[j * blocksize + k] = matr[n * (j + i) + k + i];
        //cout << "Block before LU" << endl;
        //printB(block);
        double t_block_start = omp_get_wtime();
        LUnormalB(block); // LU для блока
        double t_block_time = omp_get_wtime() - t_block_start; // LU для блока

        total_time_block += t_block_time;

#pragma omp parallel for default(none) shared(matr, block, i) collapse(2) // заполнение матрицы блоком
        for (int j = 0; j < blocksize; j++)
            for (int k = 0; k < blocksize; k++)
                matr[n * (j + i) + k + i] = block[j * blocksize + k];
        //cout << "Block after LU" << endl;
        //printB(block);
        double sum = 0.0;

#pragma omp parallel for default(none) shared(L, matr, i) collapse(2)
        for (int j = i; j < n - blocksize; j++) // заполнение L
            for (int k = 0; k < blocksize; k++)
                L[j * blocksize + k] = matr[(j + blocksize) * n + k + i];

        double t_right_block_start = omp_get_wtime();
#pragma omp parallel for default(none) shared(U, matr, i) collapse(2)
        for (int j = 0; j < blocksize; j++) // заполнение U
            for (int k = i; k < n - blocksize; k++)
                U[j * (n - blocksize) + k] = matr[(j + i) * n + blocksize + k];
        double t_right_block_time = omp_get_wtime() - t_right_block_start;
        total_time_right_block += t_right_block_time;

        double t_below_block_start = omp_get_wtime();
#pragma omp parallel for default(none) shared(L, block, i) private(sum)
        for (int j = i + blocksize; j < n; ++j)
        {
            L[(j - blocksize) * blocksize] = L[(j - blocksize) * blocksize] / block[0];
            for (int k = 1; k < blocksize; ++k)
            {
                for (int p = 0; p < k; ++p)
                    sum += L[(j - blocksize) * blocksize + p] * block[p * blocksize + k];		// L
                L[(j - blocksize) * blocksize + k] = (L[(j - blocksize) * blocksize + k] - sum) / block[k * blocksize + k];
                sum = 0.0;
            }
        }
        double t_below_block_time = omp_get_wtime() - t_below_block_start;  // Конец измерения
        total_time_below_block += t_below_block_time;

//#pragma omp parallel for default(none) shared(L, matr, i)
        for (int j = i; j < n - blocksize; j++) // заполнение матрицы элементами L
            for (int k = 0; k < blocksize; k++)
                matr[(j + blocksize) * n + k + i] = L[j * blocksize + k];

#pragma omp parallel for default(none) shared(U, block, i) private(sum) collapse(2)
        for (int j = i + blocksize; j < n; ++j)
        {
            for (int k = 1; k < blocksize; ++k)
            {
                for (int p = 0; p < k; ++p)
                {
                    sum += U[p * (n - blocksize) + j - blocksize] * block[k * blocksize + p];	// U
                }
                U[k * (n - blocksize) + j - blocksize] = (U[k * (n - blocksize) + j - blocksize] - sum);
                sum = 0.0;
            }
        }

#pragma omp parallel for default(none) shared(U, matr, i) collapse(2)
        for (int j = 0; j < blocksize; j++) // заполнение U
            for (int k = i; k < n - blocksize; k++)
                matr[(j + i) * n + blocksize + k] = U[j * (n - blocksize) + k];

        double t_remaining_square_start = omp_get_wtime();
#pragma omp parallel for default(none) shared(matr, L, U, i) private(sum) collapse(2)
        for (int j = i + blocksize; j < n; ++j)
        {
            for (int k = i + blocksize; k < n; ++k)
            {
                for (int p = 0; p < blocksize; ++p)
                    sum += L[(j - blocksize) * blocksize + p] * U[p * (n - blocksize) + k - blocksize];
                matr[j * n + k] -= sum;
                sum = 0.0;
            }
        }
        double t_remaining_square_time = omp_get_wtime() - t_remaining_square_start;
        total_time_remaining_square += t_remaining_square_time;
    }

    cout << "Total time for blocks: " << total_time_block << " seconds" << endl;
    cout << "Total time for below blocks: " << total_time_below_block << " seconds" << endl;
    cout << "Total time for right blocks: " << total_time_right_block << " seconds" << endl;
    cout << "Total time for remaining square: " << total_time_remaining_square << " seconds" << endl;

    delete[] block;
    delete[] L;
    delete[] U;
}

int main()
{
    double t1, t2, t11, t22;
    double* matrix = new double[n * n];
    for (int i = 0; i < n * n; i++)
    {
        //matrix[i] = fRand(-10, 10);
        //matrix[i] = round(sin(i+1) * 100) / 100;
        matrix[i] = double(rand() % 10);
    }
    double* matr_for_check = MatrixCopy(matrix);
    double* matrix2 = MatrixCopy(matrix);
    double* matrix3 = MatrixCopy(matrix);
    double* matrix4 = MatrixCopy(matrix);

    //print(matrix);
    cout << "n = " << n << endl;
    //print(matr_for_check);
    cout << endl;
    cout << "block = " << blocksize << endl;
    cout << endl;


    t1=omp_get_wtime();
    LUnormalDec(matrix3);
    t2 = omp_get_wtime() - t1;
    cout << "time for LUnormal = " << t2<<endl;
    cout << endl;


    t11=omp_get_wtime();
    LUnormalDecParallel(matrix4);
    t22 = omp_get_wtime() - t11;
    cout << "time for LUnormalParallel = " << t22<<endl;
    cout << endl;
    cout << "Speed LU/LU(parall) = " << t2 / t22 << endl;
    cout << endl;

    t1=omp_get_wtime();
    //LUBlockDec(matrix);
    LUBlock_v2(matrix);
    t2 = omp_get_wtime() - t1;
    cout << "time for LUBlock = " << t2<<endl;
    cout << endl;

    t11=omp_get_wtime();
    LUBlockParallel_v2(matrix2);
    t22 = omp_get_wtime() - t11;
    cout << "time for LUBlockParallel = " << t22<<endl;
    cout << endl;
    cout << "Speed LUBlock/LUBlock(parall) = " << t2 / t22 << endl;
    cout << endl;

    cout << "Maximum number of threads=" << omp_get_max_threads() << endl;
    cout << endl;
    cout << endl;
    LU_check(matrix, matr_for_check);
    delete[] matrix;
    delete[] matr_for_check;
    delete[] matrix2;
    delete[] matrix3;
    delete[] matrix4;

}