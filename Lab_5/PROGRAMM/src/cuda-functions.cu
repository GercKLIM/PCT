#include "../include/N-body-problem-cuda.cuh"

/**
 * @brief Вычисляет квадрат нормы вектора для ускорения.
 * @param x1 Компонента x.
 * @param x2 Компонента y.
 * @param x3 Компонента z.
 * @return Куб нормы вектора.
 */
__device__ mytype norm(mytype x1, mytype x2, mytype x3) {
    mytype squared_norm = (x1 * x1) + (x2 * x2) + (x3 * x3);
    return squared_norm * __fsqrt_rd(squared_norm); // Для float
}



/**
 * @brief Основной CUDA-ядро для вычисления производных координат и скоростей.
 * @param kr Вектор изменения координат.
 * @param kv Вектор изменения скоростей.
 * @param device_m Массы тел на устройстве.
 * @param device_r Координаты тел на устройстве.
 * @param device_v Скорости тел на устройстве.
 * @param N Число тел.
 */
__global__ void f(mytype* kr, mytype* kv, mytype* device_m, mytype* device_r, mytype* device_v, int N) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Глобальный индекс текущего потока
    int tIdx = threadIdx.x;                          // Локальный индекс в блоке

    // Локальные переменные для вычислений
    mytype kvx = 0, kvy = 0, kvz = 0.0;
    mytype x = device_r[3 * idx], y = device_r[3 * idx + 1], z = device_r[3 * idx + 2];
    mytype diff_x, diff_y, diff_z;

    __shared__ mytype shared_r[3 * BS], shared_m[BS]; // Общая память для обмена данными между потоками

    for (int i = 0; i < N; i += BS) {
        shared_m[tIdx] = device_m[i + tIdx];
        shared_r[3 * tIdx + 0] = device_r[3 * (i + tIdx) + 0];
        shared_r[3 * tIdx + 1] = device_r[3 * (i + tIdx) + 1];
        shared_r[3 * tIdx + 2] = device_r[3 * (i + tIdx) + 2];

        __syncthreads();

        for (int j = 0; j < BS; ++j) {
            if (i + j < N) {
                diff_x = x - shared_r[3 * j + 0];
                diff_y = y - shared_r[3 * j + 1];
                diff_z = z - shared_r[3 * j + 2];

                mytype a = __fdividef(shared_m[j], fmaxf(norm(diff_x, diff_y, diff_z), eps));

                kvx += diff_x * a;
                kvy += diff_y * a;
                kvz += diff_z * a;
            }
        }
        __syncthreads();
    }

    if (idx < N) {
        kv[3 * idx + 0] = G * kvx;
        kv[3 * idx + 1] = G * kvy;
        kv[3 * idx + 2] = G * kvz;

        for (int i = 0; i < 3; ++i) {
            kr[3 * idx + i] = device_v[3 * idx + i];
        }
    }
}

/**
 * @brief Обновляет координаты тел с использованием данных Runge-Kutta.
 * @param device_r Исходные координаты.
 * @param kr Изменения координат.
 * @param tau Шаг интегрирования.
 * @param temp_device_r Обновленные координаты.
 * @param N Число тел.
 */
__global__ void add(mytype* device_r, mytype* kr, mytype tau, mytype* temp_device_r, int N) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < N) {
        for (int i = 0; i < 3; ++i) {
            temp_device_r[3 * idx + i] = device_r[3 * idx + i] + tau * kr[3 * idx + i];
        }
    }
}

/**
 * @brief Итоговая коррекция координат и скоростей.
 * @param device_r Координаты тел.
 * @param device_v Скорости тел.
 * @param tau Шаг интегрирования.
 * @param kr1..kr4 Вектора изменений координат.
 * @param kv1..kv4 Вектора изменений скоростей.
 * @param N Число тел.
 */
__global__ void summarize(mytype* device_r, mytype* device_v, mytype tau,
                          mytype* kr1, mytype* kv1,
                          mytype* kr2, mytype* kv2,
                          mytype* kr3, mytype* kv3,
                          mytype* kr4, mytype* kv4, int N) {

    int i = threadIdx.x + blockDim.x * blockIdx.x;
    mytype tau6 = tau / 6.0f;

    if (i < N) {
        for (int j = 0; j < 3; ++j) {
            device_r[3 * i + j] += tau6 * (kr1[3 * i + j] + 2.0f * kr2[3 * i + j] + 2.0f * kr3[3 * i + j] + kr4[3 * i + j]);
            device_v[3 * i + j] += tau6 * (kv1[3 * i + j] + 2.0f * kv2[3 * i + j] + 2.0f * kv3[3 * i + j] + kv4[3 * i + j]);
        }
    }
}
