#include "../include/N-body-problem-cuda.cuh"



__host__ __device__ double my_fmax(double a, double b) {
    return fmax(a, b);
}

__host__ __device__ float my_fmax(float a, float b) {
    return fmaxf(a, b);
}



__device__ float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ float3 operator*(const float3& a, float b) {
    return make_float3(a.x * b, a.y * b, a.z * b);
}

__device__ double3 operator+(const double3& a, const double3& b) {
    return make_double3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ double3 operator-(const double3& a, const double3& b) {
    return make_double3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ double3 operator*(const double3& a, double b) {
    return make_double3(a.x * b, a.y * b, a.z * b);
}


/**
 * @brief Вычисляет квадрат нормы вектора для ускорения.
 * @param v Вектор типа mytype3.
 * @return Куб нормы вектора.
 */
__device__ float norm(const float3& v) {
    float squared_norm = v.x * v.x + v.y * v.y + v.z * v.z;
    return squared_norm * __fsqrt_rd(squared_norm);
}


/**
 * @brief Вычисляет квадрат нормы вектора для ускорения.
 * @param v Вектор типа mytype3.
 * @return Куб нормы вектора.
 */
__device__ double norm(const double3& v) {
    double squared_norm = v.x * v.x + v.y * v.y + v.z * v.z;
    return squared_norm * sqrt(squared_norm);
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
__global__ void f(mytype3* kr, mytype3* kv, mytype* device_m, mytype3* device_r, mytype3* device_v, int N) { // m: 4 + N * BS * 8   p: 1 + N * ( 2 + BS * 9)

    int idx = blockIdx.x * blockDim.x + threadIdx.x; // 1m 1p

    __shared__ mytype3 shared_r[BS];
    __shared__ mytype shared_m[BS];


    mytype3 position = device_r[idx];
    mytype3 temp_sum = {0., 0., 0.};

    for (int i = 0; i < N; i += BS) { // *N

        shared_m[threadIdx.x] = device_m[threadIdx.x + i]; // 1p
        shared_r[threadIdx.x] = device_r[threadIdx.x + i]; // 1p

        __syncthreads();

        for (int j = 0; j < BS; ++j) { // * BS
            if (i + j < N) {
                mytype3 diff = position - shared_r[j]; // 3p

                mytype distance = my_fmax(norm(diff), eps); // 4m 3p
                mytype a = shared_m[j] / distance;                   // 1m

                temp_sum = temp_sum + diff * a;                      // 3m 3p

            }
        }
        __syncthreads();
    } // end * N

    if (idx < N) {
        kv[idx] = temp_sum * G;                                       // 3m
        kr[idx] = device_v[idx];
    }
}


/**
 * @brief Обновляет координаты тел с использованием данных Runge-Kutta.
 * @param device_r Исходные координаты.
 * @param kr Изменения координат.
 * @param tau Шаг интегрирования.
 * @param temp_device_r Обновленные координаты.
 * @param N Число тел.
 *
 *         add<<<blocks, threads>>>(device_r, kr1, tau2, temp_device_r, N, nullptr);
        add<<<blocks, threads>>>(device_v, kv1, tau2, temp_device_v, N, nullptr);
 */
__global__ void
add(mytype3 *device_r, mytype3 *device_v, mytype3 *kr, mytype3 *kv, mytype3 *temp_device_r, mytype3 *temp_device_v, // 7m 7p
    int N, mytype tau) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x; // 1m 1p
    if (idx < N) {
        temp_device_r[idx] = device_r[idx] + kr[idx] * tau; // 3m 3p
        temp_device_v[idx] = device_v[idx] + kv[idx] * tau; // 3m 3p
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
__global__ void summarize(mytype3* device_r, mytype3* device_v, mytype tau,  // 26m 25p
                          mytype3* kr1, mytype3* kv1,
                          mytype3* kr2, mytype3* kv2,
                          mytype3* kr3, mytype3* kv3,
                          mytype3* kr4, mytype3* kv4, int N) {

    int idx = threadIdx.x + blockDim.x * blockIdx.x; // 1m 1p
    mytype tau6 = tau / 6.0; // 1m

    if (idx < N) {

        device_r[idx] = device_r[idx] + (kr1[idx] +
                kr2[idx] * 2.0 + kr3[idx] * 2.0 + kr4[idx]) * tau6; // 12m 12p

        device_v[idx] = device_v[idx] + (kv1[idx] +
                kv2[idx] * 2.0 + kv3[idx] * 2.0 + kv4[idx]) * tau6; // 12m 12p
    }
}

