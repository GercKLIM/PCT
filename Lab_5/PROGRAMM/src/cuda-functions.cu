#include "../include/N-body-problem-cuda.cuh"




template <typename T>
__host__ __device__ T my_fmax(T a, T b);

template <>
__host__ __device__ float my_fmax(float a, float b) {
    return fmaxf(a, b);
}

template <>
__host__ __device__ double my_fmax(double a, double b) {
    return fmax(a, b);
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

__device__ double3 operator*(const double3& a, float b) {
    return make_double3(a.x * b, a.y * b, a.z * b);
}


/**
 * @brief Вычисляет квадрат нормы вектора для ускорения.
 * @param v Вектор типа mytype3.
 * @return Куб нормы вектора.
 */
__device__ float norm(const float3& v) {
    float squared_norm = v.x * v.x + v.y * v.y + v.z * v.z;
    return squared_norm * rsqrtf(squared_norm);
}


/**
 * @brief Вычисляет квадрат нормы вектора для ускорения.
 * @param v Вектор типа mytype3.
 * @return Куб нормы вектора.
 */
__device__ double norm(const double3& v) {
    double squared_norm = v.x * v.x + v.y * v.y + v.z * v.z;
    return squared_norm * (1.0 / sqrt(squared_norm));
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
__global__ void f(mytype3* kr, mytype3* kv, mytype* device_m, mytype3* device_r, mytype3* device_v, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ mytype3 shared_r[BS];
    __shared__ float shared_m[BS];

    mytype3 acceleration;
    mytype3 position = device_r[idx];

    for (int i = 0; i < N; i += BS) {
        if (threadIdx.x + i < N) {
            shared_m[threadIdx.x] = device_m[threadIdx.x + i];
            shared_r[threadIdx.x] = device_r[threadIdx.x + i];
        }
        __syncthreads();

        for (int j = 0; j < BS; ++j) {
            if (i + j < N) {
                mytype3 diff = position - shared_r[j];
//                mytype distance = fmaxf(norm(diff), eps);
                mytype distance = my_fmax(norm(diff), eps);
                acceleration = acceleration - diff * (G * shared_m[j] / distance);
            }
        }
        __syncthreads();
    }

    if (idx < N) {
        kv[idx] = acceleration;
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
 */
__global__ void add(mytype3* device_r, mytype3* kr, float tau, mytype3* temp_device_r, int N) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < N) {
        temp_device_r[idx] = device_r[idx] + kr[idx] * tau;
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
__global__ void summarize(mytype3* device_r, mytype3* device_v, float tau,
                          mytype3* kr1, mytype3* kv1,
                          mytype3* kr2, mytype3* kv2,
                          mytype3* kr3, mytype3* kv3,
                          mytype3* kr4, mytype3* kv4, int N) {

    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    float tau6 = tau / 6.0f;

    if (idx < N) {
        device_r[idx] = device_r[idx] + (kr1[idx] + kr2[idx] * 2.0f + kr3[idx] * 2.0f + kr4[idx]) * tau6;
        device_v[idx] = device_v[idx] + (kv1[idx] + kv2[idx] * 2.0f + kv3[idx] * 2.0f + kv4[idx]) * tau6;
    }
}

