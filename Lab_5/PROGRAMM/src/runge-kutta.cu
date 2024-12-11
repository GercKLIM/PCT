#include "../include/N-body-problem-cuda.cuh"


/**
 * @brief Метод Рунге-Кутта для решения задачи о движении N тел с использованием CUDA.
 * @param path Путь к файлу для вывода данных.
 * @param global_m Массы тел.
 * @param global_r Начальные координаты тел.
 * @param global_v Начальные скорости тел.
 * @param tau Шаг интегрирования.
 * @param T Время интегрирования.
 * @param output Флаг записи результата.
 * @return Среднее время выполнения одного шага.
 */
float Runge_Kutta(const std::string& path, const std::vector<mytype>& global_m,
                  std::vector<mytype>& global_r, std::vector<mytype>& global_v,
                  mytype tau, mytype T, bool output) {

    int N = global_m.size(), N3 = 3 * N;
    mytype *device_m = nullptr;
    mytype *device_r = nullptr;
    mytype *device_v = nullptr;
    mytype *kr1 = nullptr;
    mytype *kv1 = nullptr;
    mytype *kr2 = nullptr;
    mytype *kv2 = nullptr;
    mytype *kr3 = nullptr;
    mytype *kv3 = nullptr;
    mytype *kr4 = nullptr;
    mytype *kv4 = nullptr;
    mytype *temp_device_r = nullptr;
    mytype *temp_device_v = nullptr;
    mytype tau2 = tau / 2, t0 = 0.0;
    dim3 blocks((N + BS - 1) / BS), threads(BS);

    // Вывод числа тел
    std::cout << "[LOG]: N = " << N << std::endl;

    if (output) {
        for (size_t i = 0; i < N; ++i) {
            write(path, {global_r[3 * i], global_r[3 * i + 1], global_r[3 * i + 2]}, t0, i + 1);
        }
    }

    // Выделение памяти на устройстве
    cudaMalloc(&device_m, N * sizeof(mytype));
    cudaMalloc(&device_r, N3 * sizeof(mytype));
    cudaMalloc(&device_v, N3 * sizeof(mytype));
    cudaMalloc(&temp_device_r, N3 * sizeof(mytype));
    cudaMalloc(&temp_device_v, N3 * sizeof(mytype));
    cudaMalloc(&kr1, N3 * sizeof(mytype));
    cudaMalloc(&kr2, N3 * sizeof(mytype));
    cudaMalloc(&kr3, N3 * sizeof(mytype));
    cudaMalloc(&kr4, N3 * sizeof(mytype));
    cudaMalloc(&kv1, N3 * sizeof(mytype));
    cudaMalloc(&kv2, N3 * sizeof(mytype));
    cudaMalloc(&kv3, N3 * sizeof(mytype));
    cudaMalloc(&kv4, N3 * sizeof(mytype));

    // Копирование данных на устройство
    cudaMemcpy(device_m, global_m.data(), N * sizeof(mytype), cudaMemcpyHostToDevice);
    cudaMemcpy(device_r, global_r.data(), N3 * sizeof(mytype), cudaMemcpyHostToDevice);
    cudaMemcpy(device_v, global_v.data(), N3 * sizeof(mytype), cudaMemcpyHostToDevice);

    cudaEvent_t start, finish;
    cudaEventCreate(&start);
    cudaEventCreate(&finish);

    float time = 0.0f;
    cudaEventRecord(start);

    int iter = 0;
    while (t0 <= T) {
        // Расчёт этапов метода Рунге-Кутта
        f<<<blocks, threads>>>(kr1, kv1, device_m, device_r, device_v, N);

        add<<<blocks, threads>>>(device_r, kr1, tau2, temp_device_r, N);
        add<<<blocks, threads>>>(device_v, kv1, tau2, temp_device_v, N);
        f<<<blocks, threads>>>(kr2, kv2, device_m, temp_device_r, temp_device_v, N);

        add<<<blocks, threads>>>(device_r, kr2, tau2, temp_device_r, N);
        add<<<blocks, threads>>>(device_v, kv2, tau2, temp_device_v, N);
        f<<<blocks, threads>>>(kr3, kv3, device_m, temp_device_r, temp_device_v, N);

        add<<<blocks, threads>>>(device_r, kr3, tau, temp_device_r, N);
        add<<<blocks, threads>>>(device_v, kv3, tau, temp_device_v, N);
        f<<<blocks, threads>>>(kr4, kv4, device_m, temp_device_r, temp_device_v, N);

        summarize<<<blocks, threads>>>(device_r, device_v, tau, kr1, kv1, kr2, kv2, kr3, kv3, kr4, kv4, N);

        t0 += tau;
        ++iter;

        if (output && ((int) (round(1000 * t0)) % 100 == 0)) {
            cudaMemcpy(global_r.data(), device_r, N3 * sizeof(mytype), cudaMemcpyDeviceToHost);
            for (size_t i = 0; i < N; ++i) {
                write(path, {global_r[3 * i], global_r[3 * i + 1], global_r[3 * i + 2]}, t0, i + 1);
            }
        }
    }

    cudaDeviceSynchronize();
    cudaEventRecord(finish);
    cudaEventSynchronize(finish);

    cudaEventElapsedTime(&time, start, finish);

    // Копирование данных обратно на хост
    cudaMemcpy(global_r.data(), device_r, N3 * sizeof(mytype), cudaMemcpyDeviceToHost);
    cudaMemcpy(global_v.data(), device_v, N3 * sizeof(mytype), cudaMemcpyDeviceToHost);

    // Освобождение памяти
    cudaFree(device_m);
    cudaFree(device_r);
    cudaFree(device_v);
    cudaFree(temp_device_r);
    cudaFree(temp_device_v);
    cudaFree(kr1);
    cudaFree(kr2);
    cudaFree(kr3);
    cudaFree(kr4);
    cudaFree(kv1);
    cudaFree(kv2);
    cudaFree(kv3);
    cudaFree(kv4);

    return time / iter;
}