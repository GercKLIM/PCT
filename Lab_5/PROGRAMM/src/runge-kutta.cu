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
                  std::vector<mytype3>& global_r, std::vector<mytype3>& global_v,
                  mytype tau, mytype T, bool output) {

    int N = global_m.size();  // Количество тел

    mytype  *device_m = nullptr;
    mytype3 *device_r = nullptr;
    mytype3 *device_v = nullptr;
    mytype3 *kr1 = nullptr;
    mytype3 *kv1 = nullptr;
    mytype3 *kr2 = nullptr;
    mytype3 *kv2 = nullptr;
    mytype3 *kr3 = nullptr;
    mytype3 *kv3 = nullptr;
    mytype3 *kr4 = nullptr;
    mytype3 *kv4 = nullptr;
    mytype3 *temp_device_r = nullptr;
    mytype3 *temp_device_v = nullptr;


    mytype tau2 = tau / 2.0, t0 = 0.0;

    dim3 blocks((N + BS - 1) / BS); // Число блоков
    dim3 threads(BS);               // Число потоков

    std::cout << "[LOG]: N = " << N << std::endl;

    // Запись начального положения
    if (output) {
        for (size_t i = 0; i < N; ++i) {
            write(path, global_r[i], t0, i + 1);
        }
    }

    // Выделение памяти на устройстве
    cudaMalloc(&device_m, N * sizeof(mytype));
    cudaMalloc(&device_r, N * sizeof(mytype3));
    cudaMalloc(&device_v, N * sizeof(mytype3));
    cudaMalloc(&temp_device_r, N * sizeof(mytype3));
    cudaMalloc(&temp_device_v, N * sizeof(mytype3));
    cudaMalloc(&kr1, N * sizeof(mytype3));
    cudaMalloc(&kr2, N * sizeof(mytype3));
    cudaMalloc(&kr3, N * sizeof(mytype3));
    cudaMalloc(&kr4, N * sizeof(mytype3));
    cudaMalloc(&kv1, N * sizeof(mytype3));
    cudaMalloc(&kv2, N * sizeof(mytype3));
    cudaMalloc(&kv3, N * sizeof(mytype3));
    cudaMalloc(&kv4, N * sizeof(mytype3));

    // Копирование данных на устройство
    cudaMemcpy(device_m, global_m.data(), N * sizeof(mytype), cudaMemcpyHostToDevice);
    cudaMemcpy(device_r, global_r.data(), N * sizeof(mytype3), cudaMemcpyHostToDevice);
    cudaMemcpy(device_v, global_v.data(), N * sizeof(mytype3), cudaMemcpyHostToDevice);

    cudaEvent_t start, finish;
    cudaEventCreate(&start);
    cudaEventCreate(&finish);

    /* КОЛ-ВО ОПЕРАЦИЙ
     *
     * M = threads * block * ( 4 * (4 + N * BS * 7) + 3 * ( 7 ) + 1 * 26)
     * P = threads * block * ( 4 * ( 1 + N * (2 + 9 * BS) + 3 * 7 + 1 * 25) + 2
     *
     * */


    float time = 0.0f;
    cudaEventRecord(start);

    int iter = 0;
    while (t0 <= T) { // * threads * block
        // Расчёт этапов метода Рунге-Кутта
        f<<<blocks, threads>>>(kr1, kv1, device_m, // m: 4 + N * BS * 7   p: 1 + N * ( 2 + BS * 9)
                               device_r, device_v, N);
        add<<<blocks, threads>>>(device_r, device_v, kr1,
                                 kv1, temp_device_r, temp_device_v, N, tau2); // 7m 7p

        f<<<blocks, threads>>>(kr2, kv2, device_m,
                               temp_device_r, temp_device_v, N); // m: 4 + N * BS * 7   p: 1 + N * ( 2 + BS * 9)
        add<<<blocks, threads>>>(device_r, device_v, kr2,
                                 kv2, temp_device_r, temp_device_v, N, tau2); // 7m 7p

        f<<<blocks, threads>>>(kr3, kv3, device_m,
                               temp_device_r, temp_device_v, N); // m: 4 + N * BS * 7   p: 1 + N * ( 2 + BS * 9)
        add<<<blocks, threads>>>(device_r, device_v, kr3,
                                 kv3, temp_device_r, temp_device_v, N, tau); // 7m 7p

        f<<<blocks, threads>>>(kr4, kv4, device_m,
                               temp_device_r, temp_device_v, N); // m: 4 + N * BS * 7   p: 1 + N * ( 2 + BS * 9)
        summarize<<<blocks, threads>>>(device_r, device_v,
                                       tau, kr1, kv1, kr2, kv2, kr3, kv3, kr4, kv4, N); // 26m 25p

        t0 += tau; // 1p
        ++iter; // 1p

        if (output) {
            cudaMemcpy(global_r.data(), device_r, N * sizeof(mytype3),
                       cudaMemcpyDeviceToHost);
            for (size_t i = 0; i < N; ++i) {
                write(path, global_r[i], t0, i + 1);
            }
        }
    }

    cudaEventRecord(finish);
    cudaEventSynchronize(finish);

    cudaEventElapsedTime(&time, start, finish);

    // Копирование данных обратно на хост
    cudaMemcpy(global_r.data(), device_r, N * sizeof(mytype3), cudaMemcpyDeviceToHost);
    cudaMemcpy(global_v.data(), device_v, N * sizeof(mytype3), cudaMemcpyDeviceToHost);

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

    return 0.001 * time / iter ;
}
