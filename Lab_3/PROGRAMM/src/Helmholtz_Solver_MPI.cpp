
/* -------------------------------------------------------------- */
/* ### РЕАЛИЗАЦИЯ ФУНКЦИЙ РЕШЕНИЯ УРАВНЕНИЯ ГЕЛЬМГОЛЬЦА С MPI ### */
/* -------------------------------------------------------------- */


#include "../include/Helmholtz_Solver.h"






void Method_Jacobi_P2P(MethodResultInfo& result, std::function<double(double, double)>&f, const double& K, const int& N,
                       const double& eps, const int& max_iterations){

    int NP = 1; // Начальное значение кол-ва потоков программы
    int ID = 0; // Начальное значение ID (номера) исполняемого процесса
    MPI_Comm_size(MPI_COMM_WORLD, &NP); // Получаем кол-во исполняемых программ
    MPI_Comm_rank(MPI_COMM_WORLD, &ID); // Получаем номер исполняемой программы

    std::vector<int> str_per_proc; // Строка на процесс
    std::vector<int> nums_start;   // Индекс первой строки для потока

    int str_local; // Кол-во строк в процессе
    int nums_local; // Индекс первой строки, которой процесс будет обрабатывать
    double norm_local; // Норма
    double norm_err;

    /* Раздача работы процессам */
    Work_Distribution(NP, N, str_local, nums_local, str_per_proc, nums_start);

    std::vector<double> y_local(str_local * N); // Блок матрицы на процесс
    std::vector<double> y_next_top(N);             // Текущая приближение
    std::vector<double> y_prev_low(N);             // Предыдущее приближение

    int source_proc = ID ? ID - 1 : NP - 1; // myid == 0 = np - 1 // Процесс из которого отправляем
    int dest_proc = (ID != (NP - 1)) ? ID + 1 : 0; // у myid == np - 1 = 0 // Процесс в который отправляем

    int scount = (ID != (NP - 1)) ? N : 0; // у myid == np - 1 = 0 // Кол-во отправляемых данных
    int rcount = ID ? N : 0; // myid == 0 = 0                      // Кол-во получаемых данных

    double h = 1.0 / (double)(N - 1); // Шаг сетки
    double hh = h * h;                // Квадрат шага
    double kk = K * K;                // Квадрат коэф. К
    double q = 1.0 / (4.0 + kk * hh); // Выражение из схемы


    std::vector<double> y; // Общее решение
    if (ID == 0)
        y.resize(N * N);

    double t1 = -MPI_Wtime();

    std::vector<double> temp(y_local.size()); // Копия приближения
    for (int iteration = 1; iteration < max_iterations; iteration++) {

        std::swap(temp, y_local);

        // Пересылаем нижние строки всеми процессами кроме последнего
        MPI_Send(temp.data() + (str_local - 1) * N, scount, MPI_DOUBLE, dest_proc, 1, MPI_COMM_WORLD);
        MPI_Recv(y_prev_low.data(), rcount, MPI_DOUBLE, source_proc, 1, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

        // Пересылаем верхние строки всеми процессами кроме нулевого
        MPI_Send(temp.data(), rcount, MPI_DOUBLE, source_proc, 2, MPI_COMM_WORLD);
        MPI_Recv(y_next_top.data(), scount, MPI_DOUBLE, dest_proc, 2, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

        /* Пересчитываем все строки в полосе кроме верхней и нижней */
        for (int i = 1; i < str_local - 1; ++i)
            for (int j = 1; j < N - 1; ++j)
                y_local[i * N + j] = (temp[(i + 1) * N + j] + temp[(i - 1) * N + j] + temp[i * N + (j + 1)]
                                      + temp[i * N + (j - 1)] + h * h * f((nums_local + i) * h, j * h)) * q;

        /* Пересчитываем верхние строки */
        if (ID != 0)
            for (int j = 1; j < N - 1; ++j)
                y_local[j] = (temp[N + j] + y_prev_low[j] + temp[j + 1] + temp[j - 1]
                              + hh * f(nums_local * h, j * h)) * q;

        /* Пересчитываем нижние строки */
        if (ID != NP - 1)
            for (int j = 1; j < N - 1; ++j)
                y_local[(str_local - 1) * N + j] = (y_next_top[j] + temp[(str_local - 2) * N + j]
                                                    + temp[(str_local - 1) * N + (j + 1)] + temp[(str_local - 1) * N + (j - 1)]
                                                    + hh * f((nums_local + (str_local - 1)) * h, j * h)) * q;

        norm_local = NOD((int)temp.size(), h, temp, y_local);

        /* Выбираем максимальную ошибку среди всех прцоессов */
        MPI_Allreduce(&norm_local, &norm_err, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        if (norm_err < eps) {
            result.iterations = iteration;
            break;
        }
    }

    t1 += MPI_Wtime();

    for (int i = 0; i < NP; ++i) {
        str_per_proc[i] *= N;
        nums_start[i] *= N;
    }

// Ошибка здесь!

    /* Собираем общее решение */
    MPI_Gatherv(y_local.data(), str_local * N,
                MPI_DOUBLE, y.data(),
                str_per_proc.data(),
                nums_start.data(),
                MPI_DOUBLE, 0,
                MPI_COMM_WORLD);


    /* Записываем результаты */
    if (ID == 0) {
        result.time = t1;
        result.norm_iter = norm_err;
        result.NP = NP;
        result.Y = y;
        result.Yp = temp;
        result.method_name =  "Method Yacobi (SEND, RECV)";
        result.NP = NP;
    }
}

void Method_Jacobi_SIMULT(MethodResultInfo& result, std::function<double(double, double)>&f, const double& K, const int& N,
                          const double& eps, const int& max_iterations) {

    int NP = 1; // Начальное значение кол-ва потоков программы
    int ID = 0; // Начальное значение ID (номера) исполняемого процесса
    MPI_Comm_size(MPI_COMM_WORLD, &NP); // Получаем кол-во исполняемых программ
    MPI_Comm_rank(MPI_COMM_WORLD, &ID); // Получаем номер исполняемой программы

    std::vector<int> str_per_proc; // Строка на процесс
    std::vector<int> nums_start;   // Индекс первой строки для потока

    int str_local; // Кол-во строк в процессе
    int nums_local; // Индекс первой строки, которой процесс будет обрабатывать
    double norm_local; // Норма
    double norm_err;

    /* Раздача работы процессам */
    Work_Distribution(NP, N, str_local, nums_local, str_per_proc, nums_start);

    std::vector<double> y_local(str_local * N); // Блок матрицы на процесс
    std::vector<double> y_next_top(N);             // Текущая приближение
    std::vector<double> y_prev_low(N);             // Предыдущее приближение


    int source_proc = ID ? ID - 1 : NP - 1; // myid == 0 = np - 1 // Процесс из которого отправляем
    int dest_proc = (ID != (NP - 1)) ? ID + 1 : 0; // у myid == np - 1 = 0 // Процесс в который отправляем

    int scount = (ID != (NP - 1)) ? N : 0; // у myid == np - 1 = 0 // Кол-во отправляемых данных
    int rcount = ID ? N : 0; // myid == 0 = 0                      // Кол-во получаемых данных

    double h = 1.0 / (double)(N - 1); // Шаг сетки
    double hh = h * h;                // Квадрат шага
    double kk = K * K;                // Квадрат коэф. К
    double q = 1.0 / (4.0 + kk * hh); // Выражение из схемы

    std::vector<double> y; // Общее решение
    if (ID == 0)
        y.resize(N * N);

    double t1 = -MPI_Wtime();

    std::vector<double> temp(y_local.size()); // Копия приближения
    for (int iteration = 1; iteration < max_iterations; iteration++) {

        std::swap(temp, y_local);

        // пересылаем нижние и верхние строки
        MPI_Sendrecv(temp.data() + (str_local - 1) * N, scount,
                     MPI_DOUBLE, dest_proc, 42, y_prev_low.data(),
                     rcount, MPI_DOUBLE, source_proc, 42,
                     MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

        MPI_Sendrecv(temp.data(), rcount, MPI_DOUBLE,
                     source_proc, 46, y_next_top.data(), scount,
                     MPI_DOUBLE, dest_proc, 46, MPI_COMM_WORLD,
                     MPI_STATUSES_IGNORE);

        /* пересчитываем все строки в полосе кроме верхней и нижней */
        for (int i = 1; i < str_local - 1; ++i)
            for (int j = 1; j < N - 1; ++j)
                y_local[i * N + j] = (temp[(i + 1) * N + j] + temp[(i - 1) * N + j] +
                                      temp[i * N + (j + 1)] + temp[i * N + (j - 1)] +
                                      hh * f((nums_local + i) * h, j * h)) * q;

        /* пересчитываем верхние строки */
        if (ID != 0)
            for (int j = 1; j < N - 1; ++j)
                y_local[j] = (temp[N + j] + y_prev_low[j] + temp[j + 1]
                              + temp[j - 1] + hh * f(nums_local * h, j * h)) * q;

        /* пересчитываем нижние строки */
        if (ID != NP - 1)
            for (int j = 1; j < N - 1; ++j)
                y_local[(str_local - 1) * N + j] = (y_next_top[j] + temp[(str_local - 2) * N + j]
                                                    + temp[(str_local - 1) * N + (j + 1)] +
                                                    temp[(str_local - 1) * N + (j - 1)] +
                                                    hh * f((nums_local + (str_local - 1)) * h, j * h)) * q;

        norm_local = NOD((int)temp.size(), h, temp, y_local);

        MPI_Allreduce(&norm_local, &norm_err, 1,
                      MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        if (norm_err < eps) {
            result.iterations = iteration;
            break;
        }

    }

    t1 += MPI_Wtime();

    for (int i = 0; i < NP; ++i) {
        str_per_proc[i] *= N;
        nums_start[i] *= N;
    }

    /* Собираем общее решение */
    MPI_Gatherv(y_local.data(), str_local * N, MPI_DOUBLE, y.data(),
                str_per_proc.data(), nums_start.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* Записываем результаты */
    if (ID == 0) {
        result.time = t1;
        result.norm_iter = norm_err;
        result.NP = NP;
        result.Y = y;
        result.Yp = temp;
        result.method_name =  "Method Yacobi (SENDRECV)";
        result.NP = NP;
    }
}

void Method_Jacobi_NOBLOCK(MethodResultInfo& result, std::function<double(double, double)>&f, const double& K, const int& N,
                           const double& eps, const int& max_iterations) {

    MPI_Request* send_req1;
    MPI_Request* send_req2;
    MPI_Request* recv_req1;
    MPI_Request* recv_req2;

    int NP = 1; // Начальное значение кол-ва потоков программы
    int ID = 0; // Начальное значение ID (номера) исполняемого процесса
    MPI_Comm_size(MPI_COMM_WORLD, &NP); // Получаем кол-во исполняемых программ
    MPI_Comm_rank(MPI_COMM_WORLD, &ID); // Получаем номер исполняемой программы

    std::vector<int> str_per_proc; // Строка на процесс
    std::vector<int> nums_start;   // Индекс первой строки для потока

    int str_local; // Кол-во строк в процессе
    int nums_local; // Индекс первой строки, которой процесс будет обрабатывать
    double norm_local; // Норма
    double norm_err;

    /* Раздача работы процессам */
    Work_Distribution(NP, N, str_local, nums_local, str_per_proc, nums_start);

    send_req1 = new MPI_Request[2], recv_req1 = new MPI_Request[2];
    send_req2 = new MPI_Request[2], recv_req2 = new MPI_Request[2];

    std::vector<double> y_local(str_local * N); // Блок матрицы на процесс
    std::vector<double> y_next_top(N);             // Текущая приближение
    std::vector<double> y_prev_low(N);             // Предыдущее приближение

    int source_proc = ID ? ID - 1 : NP - 1; // myid == 0 = np - 1 // Процесс из которого отправляем
    int dest_proc = (ID != (NP - 1)) ? ID + 1 : 0; // у myid == np - 1 = 0 // Процесс в который отправляем

    int scount = (ID != (NP - 1)) ? N : 0; // у myid == np - 1 = 0 // Кол-во отправляемых данных
    int rcount = ID ? N : 0; // myid == 0 = 0                      // Кол-во получаемых данных

    double h = 1.0 / (double)(N - 1); // Шаг сетки
    double hh = h * h;                // Квадрат шага
    double kk = K * K;                // Квадрат коэф. К
    double q = 1.0 / (4.0 + kk * hh); // Выражение из схемы


    std::vector<double> temp(y_local.size()); // Копия приближения

    // пересылаем верхние и нижние строки temp
    MPI_Send_init(temp.data(), rcount, MPI_DOUBLE,
                  source_proc, 0, MPI_COMM_WORLD, send_req1);

    MPI_Recv_init(y_prev_low.data(), rcount, MPI_DOUBLE,
                  source_proc, 1, MPI_COMM_WORLD, recv_req1);

    MPI_Send_init(temp.data() + (str_local - 1) * N, scount,
                  MPI_DOUBLE, dest_proc, 1, MPI_COMM_WORLD,
                  send_req1 + 1);

    MPI_Recv_init(y_next_top.data(), scount, MPI_DOUBLE,
                  dest_proc, 0, MPI_COMM_WORLD, recv_req1 + 1);

    // пересылаем верхние и нижние строки y_local
    MPI_Send_init(y_local.data(), rcount, MPI_DOUBLE,
                  source_proc, 0, MPI_COMM_WORLD, send_req2);

    MPI_Recv_init(y_prev_low.data(), rcount, MPI_DOUBLE,
                  source_proc, 1, MPI_COMM_WORLD, recv_req2);

    MPI_Send_init(y_local.data() + (str_local - 1) * N, scount,
                  MPI_DOUBLE, dest_proc, 1, MPI_COMM_WORLD, send_req2 + 1);

    MPI_Recv_init(y_next_top.data(), scount, MPI_DOUBLE,
                  dest_proc, 0, MPI_COMM_WORLD, recv_req2 + 1);

    std::vector<double> y; // Общее решение
    if (ID == 0)
        y.resize(N * N);

    double t1 = -MPI_Wtime();
    for (int iteration = 1; iteration < max_iterations; iteration++) {

        std::swap(temp, y_local);

        if (iteration  % 2 == 0) {
            MPI_Startall(2, send_req1);
            MPI_Startall(2, recv_req1);
        } else {
            MPI_Startall(2, send_req2);
            MPI_Startall(2, recv_req2);
        }

        /* пересчитываем все строки в полосе кроме верхней и нижней пока идёт пересылка */
        for (int i = 1; i < str_local - 1; ++i) {
            for (int j = 1; j < N - 1; ++j) {
                y_local[i * N + j] = (temp[(i + 1) * N + j] +
                                      temp[(i - 1) * N + j] +
                                      temp[i * N + (j + 1)] +
                                      temp[i * N + (j - 1)] +
                                      hh * f((nums_local + i) * h, j * h)) * q;
            }
        }

        if (iteration % 2 == 0) {
            MPI_Waitall(2, send_req1, MPI_STATUSES_IGNORE);
            MPI_Waitall(2, recv_req1, MPI_STATUSES_IGNORE);
        } else {
            MPI_Waitall(2, send_req2, MPI_STATUSES_IGNORE);
            MPI_Waitall(2, recv_req2, MPI_STATUSES_IGNORE);
        }

        /* пересчитываем верхние строки */
        if (ID != 0) {
            for (int j = 1; j < N - 1; ++j) {
                y_local[j] = (temp[N + j] + y_prev_low[j] +
                              temp[j + 1] + temp[j - 1] +
                              hh * f(nums_local * h, j * h)) * q;
            }
        }
        /* пересчитываем нижние строки */
        if (ID != NP - 1) {
            for (int j = 1; j < N - 1; ++j) {
                y_local[(str_local - 1) * N + j] = (y_next_top[j] +
                                                    temp[(str_local - 2) * N + j] +
                                                    temp[(str_local - 1) * N + (j + 1)] +
                                                    temp[(str_local - 1) * N + (j - 1)] +
                                                    hh * f((nums_local + (str_local - 1)) * h, j * h)) * q;
            }
        }
        norm_local = NOD((int)temp.size(), h, temp, y_local);

        MPI_Allreduce(&norm_local, &norm_err, 1, MPI_DOUBLE,
                      MPI_MAX, MPI_COMM_WORLD);

        if (norm_err < eps) {
            result.iterations = iteration;
            break;
        }
    }

    t1 += MPI_Wtime();

    for (int i = 0; i < NP; ++i) {
        str_per_proc[i] *= N;
        nums_start[i] *= N;
    }

    /* Собираем общее решение */
    MPI_Gatherv(y_local.data(), str_local * N, MPI_DOUBLE, y.data(),
                str_per_proc.data(), nums_start.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

//    MPI_Gather(y_local.data(), str_local * N, MPI_DOUBLE,
//               y.data(), str_local * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* Записываем результаты */
    if (ID == 0) {
        result.time = t1;
        result.norm_iter = norm_err;
        result.NP = NP;
        result.Y = y;
        result.Yp = temp;
        result.method_name =  "Method Yacobi (ISEND, IRECV)";
        result.NP = NP;
    }

    delete[] send_req1;
    delete[] recv_req1;
    delete[] send_req2;
    delete[] recv_req2;
}

void Method_Zeidel_P2P(MethodResultInfo& result, std::function<double(double, double)>&f, const double& K, const int& N,
                       const double& eps, const int& max_iterations) {


    int NP = 1; // Начальное значение кол-ва потоков программы
    int ID = 0; // Начальное значение ID (номера) исполняемого процесса
    MPI_Comm_size(MPI_COMM_WORLD, &NP); // Получаем кол-во исполняемых программ
    MPI_Comm_rank(MPI_COMM_WORLD, &ID); // Получаем номер исполняемой программы

    std::vector<int> str_per_proc; // Строка на процесс
    std::vector<int> nums_start;   // Индекс первой строки для потока

    int str_local; // Кол-во строк в процессе
    int nums_local; // Индекс первой строки, которой процесс будет обрабатывать
    double norm_local; // Норма
    double norm_err;

    /* Раздача работы процессам */
    Work_Distribution(NP, N, str_local, nums_local, str_per_proc, nums_start);

    std::vector<double> y_local(str_local * N); // Блок матрицы на процесс
    std::vector<double> y_next_top(N);             // Текущая приближение
    std::vector<double> y_prev_low(N);             // Предыдущее приближение

    int source_proc = ID ? ID - 1 : NP - 1; // myid == 0 = np - 1 // Процесс из которого отправляем
    int dest_proc = (ID != (NP - 1)) ? ID + 1 : 0; // у myid == np - 1 = 0 // Процесс в который отправляем

    int scount = (ID != (NP - 1)) ? N : 0; // у myid == np - 1 = 0 // Кол-во отправляемых данных
    int rcount = ID ? N : 0; // myid == 0 = 0                      // Кол-во получаемых данных

    double h = 1.0 / (double)(N - 1); // Шаг сетки
    double hh = h * h;                // Квадрат шага
    double kk = K * K;                // Квадрат коэф. К
    double q = 1.0 / (4.0 + kk * hh); // Выражение из схемы


    std::vector<double> y; // Общее решение
    if (ID == 0)
        y.resize(N * N);

    double t1 = -MPI_Wtime();

    std::vector<double> temp(y_local.size()); // Копия приближения
    for (int iteration = 1; iteration < max_iterations; iteration++) {

        std::swap(temp, y_local);

        // пересылаем нижние строки всеми процессами кроме последнего
        MPI_Send(temp.data() + (str_local - 1) * N, scount, MPI_DOUBLE,
                 dest_proc, 42, MPI_COMM_WORLD);
        MPI_Recv(y_prev_low.data(), rcount, MPI_DOUBLE,
                 source_proc, 42, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

        // пересылаем верхние строки всеми процессами кроме нулевого
        MPI_Send(temp.data(), rcount, MPI_DOUBLE,
                 source_proc, 46, MPI_COMM_WORLD);
        MPI_Recv(y_next_top.data(), scount, MPI_DOUBLE,
                 dest_proc, 46, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

        // внутренние строки (красные)
        for (int i = 1; i < str_local - 1; ++i)
            for (int j = ((i + 1) % 2 + 1); j < N - 1; j += 2)
                y_local[i * N + j] = (temp[(i + 1) * N + j] + temp[(i - 1) * N + j] +
                                      temp[i * N + (j + 1)] + temp[i * N + (j - 1)] +
                                      hh * f((nums_local + i) * h, j * h)) * q;

        // верхние строки (красные)
        if (ID != 0)
            for (int j = 2; j < N - 1; j += 2)
                y_local[j] = (temp[N + j] + y_prev_low[j] +
                              temp[j + 1] + temp[j - 1] +
                              hh * f(nums_local * h, j * h)) * q;

        // нижние строки (красные)
        if (ID != NP - 1)
            for (int j = 1 + str_local % 2; j < N - 1; j += 2)
                y_local[(str_local - 1) * N + j] = (y_next_top[j] +
                                                    temp[(str_local - 2) * N + j] + temp[(str_local - 1) * N +
                                                                                         (j + 1)] + temp[(str_local - 1) * N + (j - 1)] +
                                                    hh * f((nums_local + (str_local - 1)) * h, j * h)) * q;

        //MPI_Barrier;

        //пересылаем нижние строки всеми процессами кроме последнего
        MPI_Send(y_local.data() + (str_local - 1) * N, scount, MPI_DOUBLE,
                 dest_proc, 42, MPI_COMM_WORLD);
        MPI_Recv(y_prev_low.data(), rcount, MPI_DOUBLE,
                 source_proc, 42, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

        // пересылаем верхние строки всеми процессами кроме нулевого
        MPI_Send(y_local.data(), rcount, MPI_DOUBLE,
                 source_proc, 46, MPI_COMM_WORLD);
        MPI_Recv(y_next_top.data(), scount, MPI_DOUBLE,
                 dest_proc, 46, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);


        // внутренние строки (чёрные)
        for (int i = 1; i < str_local - 1; ++i)
            for (int j = (i % 2 + 1); j < N - 1; j += 2)
                y_local[i * N + j] = (y_local[(i + 1) * N + j] + y_local[(i - 1) * N + j] +
                                      y_local[i * N + (j + 1)] +
                                      y_local[i * N + (j - 1)] +
                                      hh * f((nums_local + i) * h, j * h)) * q;

        // верхние строки (чёрные)
        if (ID != 0)
            for (int j = 1; j < N - 1; j += 2)
                y_local[j] = (y_local[N + j] + y_prev_low[j] + y_local[j + 1] +
                              y_local[j - 1] + hh * f(nums_local * h, j * h)) * q;

        // нижние строки (чёрные)
        if (ID != NP - 1)
            for (int j = 1 + (str_local - 1) % 2; j < N - 1; j += 2)
                y_local[(str_local - 1) * N + j] = (y_next_top[j] +
                                                    y_local[(str_local - 2) * N + j] +
                                                    y_local[(str_local - 1) * N + (j + 1)] +
                                                    y_local[(str_local - 1) * N + (j - 1)] +
                                                    hh * f((nums_local + (str_local - 1)) * h, j * h)) * q;

        norm_local = NOD((int)temp.size(), h, temp, y_local);

        MPI_Allreduce(&norm_local, &norm_err, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        if (norm_err < eps) {
            result.iterations = iteration;
            break;
        }
    }

    t1 += MPI_Wtime();

    for (int i = 0; i < NP; ++i) {
        str_per_proc[i] *= N;
        nums_start[i] *= N;
    }

    /* Собираем общее решение */
    MPI_Gatherv(y_local.data(), str_local * N, MPI_DOUBLE, y.data(),
                str_per_proc.data(), nums_start.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* Записываем результаты */
    if (ID == 0) {
        result.time = t1;
        result.norm_iter = norm_err;
        result.NP = NP;
        result.Y = y;
        result.Yp = temp;
        result.method_name =  "Method Zeidel (SEND, RECV)";
        result.NP = NP;
    }
}

void Method_Zeidel_SIMULT(MethodResultInfo& result, std::function<double(double, double)>&f, const double& K, const int& N,
                          const double& eps, const int& max_iterations) {
    //cout << "\n\n The begin of programm by proc. " << myid << endl;

    int NP = 1; // Начальное значение кол-ва потоков программы
    int ID = 0; // Начальное значение ID (номера) исполняемого процесса
    MPI_Comm_size(MPI_COMM_WORLD, &NP); // Получаем кол-во исполняемых программ
    MPI_Comm_rank(MPI_COMM_WORLD, &ID); // Получаем номер исполняемой программы

    std::vector<int> str_per_proc; // Строка на процесс
    std::vector<int> nums_start;   // Индекс первой строки для потока

    int str_local; // Кол-во строк в процессе
    int nums_local; // Индекс первой строки, которой процесс будет обрабатывать
    double norm_local; // Норма
    double norm_err;

    /* Раздача работы процессам */
    Work_Distribution(NP, N, str_local, nums_local, str_per_proc, nums_start);

    std::vector<double> y_local(str_local * N); // Блок матрицы на процесс
    std::vector<double> y_next_top(N);             // Текущая приближение
    std::vector<double> y_prev_low(N);             // Предыдущее приближение

    int source_proc = ID ? ID - 1 : NP - 1; // myid == 0 = np - 1 // Процесс из которого отправляем
    int dest_proc = (ID != (NP - 1)) ? ID + 1 : 0; // у myid == np - 1 = 0 // Процесс в который отправляем

    int scount = (ID != (NP - 1)) ? N : 0; // у myid == np - 1 = 0 // Кол-во отправляемых данных
    int rcount = ID ? N : 0; // myid == 0 = 0                      // Кол-во получаемых данных

    double h = 1.0 / (double)(N - 1); // Шаг сетки
    double hh = h * h;                // Квадрат шага
    double kk = K * K;                // Квадрат коэф. К
    double q = 1.0 / (4.0 + kk * hh); // Выражение из схемы


    std::vector<double> y; // Общее решение
    if (ID == 0)
        y.resize(N * N);

    double t1 = -MPI_Wtime();

    std::vector<double> temp(y_local.size()); // Копия приближения
    for (int iteration = 1; iteration < max_iterations; iteration++) {

        std::swap(temp, y_local);

        // пересылаем нижние и верхние строки
        MPI_Sendrecv(temp.data() + (str_local - 1) * N, scount,
                     MPI_DOUBLE, dest_proc, 42,
                     y_prev_low.data(), rcount, MPI_DOUBLE,
                     source_proc, 42, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

        MPI_Sendrecv(temp.data(), rcount, MPI_DOUBLE,
                     source_proc, 46, y_next_top.data(),
                     scount, MPI_DOUBLE, dest_proc, 46,
                     MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

        // внутренние строки (красные)
        for (int i = 1; i < str_local - 1; ++i)
            for (int j = ((i + 1) % 2 + 1); j < N - 1; j += 2)
                y_local[i * N + j] = (temp[(i + 1) * N + j] + temp[(i - 1) * N + j] +
                                      temp[i * N + (j + 1)] + temp[i * N + (j - 1)] +
                                      hh * f((nums_local + i) * h, j * h)) * q;

        // верхние строки (красные)
        if (ID != 0)
            for (int j = 2; j < N - 1; j += 2)
                y_local[j] = (temp[N + j] + y_prev_low[j] + temp[j + 1] + temp[j - 1] + hh * f(nums_local * h, j * h)) * q;

        // нижние строки (красные)
        if (ID != NP - 1)
            for (int j = 1 + str_local % 2; j < N - 1; j += 2)
                y_local[(str_local - 1) * N + j] = (y_next_top[j] +
                                                    temp[(str_local - 2) * N + j] +
                                                    temp[(str_local - 1) * N + (j + 1)] +
                                                    temp[(str_local - 1) * N + (j - 1)] +
                                                    hh * f((nums_local + (str_local - 1)) * h, j * h)) * q;

        //MPI_Barrier;

        // пересылаем нижние и верхние строки
        MPI_Sendrecv(y_local.data() + (str_local - 1) * N, scount,
                     MPI_DOUBLE, dest_proc, 42, y_prev_low.data(),
                     rcount, MPI_DOUBLE, source_proc, 42,
                     MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

        MPI_Sendrecv(y_local.data(), rcount, MPI_DOUBLE,
                     source_proc, 46, y_next_top.data(),
                     scount, MPI_DOUBLE, dest_proc, 46,
                     MPI_COMM_WORLD, MPI_STATUSES_IGNORE);


        // внутренние строки (чёрные)
        for (int i = 1; i < str_local - 1; ++i)
            for (int j = (i % 2 + 1); j < N - 1; j += 2)
                y_local[i * N + j] = (y_local[(i + 1) * N + j] +
                                      y_local[(i - 1) * N + j] + y_local[i * N + (j + 1)] +
                                      y_local[i * N + (j - 1)] + hh * f((nums_local + i) * h, j * h)) * q;

        // верхние строки (чёрные)
        if (ID != 0)
            for (int j = 1; j < N - 1; j += 2)
                y_local[j] = (y_local[N + j] + y_prev_low[j] + y_local[j + 1] +
                              y_local[j - 1] + hh * f(nums_local * h, j * h)) * q;

        // нижние строки (чёрные)
        if (ID != NP - 1)
            for (int j = 1 + (str_local - 1) % 2; j < N - 1; j += 2)
                y_local[(str_local - 1) * N + j] = (y_next_top[j] +
                                                    y_local[(str_local - 2) * N + j] +
                                                    y_local[(str_local - 1) * N + (j + 1)] +
                                                    y_local[(str_local - 1) * N + (j - 1)] +
                                                    hh * f((nums_local + (str_local - 1)) * h, j * h)) * q;


        norm_local = NOD((int)temp.size(), h, temp, y_local);

        MPI_Allreduce(&norm_local, &norm_err, 1,
                      MPI_DOUBLE,MPI_MAX, MPI_COMM_WORLD);

        if (norm_err < eps) {
            result.iterations = iteration;
            break;
        }
    }

    t1 += MPI_Wtime();

    for (int i = 0; i < NP; ++i) {
        str_per_proc[i] *= N;
        nums_start[i] *= N;
    }

    /* Собираем общее решение */
    MPI_Gatherv(y_local.data(), str_local * N, MPI_DOUBLE, y.data(),
                str_per_proc.data(), nums_start.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* Записываем результаты */
    if (ID == 0) {
        result.time = t1;
        result.norm_iter = norm_err;
        result.NP = NP;
        result.Y = y;
        result.Yp = temp;
        result.method_name =  "Method Zeidel (SENDRECV)";
        result.NP = NP;
    }
}


void Method_Zeidel_NOBLOCK(MethodResultInfo& result, std::function<double(double, double)>&f, const double& K, const int& N,
                           const double& eps, const int& max_iterations) {


    int NP = 1; // Начальное значение кол-ва потоков программы
    int ID = 0; // Начальное значение ID (номера) исполняемого процесса
    MPI_Comm_size(MPI_COMM_WORLD, &NP); // Получаем кол-во исполняемых программ
    MPI_Comm_rank(MPI_COMM_WORLD, &ID); // Получаем номер исполняемой программы



    std::vector<int> str_per_proc; // Строка на процесс
    std::vector<int> nums_start;   // Индекс первой строки для потока

    int str_local; // Кол-во строк в процессе
    int nums_local; // Индекс первой строки, которой процесс будет обрабатывать
    double norm_local; // Норма
    double norm_err;

    MPI_Request* send_req1;
    MPI_Request* send_req2;
    MPI_Request* recv_req1;
    MPI_Request* recv_req2;

    int source_proc = ID ? ID - 1 : NP - 1; // myid == 0 = np - 1 // Процесс из которого отправляем
    int dest_proc = (ID != (NP - 1)) ? ID + 1 : 0; // у myid == np - 1 = 0 // Процесс в который отправляем

    int scount = (ID != (NP - 1)) ? N : 0; // у myid == np - 1 = 0 // Кол-во отправляемых данных
    int rcount = ID ? N : 0; // myid == 0 = 0                      // Кол-во получаемых данных

    /* Раздача работы процессам */
    Work_Distribution(NP, N, str_local, nums_local, str_per_proc, nums_start);


    double h = 1.0 / (double)(N - 1); // Шаг сетки
    double hh = h * h;                // Квадрат шага
    double kk = K * K;                // Квадрат коэф. К
    double q = 1.0 / (4.0 + kk * hh); // Выражение из схемы

    send_req1 = new MPI_Request[2], recv_req1 = new MPI_Request[2];
    send_req2 = new MPI_Request[2], recv_req2 = new MPI_Request[2];

    std::vector<double> y_local(str_local * N); // Блок матрицы на процесс
    std::vector<double> y_next_top(N);             // Текущая приближение
    std::vector<double> y_prev_low(N);             // Предыдущее приближение
    std::vector<double> temp(y_local.size()); // Копия приближения

    // пересылаем верхние и нижние строки temp
    MPI_Send_init(temp.data(), rcount, MPI_DOUBLE, source_proc, 0, MPI_COMM_WORLD, send_req1);
    MPI_Recv_init(y_prev_low.data(), rcount, MPI_DOUBLE, source_proc, 1, MPI_COMM_WORLD, recv_req1);

    MPI_Send_init(temp.data() + (str_local - 1) * N, scount, MPI_DOUBLE, dest_proc, 1, MPI_COMM_WORLD, send_req1 + 1);
    MPI_Recv_init(y_next_top.data(), scount, MPI_DOUBLE, dest_proc, 0, MPI_COMM_WORLD, recv_req1 + 1);

    // пересылаем верхние и нижние строки y_local
    MPI_Send_init(y_local.data(), rcount, MPI_DOUBLE, source_proc, 0, MPI_COMM_WORLD, send_req2);
    MPI_Recv_init(y_prev_low.data(), rcount, MPI_DOUBLE, source_proc, 1, MPI_COMM_WORLD, recv_req2);

    MPI_Send_init(y_local.data() + (str_local - 1) * N, scount, MPI_DOUBLE, dest_proc, 1, MPI_COMM_WORLD, send_req2 + 1);
    MPI_Recv_init(y_next_top.data(), scount, MPI_DOUBLE, dest_proc, 0, MPI_COMM_WORLD, recv_req2 + 1);

    std::vector<double> y; // Общее решение
    if (ID == 0) {
        y.resize(N * N);
    }

    double t1 = -MPI_Wtime();

    for (int iteration = 1; iteration < max_iterations; iteration++) {

        std::swap(temp, y_local);

        if (iteration % 2 == 0){
            MPI_Startall(2, send_req1);
            MPI_Startall(2, recv_req1);
        } else {
            MPI_Startall(2, send_req2);
            MPI_Startall(2, recv_req2);
        }

        // внутренние строки (красные)
        for (int i = 1; i < str_local - 1; ++i) {
            for (int j = ((i + 1) % 2 + 1); j < N - 1; j += 2) {
                y_local[i * N + j] = (temp[(i + 1) * N + j]
                                      + temp[(i - 1) * N + j] +
                                      temp[i * N + (j + 1)] +
                                      temp[i * N + (j - 1)] +
                                      hh * f((nums_local + i) * h, j * h)) * q;
            }
        }

        if (iteration % 2 == 0) {
            MPI_Waitall(2, send_req1, MPI_STATUSES_IGNORE);
            MPI_Waitall(2, recv_req1, MPI_STATUSES_IGNORE);
        } else {
            MPI_Waitall(2, send_req2, MPI_STATUSES_IGNORE);
            MPI_Waitall(2, recv_req2, MPI_STATUSES_IGNORE);
        }

        // верхние строки (красные)
        if (ID != 0){
            for (int j = 2; j < N - 1; j += 2) {
                y_local[j] = (temp[N + j] +
                              y_prev_low[j] +
                              temp[j + 1] +
                              temp[j - 1] +
                              hh * f(nums_local * h, j * h)) * q;
            }
        }

        // нижние строки (красные)
        if (ID != NP - 1) {
            for (int j = 1 + str_local % 2; j < N - 1; j += 2) {
                y_local[(str_local - 1) * N + j] = (y_next_top[j] +
                                                    temp[(str_local - 2) * N + j] +
                                                    temp[(str_local - 1) * N + (j + 1)] +
                                                    temp[(str_local - 1) * N + (j - 1)] +
                                                    hh * f((nums_local + (str_local - 1)) * h, j * h)) * q;
            }
        }

        if (iteration % 2 == 0) {
            MPI_Startall(2, send_req2);
            MPI_Startall(2, recv_req2);
        } else {
            MPI_Startall(2, send_req1);
            MPI_Startall(2, recv_req1);
        }

        // внутренние строки (чёрные)
        for (int i = 1; i < str_local - 1; ++i) {
            for (int j = (i % 2 + 1); j < N - 1; j += 2) {
                y_local[i * N + j] = (y_local[(i + 1) * N + j] +
                                      y_local[(i - 1) * N + j] +
                                      y_local[i * N + (j + 1)] +
                                      y_local[i * N + (j - 1)] +
                                      hh * f((nums_local + i) * h, j * h)) * q;
            }
        }

        if (iteration % 2 == 0) {
            MPI_Waitall(2, send_req2, MPI_STATUSES_IGNORE);
            MPI_Waitall(2, recv_req2, MPI_STATUSES_IGNORE);
        } else {
            MPI_Waitall(2, send_req1, MPI_STATUSES_IGNORE);
            MPI_Waitall(2, recv_req1, MPI_STATUSES_IGNORE);
        }

        // верхние строки (чёрные)
        if (ID != 0) {
            for (int j = 1; j < N - 1; j += 2) {
                y_local[j] = (y_local[N + j] + y_prev_low[j] +
                              y_local[j + 1] + y_local[j - 1] +
                              hh * f(nums_local * h, j * h)) * q;
            }
        }

        // нижние строки (чёрные)
        if (ID != NP - 1) {
            for (int j = 1 + (str_local - 1) % 2; j < N - 1; j += 2) {
                y_local[(str_local - 1) * N + j] = (y_next_top[j] +
                                                    y_local[(str_local - 2) * N + j] +
                                                    y_local[(str_local - 1) * N + (j + 1)] +
                                                    y_local[(str_local - 1) * N + (j - 1)] +
                                                    hh * f((nums_local + (str_local - 1)) * h, j * h)) * q;
            }
        }


        norm_local = NOD((int)temp.size(), h, temp, y_local);

        MPI_Allreduce(&norm_local, &norm_err, 1,
                      MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        if (norm_err < eps) {
            result.iterations = iteration;
            break;
        }
    }

    t1 += MPI_Wtime();

    for (int i = 0; i < NP; ++i) {
        str_per_proc[i] *= N;
        nums_start[i] *= N;
    }
    /* Собираем общее решение */
//    MPI_Gather(y_local.data(), str_local * N, MPI_DOUBLE,
//               y.data(), str_local * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(y_local.data(), str_local * N, MPI_DOUBLE,
                y.data(), str_per_proc.data(), nums_start.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* Записываем результаты */
    if (ID == 0) {
        result.time = t1;
        result.norm_iter = norm_err;
        result.NP = NP;
        result.Y = y;
        result.Yp = temp;
        result.method_name =  "Method Zeidel (ISEND, IRECV)";
        result.NP = NP;
    }

    delete[] send_req1;
    delete[] recv_req1;
    delete[] send_req2;
    delete[] recv_req2;
}
