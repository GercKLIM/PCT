
/* -------------------------------------------------------------- */
/* ### РЕАЛИЗАЦИЯ ФУНКЦИЙ РЕШЕНИЯ УРАВНЕНИЯ ГЕЛЬМГОЛЬЦА С MPI ### */
/* -------------------------------------------------------------- */


#include "../INCLUDE/Helmholtz_Solver.h"






MethodResultInfo Method_Jacobi_P2P(int argc, char** argv, std::function<double(double, double)>&f, const double& k, const int& N,
                                   const double& eps, const int& max_iterations){

    int NP = 1; // Начальное значение кол-ва потоков программы
    int ID = 0; // Начальное значение ID (номера) исполняемого процесса

    MPI_Init(&argc, &argv); // Запускаем MPI
    MPI_Comm_size(MPI_COMM_WORLD, &NP); // Получаем кол-во исполняемых программ
    MPI_Comm_rank(MPI_COMM_WORLD, &ID); // Получаем номер исполняемой программы

    std::vector<int> str_per_proc;
    std::vector<int> nums_start;

    int str_local;
    int nums_local;
    double norm_local;
    double norm_err;

    /* Раздача работы процессам */
    Work_Distribution(NP, N, str_local, nums_local, str_per_proc, nums_start);

    std::vector<double> y_local(str_local * N);
    std::vector<double> y_next_top(N);
    std::vector<double> y_prev_low(N);

    int source_proc = ID ? ID - 1 : NP - 1; // myid == 0 = np - 1
    int dest_proc = (ID != (NP - 1)) ? ID + 1 : 0; // у myid == np - 1 = 0

    int scount = (ID != (NP - 1)) ? N : 0; // у myid == np - 1 = 0
    int rcount = ID ? N : 0; // myid == 0 = 0

    double h = 1.0 / (double)(N - 1);
    double hh = h * h;
    double kk = k * k;
    double q = 1.0 / (4.0 + kk * hh);


    std::vector<double> y;
    if (ID == 0)
        y.resize(N * N);

    double t1 = -MPI_Wtime();

    int it = 0;
    bool flag = true;
    std::vector<double> temp(y_local.size());
    for (int iteration = 1; iteration < max_iterations; iteration++) {
        //cout << "\n it = " << it;

        std::swap(temp, y_local);

        // пересылаем нижние строки всеми процессами кроме последнего
        MPI_Send(temp.data() + (str_local - 1) * N, scount, MPI_DOUBLE, dest_proc, 42, MPI_COMM_WORLD);
        MPI_Recv(y_prev_low.data(), rcount, MPI_DOUBLE, source_proc, 42, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

        // пересылаем верхние строки всеми процессами кроме нулевого
        MPI_Send(temp.data(), rcount, MPI_DOUBLE, source_proc, 46, MPI_COMM_WORLD);
        MPI_Recv(y_next_top.data(), scount, MPI_DOUBLE, dest_proc, 46, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

        /* пересчитываем все строки в полосе кроме верхней и нижней */
        for (int i = 1; i < str_local - 1; ++i)
            for (int j = 1; j < N - 1; ++j)
                y_local[i * N + j] = (temp[(i + 1) * N + j] + temp[(i - 1) * N + j] + temp[i * N + (j + 1)]
                        + temp[i * N + (j - 1)] + h * h * f((nums_local + i) * h, j * h)) * q;

        /* пересчитываем верхние строки */
        if (ID != 0)
            for (int j = 1; j < N - 1; ++j)
                y_local[j] = (temp[N + j] + y_prev_low[j] + temp[j + 1] + temp[j - 1]
                        + hh * f(nums_local * h, j * h)) * q;

        /* пересчитываем нижние строки */
        if (ID != NP - 1)
            for (int j = 1; j < N - 1; ++j)
                y_local[(str_local - 1) * N + j] = (y_next_top[j] + temp[(str_local - 2) * N + j]
                        + temp[(str_local - 1) * N + (j + 1)] + temp[(str_local - 1) * N + (j - 1)]
                        + hh * f((nums_local + (str_local - 1)) * h, j * h)) * q;

        norm_local = NOD((int)temp.size(), h, temp, y_local);

        MPI_Allreduce(&norm_local, &norm_err, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        if (norm_err < eps) {
            break;
        }
    }

    t1 += MPI_Wtime();

    for (int i = 0; i < NP; ++i) {
        str_per_proc[i] *= N;
        nums_start[i] *= N;
    }

    //MPI_Gather(y_local.data(), str_local * n, MPI_DOUBLE, y.data(), str_local * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(y_local.data(), str_local * N, MPI_DOUBLE, y.data(),
                str_per_proc.data(), nums_start.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Finalize();

    MethodResultInfo result;
    result.iterations = it;
    result.time = t1;
    result.norm_iter = norm_err;
    result.NP = NP;
    result.Y = y;
    result.Yp = temp;

    return result;
}

MethodResultInfo Method_Jacobi_SIMULT(std::vector<double>& y, std::function<double(double, double)>&f, const double& k, const int& N,
                                      const double& eps, const int& max_num_iterations);

MethodResultInfo Method_Jacobi_NOBLOCK(std::vector<double>& y, std::function<double(double, double)>&f, const double& k, const int& N,
                                       const double& eps, const int& max_num_iterations);

MethodResultInfo Method_Zeidel_P2P(std::vector<double>& y, std::function<double(double, double)>&f, const double& k, const int& N,
                                   const double& eps, const int& max_num_iterations);

MethodResultInfo Method_Zeidel_SIMULT(std::vector<double>& y, std::function<double(double, double)>&f, const double& k, const int& N,
                                      const double& eps, const int& max_num_iterations);

MethodResultInfo Method_Zeidel_NOBLOCK(std::vector<double>& y, std::function<double(double, double)>&f, const double& k, const int& N,
                                       const double& eps, const int& max_num_iterations);