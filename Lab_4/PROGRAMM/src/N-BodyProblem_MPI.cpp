/*  РЕАЛИЗАЦИЯ ФУНКЦИЙ РЕШЕНИЯ ЗАДАЧИ N-ТЕЛ
 *
 * */

#include "../include/N-BodyProblem.h"



/* ### Методы Рунге-Кутты ### */



/* Функция, описывающая систему диффуров, по формулам из файла */
void f(std::vector<Body>& left, const std::vector<Body>& right, int start, int end, const double& EPS) {

    std::array<double, 3> a{ 0,0,0 }, diff{ 0,0,0 };
    double tmp_norm;

    for (int i = 0; i < left.size(); ++i) {
        left[i].r = right[i].v;
    }
    for (int i = start; i < end; ++i) {
        for (int j = 0; j < left.size(); ++j) {

            array_diff(right[i].r, right[j].r, diff);
            tmp_norm = norm(diff);
            tmp_norm = tmp_norm * tmp_norm * tmp_norm;
            diff *= right[j].m / std::max(tmp_norm, EPS * EPS * EPS);
            a += diff;
        }

        a *= -G;
        left[i].v = a;
        a.fill(0.0);
    }
}

/* Функция метода рунге-кутта 4-го порядка */
void Runge_Kutta(const std::string& path, const std::vector<Body>& init, double tau,
                 double T, double& t, const double& EPS, bool output) {

    int size = init.size();

    std::vector<Body> res(init);
    std::vector<Body> k1(size), k2(size), k3(size), k4(size), temp(size);

    double tau6 = tau / 6., tau3 = tau / 3., tau2 = tau / 2., t0 = 0.;

    if (output) {
        for (int i = 0; i < size; ++i) {
            write(path, res[i], t0, i + 1);
        }
    }

    t = -MPI_Wtime();

    while (t0 <= T) {

        f(k1, res, 0, size, EPS);

        vec_add_mul(res, k1, tau2, temp);
        f(k2, temp, 0, size, EPS);

        vec_add_mul(res, k2, tau2, temp);
        f(k3, temp, 0, size, EPS);

        vec_add_mul(res, k3, tau, temp);
        f(k4, temp, 0, size, EPS);

        k1 *= tau6; k2 *= tau3; k3 *= tau3; k4 *= tau6;
        res += k1; res += k2; res += k3; res += k4;

        t0 += tau;

        if (output)
            for (int i = 0; i < size; ++i)
                write(path, res[i], t0, i + 1);
    }

    t += MPI_Wtime();
}

// параллельный рунге-кутта
//void Runge_Kutta_MPI(const std::string& path, const std::vector<Body>& init, double tau,
//                     double T, double& t, const double& EPS, bool output,
//                     int NP, int ID, int N, MPI_Datatype MPI_BODY_VPART,const int& max_iteration) {
//
//    int size = init.size();
//    int count = N / NP;
//    int start = ID * count;
//    int end = start + count;
//
//    std::vector<Body> res(init);
//    std::vector<Body> k1(size), k2(size), k3(size), k4(size), temp(size);
//
//    double tau6 = tau / 6., tau3 = tau / 3., tau2 = tau / 2.;
//    double t0 = 0.;
//
//    if (output && ID== 0)
//        for (int i = 0; i < size; ++i)
//            write(path, res[i], t0, i + 1);
//
//    if (ID == 0)
//        t = -MPI_Wtime();
//
//    int iter = 0;
//
//    // N тел, делим массив k1, k2, .. на NP частей, каждый процесс считает свою часть массива
//    // и пересылает с помощью Bcast остальным процессам
//    while ((t0 <= T) and (iter < max_iteration)){
//        iter++;
//        f(k1, res, start, end, EPS);
////        for (int i = 0; i < NP; ++i) {
////            MPI_Bcast(k1.data() + i * count, count,
////                      MPI_BODY_VPART,i, MPI_COMM_WORLD);
////        }
//        MPI_Allgather(k1.data() + start, count, MPI_BODY_VPART, k1.data(),
//                      count, MPI_BODY_VPART, MPI_COMM_WORLD);
//
//        vec_add_mul(res, k1, tau2, temp);
//        f(k2, temp, start, end, EPS);
////        for (int i = 0; i < NP; ++i) {
////            MPI_Bcast(k2.data() + i * count, count,
////                      MPI_BODY_VPART, i, MPI_COMM_WORLD);
////        }
//        MPI_Allgather(k2.data() + start, count, MPI_BODY_VPART, k2.data(),
//                      count, MPI_BODY_VPART, MPI_COMM_WORLD);
//
//        vec_add_mul(res, k2, tau2, temp);
//        f(k3, temp, start, end, EPS);
//
////        for (int i = 0; i < NP; ++i) {
////            MPI_Bcast(k3.data() + i * count, count,
////                      MPI_BODY_VPART, i, MPI_COMM_WORLD);
////        }
//        MPI_Allgather(k3.data() + start, count, MPI_BODY_VPART, k3.data(),
//                      count, MPI_BODY_VPART, MPI_COMM_WORLD);
//
//        vec_add_mul(res, k3, tau, temp);
//        f(k4, temp, start, end, EPS);
////        for (int i = 0; i < NP; ++i) {
////            MPI_Bcast(k4.data() + i * count, count,
////                      MPI_BODY_VPART, i, MPI_COMM_WORLD);
////        }
//        MPI_Allgather(k4.data() + start, count, MPI_BODY_VPART, k4.data(), count, MPI_BODY_VPART, MPI_COMM_WORLD);
//
//        k1 *= tau6; k2 *= tau3; k3 *= tau3; k4 *= tau6;
//        res += k1; res += k2; res += k3; res += k4;
//
//        t0 += tau;
//
//        if ((output && ID== 0) /*&& ((int)(round)(10000 * t0)) % 1000 == 0*/) {
////            if (output && ID == 0)
//            for (int i = 0; i < size; ++i)
//                write(path, res[i], t0, i + 1);
//        }
//    }
//
//    if (ID== 0)
//        t += MPI_Wtime();
//}

// Параллельный метод Рунге-Кутта с использованием MPI_Allgatherv
void Runge_Kutta_MPI(const std::string& path, const std::vector<Body>& init, double tau,
                     double T, double& t, const double& EPS, bool output,
                     int NP, int ID, int N, MPI_Datatype MPI_BODY_VPART, const int& max_iteration) {

    int size = N; //init.size();  // Кол-во тел
    std::vector<int> counts(NP), displs(NP);

    // Определяем размеры и смещения для каждого процесса
    for (int i = 0; i < NP; ++i) {
        counts[i] = N / NP + (i < N % NP ? 1 : 0);                // Кол-во тел на данный поток
        displs[i] = (i == 0) ? 0 : displs[i - 1] + counts[i - 1]; // Номер первого тела, которое будет обрабатывать поток
    }

    int count = counts[ID];
    int start = displs[ID];
    int end = start + count;

    std::vector<Body> res(init);
    std::vector<Body> k1(size), k2(size), k3(size), k4(size), temp(size);

    double tau6 = tau / 6., tau3 = tau / 3., tau2 = tau / 2.;
    double t0 = 0.;

    if (output && ID == 0) {
        for (int i = 0; i < size; ++i) {
            write(path, res[i], t0, i + 1);
        }
    }

    if (ID == 0) {
        t = -MPI_Wtime();
    }

    int iter = 0;

    // Основной цикл интегрирования
    while ((t0 <= T) && (iter < max_iteration)) {
        iter++;

        // Вычисление k1
        f(k1, res, start, end, EPS);
        MPI_Allgatherv(k1.data() + start, count, MPI_BODY_VPART,
                       k1.data(), counts.data(), displs.data(), MPI_BODY_VPART, MPI_COMM_WORLD);

        // Вычисление k2
        vec_add_mul(res, k1, tau2, temp);
        f(k2, temp, start, end, EPS);
        MPI_Allgatherv(k2.data() + start, count, MPI_BODY_VPART,
                       k2.data(), counts.data(), displs.data(), MPI_BODY_VPART, MPI_COMM_WORLD);

        // Вычисление k3
        vec_add_mul(res, k2, tau2, temp);
        f(k3, temp, start, end, EPS);
        MPI_Allgatherv(k3.data() + start, count, MPI_BODY_VPART,
                       k3.data(), counts.data(), displs.data(), MPI_BODY_VPART, MPI_COMM_WORLD);

        // Вычисление k4
        vec_add_mul(res, k3, tau, temp);
        f(k4, temp, start, end, EPS);
        MPI_Allgatherv(k4.data() + start, count, MPI_BODY_VPART,
                       k4.data(), counts.data(), displs.data(), MPI_BODY_VPART, MPI_COMM_WORLD);

        // Обновление результата
        k1 *= tau6; k2 *= tau3; k3 *= tau3; k4 *= tau6;
        res += k1; res += k2; res += k3; res += k4;

        t0 += tau;

        // Запись результатов (если требуется)
        if (output && ID == 0) {
            for (int i = 0; i < size; ++i) {
                write(path, res[i], t0, i + 1);
            }
        }
    }

    if (ID == 0) {
        t += MPI_Wtime();
    }
}