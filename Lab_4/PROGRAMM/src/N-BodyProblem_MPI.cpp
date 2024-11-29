/*  РЕАЛИЗАЦИЯ ФУНКЦИЙ РЕШЕНИЯ ЗАДАЧИ N-ТЕЛ
 *
 * */

#include "../include/N-BodyProblem.h"

/* Операции над array и vector */
// умножение массива на число
std::array<double, 3>& operator*= (std::array<double, 3>& arr, double alpha) {

    for (int i = 0; i < 3; ++i)
        arr[i] *= alpha;
    return arr;
}

// сложение массивов
std::array<double, 3>& operator+= (std::array<double, 3>& a, const std::array<double, 3> b) {

    for (int i = 0; i < 3; ++i)
        a[i] += b[i];
    return a;
}

/* Операция домножения координат и скоростей на число */
std::vector<Body>& operator*= (std::vector<Body>& bodies, double alpha) {

    for (int i = 0; i < bodies.size(); ++i) {
        bodies[i].r *= alpha;
        bodies[i].v *= alpha;
    }
    return bodies;
}

/* Функция сложения двух массивов тел */
std::vector<Body>& operator+= (std::vector<Body>& a, const std::vector<Body>& b) {

    for (int i = 0; i < a.size(); ++i)
    {
        a[i].r += b[i].r;
        a[i].v += b[i].v;
    }
    return a;
}

/* Функция разности массивов */
// res = a - b
void array_diff(const std::array<double, 3>& a, const std::array<double, 3>& b, std::array<double, 3>& res) {

    for (int i = 0; i < 3; ++i)
        res[i] = a[i] - b[i];
}

/* Функция сложения? */
// res = a + alpha * b;
void vec_add_mul(const std::vector<Body>& a, const std::vector<Body>& b, double alpha, std::vector<Body>& res) {

    res = a;
    for (int i = 0; i < a.size(); ++i)
        for (int j = 0; j < 3; ++j)
        {
            res[i].r[j] += alpha * b[i].r[j];
            res[i].v[j] += alpha * b[i].v[j];
        }
}

/* Функция нормы R2 */
double norm(const std::array<double, 3>& arr) {

    double sum = 0.0;
    for (int i = 0; i < 3; ++i)
        sum += arr[i] * arr[i];

    return sqrt(sum);
}

/* Функция чтения из файла */
void read(const std::string& path, std::vector<Body>& bodies, int& N) {

    std::ifstream input(path);

    input >> N;
    bodies.resize(N);

    for (auto& body : bodies) {
        input >> body.m >> body.r[0] >> body.r[1] >> body.r[2] >> body.v[0] >> body.v[1] >> body.v[2];
    }
    input.close();
}

/* Функция записи в файл */
void write(const std::string& path, const Body& body, double t, int num) {

    std::ofstream output(path + std::to_string(num) + ").txt", std::ios::app);

    output << t << "\t" << std::fixed << std::setprecision(16) << body.r[0] << "\t" << body.r[1] << "\t" << body.r[2] << std::endl;

    output.close();
}

/* Функция очистки файла */
void clear_files(const std::string& path, int N) {

    for (int i = 1; i < N + 1; ++i) {
        std::ofstream output(path + std::to_string(i) + ").txt");
        //cout << "\n Clearing " << path + std::to_string(i) + ").txt" << "\n";
        output << "";
        output.close();
    }
}



/* ### Методы Рунге-Кутты ### */



/* Функция, описывающая систему диффуров, по формулам из файла ИК */
void f(std::vector<Body>& left, const std::vector<Body>& right, int start, int end) {

    std::array<double, 3> a{ 0,0,0 }, diff{ 0,0,0 };
    double tmp_norm;

    for (int i = 0; i < left.size(); ++i)
        left[i].r = right[i].v;

    for (int i = start; i < end; ++i)
    {
        for (int j = 0; j < left.size(); ++j)
        {
            array_diff(right[i].r, right[j].r, diff);
            tmp_norm = norm(diff);
            tmp_norm = tmp_norm * tmp_norm * tmp_norm;
            diff *= right[j].m / std::max(tmp_norm, eps * eps * eps);
            a += diff;
        }
        a *= -G;
        left[i].v = a;
        a.fill(0.0);
    }
}

// метод рунге-кутта 4-го порядка
void Runge_Kutta(const std::string& path, const std::vector<Body>& init, double tau,
                 double T, double& t, bool output) {

    int size = init.size();

    std::vector<Body> res(init);
    std::vector<Body> k1(size), k2(size), k3(size), k4(size), temp(size);

    double tau6 = tau / 6., tau3 = tau / 3., tau2 = tau / 2., t0 = 0.;

    if (output)
        for (int i = 0; i < size; ++i)
            write(path, res[i], t0, i + 1);

    t = -MPI_Wtime();

    while (t0 <= T)
    {
        f(k1, res, 0, size);

        vec_add_mul(res, k1, tau2, temp);
        f(k2, temp, 0, size);

        vec_add_mul(res, k2, tau2, temp);
        f(k3, temp, 0, size);

        vec_add_mul(res, k3, tau, temp);
        f(k4, temp, 0, size);

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
void Runge_Kutta_MPI(const std::string& path, const std::vector<Body>& init, double tau,
                     double T, double& t, bool output,
                     int np, int myid, int N, MPI_Datatype MPI_BODY_VPART) {

    int size = init.size(), count = N / np, start = myid * count, end = start + count;

    std::vector<Body> res(init);
    std::vector<Body> k1(size), k2(size), k3(size), k4(size), temp(size);

    double tau6 = tau / 6., tau3 = tau / 3., tau2 = tau / 2., t0 = 0.;

    if (output && myid == 0)
        for (int i = 0; i < size; ++i)
            write(path, res[i], t0, i + 1);

    if (myid == 0)
        t = -MPI_Wtime();

    // N тел, делим массив k1, k2, .. на np частей, каждый процесс считает свою часть массива
    // и пересылает с помощью Bcast остальным процессам
    while (t0 <= T)
    {
        f(k1, res, start, end);
        for (int i = 0; i < np; ++i)
            MPI_Bcast(k1.data() + i * count, count, MPI_BODY_VPART, i, MPI_COMM_WORLD);
        //MPI_Allgather(k1.data() + start, count, MPI_BODY_VPART, k1.data(), count, MPI_BODY_VPART, MPI_COMM_WORLD);

        vec_add_mul(res, k1, tau2, temp);
        f(k2, temp, start, end);
        for (int i = 0; i < np; ++i)
            MPI_Bcast(k2.data() + i * count, count, MPI_BODY_VPART, i, MPI_COMM_WORLD);
        //MPI_Allgather(k2.data() + start, count, MPI_BODY_VPART, k2.data(), count, MPI_BODY_VPART, MPI_COMM_WORLD);

        vec_add_mul(res, k2, tau2, temp);
        f(k3, temp, start, end);
        for (int i = 0; i < np; ++i)
            MPI_Bcast(k3.data() + i * count, count, MPI_BODY_VPART, i, MPI_COMM_WORLD);
        //MPI_Allgather(k3.data() + start, count, MPI_BODY_VPART, k3.data(), count, MPI_BODY_VPART, MPI_COMM_WORLD);

        vec_add_mul(res, k3, tau, temp);
        f(k4, temp, start, end);
        for (int i = 0; i < np; ++i)
            MPI_Bcast(k4.data() + i * count, count, MPI_BODY_VPART, i, MPI_COMM_WORLD);
        //MPI_Allgather(k4.data() + start, count, MPI_BODY_VPART, k4.data(), count, MPI_BODY_VPART, MPI_COMM_WORLD);

        k1 *= tau6; k2 *= tau3; k3 *= tau3; k4 *= tau6;
        res += k1; res += k2; res += k3; res += k4;

        t0 += tau;

        if ((output && myid == 0) && ((int)(round)(10000 * t0)) % 1000 == 0)
            //if (output && myid == 0)
            for (int i = 0; i < size; ++i)
                write(path, res[i], t0, i + 1);
    }

    if (myid == 0)
        t += MPI_Wtime();
}