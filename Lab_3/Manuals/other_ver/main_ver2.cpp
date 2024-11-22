#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include "mpi.h"

using std::vector;
using std::cout;
using std::endl;

const double pi = 3.141592653589793;

#define eps 1e-06

void red_black_Send_Recv(int myid, int np);
void red_black_SendRecv(int myid, int np);
void red_black_ISend_IRecv(int myid, int np);
void Jacobi_Send_Recv(int myid, int np);
void Jacobi_SendRecv(int myid, int np);
void Jacobi_ISend_IRecv(int myid, int np);

double ff(double x, double y);
double u_f(double x, double y);
void print(const vector<double>& vec, int m, int n);
void print(const vector<int>& vec);
double norm(const vector<double>& vec);
double norm_minus(const vector<double>& a, const vector<double>& b);
int t = 8*1024;
const int n = t;
double k = (double)n;
double h = 1.0 / (double)(n - 1);
double hh = h * h;
double kk = k * k;
double pp = pi * pi;
double q = 1.0 / (4.0 + kk * hh);

int main(int argc, char** argv)
{
    int myid, np;

    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &np);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    if (myid == 0)
        cout << "\n Parametres: n = " << n << ", k = " << k << ", np = " << np;

    if (myid == 0) cout << "\n\n %%%%%%% Red-black %%%%%%%";
    red_black_Send_Recv(myid, np);
    red_black_SendRecv(myid, np);
    red_black_ISend_IRecv(myid, np);

    if (myid == 0) cout << "\n\n %%%%%%%% Jacobi %%%%%%%%%";
    Jacobi_Send_Recv(myid, np);
    Jacobi_SendRecv(myid, np);
    Jacobi_ISend_IRecv(myid, np);

    MPI_Finalize();

    //cout << "\n\n The end of programm by proc. " << myid << endl;
    return 0;
}

double ff(double x, double y)
{
    return 2.0 * sin(pi * y) + kk * (1.0 - x) * x * sin(pi * y) + pp * (1.0 - x) * x * sin(pi * y);
}

double u_f(double x, double y)
{
    return (1.0 - x) * x * sin(pi * y);
}

//void str_split(int myid, int np, int& str_local, int& nums_local)
void str_split(int myid, int np, int& str_local, int& nums_local, vector<int>& str_per_proc, vector<int>& nums_start)
{
    //vector<int> str_per_proc, nums_start;

    //if (myid == 0)
    //{
    str_per_proc.resize(np, n / np);
    nums_start.resize(np, 0);

    for (int i = 0; i < n % np; ++i)
        ++str_per_proc[i];
    //str_per_proc[np - 1] += n % np;
    for (int i = 1; i < np; ++i)
        nums_start[i] = nums_start[i - 1] + str_per_proc[i - 1];
    //}

    //MPI_Bcast(str_per_proc.data(), np, MPI_INT, 0, MPI_COMM_WORLD);
    //MPI_Bcast(nums_start.data(), np, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Scatter(str_per_proc.data(), 1, MPI_INT, &str_local, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(nums_start.data(), 1, MPI_INT, &nums_local, 1, MPI_INT, 0, MPI_COMM_WORLD);
}

void red_black_Send_Recv(int myid, int np)
{
    //cout << "\n\n The begin of programm by proc. " << myid << endl;

    vector<int> str_per_proc, nums_start;
    int str_local, nums_local;
    double norm_local, norm_err;

    str_split(myid, np, str_local, nums_local, str_per_proc, nums_start);

    //cout << "\n Proc. " << myid << " has " << str_local << " strings starting at " << nums_local << ".\n";

    //if (myid == 0)
    //{
    //    cout << "\n\n";
    //    print(str_per_proc);
    //    cout << "\n";
    //    print(nums_start);
    //}

    vector<double> y_local(str_local * n);
    vector<double> y_next_top(n);
    vector<double> y_prev_low(n);

    int source_proc = myid ? myid - 1 : np - 1; // myid == 0 = np - 1
    int dest_proc = (myid != (np - 1)) ? myid + 1 : 0; // у myid == np - 1 = 0

    int scount = (myid != (np - 1)) ? n : 0; // у myid == np - 1 = 0
    int rcount = myid ? n : 0; // myid == 0 = 0

    vector<double> y;
    if (myid == 0)
        y.resize(n * n);

    double t1 = -MPI_Wtime();

    int it = 0;
    bool flag = true;
    vector<double> temp(y_local.size());
    while (flag)
    {
        it++;
        //cout << "\n it = " << it;

        std::swap(temp, y_local);

        //y_local = temp;

        // пересылаем нижние строки всеми процессами кроме последнего 
        MPI_Send(temp.data() + (str_local - 1) * n, scount, MPI_DOUBLE, dest_proc, 42, MPI_COMM_WORLD);
        MPI_Recv(y_prev_low.data(), rcount, MPI_DOUBLE, source_proc, 42, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

        // пересылаем верхние строки всеми процессами кроме нулевого
        MPI_Send(temp.data(), rcount, MPI_DOUBLE, source_proc, 46, MPI_COMM_WORLD);
        MPI_Recv(y_next_top.data(), scount, MPI_DOUBLE, dest_proc, 46, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

        // внутренние строки (красные)
        for (int i = 1; i < str_local - 1; ++i)
            for (int j = ((i + 1) % 2 + 1); j < n - 1; j += 2)
                y_local[i * n + j] = (temp[(i + 1) * n + j] + temp[(i - 1) * n + j] + temp[i * n + (j + 1)] + temp[i * n + (j - 1)] + hh * ff((nums_local + i) * h, j * h)) * q;

        // верхние строки (красные)
        if (myid != 0)
            for (int j = 2; j < n - 1; j += 2)
                y_local[j] = (temp[n + j] + y_prev_low[j] + temp[j + 1] + temp[j - 1] + hh * ff(nums_local * h, j * h)) * q;

        // нижние строки (красные)
        if (myid != np - 1)
            for (int j = 1 + str_local % 2; j < n - 1; j += 2)
                y_local[(str_local - 1) * n + j] = (y_next_top[j] + temp[(str_local - 2) * n + j] + temp[(str_local - 1) * n + (j + 1)] + temp[(str_local - 1) * n + (j - 1)] + hh * ff((nums_local + (str_local - 1)) * h, j * h)) * q;

        //MPI_Barrier;

        //пересылаем нижние строки всеми процессами кроме последнего 
        MPI_Send(y_local.data() + (str_local - 1) * n, scount, MPI_DOUBLE, dest_proc, 42, MPI_COMM_WORLD);
        MPI_Recv(y_prev_low.data(), rcount, MPI_DOUBLE, source_proc, 42, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

        // пересылаем верхние строки всеми процессами кроме нулевого
        MPI_Send(y_local.data(), rcount, MPI_DOUBLE, source_proc, 46, MPI_COMM_WORLD);
        MPI_Recv(y_next_top.data(), scount, MPI_DOUBLE, dest_proc, 46, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);


        // внутренние строки (чёрные)
        for (int i = 1; i < str_local - 1; ++i)
            for (int j = (i % 2 + 1); j < n - 1; j += 2)
                y_local[i * n + j] = (y_local[(i + 1) * n + j] + y_local[(i - 1) * n + j] + y_local[i * n + (j + 1)] + y_local[i * n + (j - 1)] + hh * ff((nums_local + i) * h, j * h)) * q;

        // верхние строки (чёрные)
        if (myid != 0)
            for (int j = 1; j < n - 1; j += 2)
                y_local[j] = (y_local[n + j] + y_prev_low[j] + y_local[j + 1] + y_local[j - 1] + hh * ff(nums_local * h, j * h)) * q;

        // нижние строки (чёрные)
        if (myid != np - 1)
            for (int j = 1 + (str_local - 1) % 2; j < n - 1; j += 2)
                y_local[(str_local - 1) * n + j] = (y_next_top[j] + y_local[(str_local - 2) * n + j] + y_local[(str_local - 1) * n + (j + 1)] + y_local[(str_local - 1) * n + (j - 1)] + hh * ff((nums_local + (str_local - 1)) * h, j * h)) * q;

        norm_local = norm_minus(temp, y_local);

        MPI_Allreduce(&norm_local, &norm_err, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        //if (myid == 0)
        //    cout << "\n norm on " << it << "-th iteration: " << norm_err;

        if (norm_err < eps)
            flag = false;
    }

    t1 += MPI_Wtime();

    for (int i = 0; i < np; ++i)
    {
        str_per_proc[i] *= n;
        nums_start[i] *= n;
    }

    //MPI_Gather(y_local.data(), str_local * n, MPI_DOUBLE, y.data(), str_local * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(y_local.data(), str_local * n, MPI_DOUBLE, y.data(), str_per_proc.data(), nums_start.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (myid == 0)
    {
        // точное решение
        vector<double> u(n * n);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                u[i * n + j] = u_f(i * h, j * h);

        cout << "\n\n 1. Send + Recv";
        cout << "\n\t norm: " << norm_minus(y, u);
        cout << "\n\t iterations: " << it;
        printf("\n\t time: %.4f", t1);
    }
}

void red_black_SendRecv(int myid, int np)
{
    //cout << "\n\n The begin of programm by proc. " << myid << endl;

    vector<int> str_per_proc, nums_start;
    int str_local, nums_local;
    double norm_local, norm_err;

    //str_split(myid, np, str_local, nums_local);
    str_split(myid, np, str_local, nums_local, str_per_proc, nums_start);

    vector<double> y_local(str_local * n);
    vector<double> y_next_top(n);
    vector<double> y_prev_low(n);

    vector<double> y;
    if (myid == 0)
        y.resize(n * n);

    double t1 = -MPI_Wtime();

    int source_proc = myid ? myid - 1 : np - 1;
    int dest_proc = (myid != (np - 1)) ? myid + 1 : 0;

    int scount = (myid != (np - 1)) ? n : 0;
    int rcount = myid ? n : 0;

    int it = 0;
    bool flag = true;
    vector<double> temp(y_local.size());
    while (flag)
    {
        it++;
        //cout << "\n it = " << it;

        std::swap(temp, y_local);

        //y_local = temp;

        // пересылаем нижние и верхние строки
        MPI_Sendrecv(temp.data() + (str_local - 1) * n, scount, MPI_DOUBLE, dest_proc, 42, y_prev_low.data(), rcount, MPI_DOUBLE, source_proc, 42, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
        MPI_Sendrecv(temp.data(), rcount, MPI_DOUBLE, source_proc, 46, y_next_top.data(), scount, MPI_DOUBLE, dest_proc, 46, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

        // внутренние строки (красные)
        for (int i = 1; i < str_local - 1; ++i)
            for (int j = ((i + 1) % 2 + 1); j < n - 1; j += 2)
                y_local[i * n + j] = (temp[(i + 1) * n + j] + temp[(i - 1) * n + j] + temp[i * n + (j + 1)] + temp[i * n + (j - 1)] + hh * ff((nums_local + i) * h, j * h)) * q;

        // верхние строки (красные)
        if (myid != 0)
            for (int j = 2; j < n - 1; j += 2)
                y_local[j] = (temp[n + j] + y_prev_low[j] + temp[j + 1] + temp[j - 1] + hh * ff(nums_local * h, j * h)) * q;

        // нижние строки (красные)
        if (myid != np - 1)
            for (int j = 1 + str_local % 2; j < n - 1; j += 2)
                y_local[(str_local - 1) * n + j] = (y_next_top[j] + temp[(str_local - 2) * n + j] + temp[(str_local - 1) * n + (j + 1)] + temp[(str_local - 1) * n + (j - 1)] + hh * ff((nums_local + (str_local - 1)) * h, j * h)) * q;

        //MPI_Barrier;

        // пересылаем нижние и верхние строки
        MPI_Sendrecv(y_local.data() + (str_local - 1) * n, scount, MPI_DOUBLE, dest_proc, 42, y_prev_low.data(), rcount, MPI_DOUBLE, source_proc, 42, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
        MPI_Sendrecv(y_local.data(), rcount, MPI_DOUBLE, source_proc, 46, y_next_top.data(), scount, MPI_DOUBLE, dest_proc, 46, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);


        // внутренние строки (чёрные)
        for (int i = 1; i < str_local - 1; ++i)
            for (int j = (i % 2 + 1); j < n - 1; j += 2)
                y_local[i * n + j] = (y_local[(i + 1) * n + j] + y_local[(i - 1) * n + j] + y_local[i * n + (j + 1)] + y_local[i * n + (j - 1)] + hh * ff((nums_local + i) * h, j * h)) * q;

        // верхние строки (чёрные)
        if (myid != 0)
            for (int j = 1; j < n - 1; j += 2)
                y_local[j] = (y_local[n + j] + y_prev_low[j] + y_local[j + 1] + y_local[j - 1] + hh * ff(nums_local * h, j * h)) * q;

        // нижние строки (чёрные)
        if (myid != np - 1)
            for (int j = 1 + (str_local - 1) % 2; j < n - 1; j += 2)
                y_local[(str_local - 1) * n + j] = (y_next_top[j] + y_local[(str_local - 2) * n + j] + y_local[(str_local - 1) * n + (j + 1)] + y_local[(str_local - 1) * n + (j - 1)] + hh * ff((nums_local + (str_local - 1)) * h, j * h)) * q;


        norm_local = norm_minus(temp, y_local);

        MPI_Allreduce(&norm_local, &norm_err, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        if (norm_err < eps)
            flag = false;
    }

    t1 += MPI_Wtime();

    for (int i = 0; i < np; ++i)
    {
        str_per_proc[i] *= n;
        nums_start[i] *= n;
    }

    //MPI_Gather(y_local.data(), str_local * n, MPI_DOUBLE, y.data(), str_local * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(y_local.data(), str_local * n, MPI_DOUBLE, y.data(), str_per_proc.data(), nums_start.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (myid == 0)
    {
        // точное решение
        vector<double> u(n * n);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                u[i * n + j] = u_f(i * h, j * h);

        cout << "\n\n 2. SendRecv";
        cout << "\n\t norm: " << norm_minus(y, u);
        cout << "\n\t iterations: " << it;
        printf("\n\t time: %.4f", t1);
    }
}

void red_black_ISend_IRecv(int myid, int np)
{
    //cout << "\n\n The begin of programm by proc. " << myid << endl;

    vector<int> str_per_proc, nums_start;
    int str_local, nums_local;
    double norm_local, norm_err;

    MPI_Request* send_req1;
    MPI_Request* send_req2;
    MPI_Request* recv_req1;
    MPI_Request* recv_req2;

    int source_proc = myid ? myid - 1 : np - 1;
    int dest_proc = (myid != (np - 1)) ? myid + 1 : 0;

    int scount = (myid != (np - 1)) ? n : 0;
    int rcount = myid ? n : 0;

    //str_split(myid, np, str_local, nums_local);
    str_split(myid, np, str_local, nums_local, str_per_proc, nums_start);

    send_req1 = new MPI_Request[2], recv_req1 = new MPI_Request[2];
    send_req2 = new MPI_Request[2], recv_req2 = new MPI_Request[2];

    vector<double> y_local(str_local * n);
    vector<double> y_next_top(n);
    vector<double> y_prev_low(n);
    vector<double> temp(y_local.size());

    // пересылаем верхние и нижние строки temp
    MPI_Send_init(temp.data(), rcount, MPI_DOUBLE, source_proc, 0, MPI_COMM_WORLD, send_req1);
    MPI_Recv_init(y_prev_low.data(), rcount, MPI_DOUBLE, source_proc, 1, MPI_COMM_WORLD, recv_req1);

    MPI_Send_init(temp.data() + (str_local - 1) * n, scount, MPI_DOUBLE, dest_proc, 1, MPI_COMM_WORLD, send_req1 + 1);
    MPI_Recv_init(y_next_top.data(), scount, MPI_DOUBLE, dest_proc, 0, MPI_COMM_WORLD, recv_req1 + 1);

    // пересылаем верхние и нижние строки y_local
    MPI_Send_init(y_local.data(), rcount, MPI_DOUBLE, source_proc, 0, MPI_COMM_WORLD, send_req2);
    MPI_Recv_init(y_prev_low.data(), rcount, MPI_DOUBLE, source_proc, 1, MPI_COMM_WORLD, recv_req2);

    MPI_Send_init(y_local.data() + (str_local - 1) * n, scount, MPI_DOUBLE, dest_proc, 1, MPI_COMM_WORLD, send_req2 + 1);
    MPI_Recv_init(y_next_top.data(), scount, MPI_DOUBLE, dest_proc, 0, MPI_COMM_WORLD, recv_req2 + 1);

    vector<double> y;
    if (myid == 0)
        y.resize(n * n);

    double t1 = -MPI_Wtime();

    int it = 0;
    bool flag = true;
    while (flag)
    {
        it++;
        //cout << "\n it = " << it;

        std::swap(temp, y_local);

        //y_local = temp;

        if (it % 2 == 0)
        {
            MPI_Startall(2, send_req1);
            MPI_Startall(2, recv_req1);
        }
        else
        {
            MPI_Startall(2, send_req2);
            MPI_Startall(2, recv_req2);
        }

        // внутренние строки (красные)
        for (int i = 1; i < str_local - 1; ++i)
            for (int j = ((i + 1) % 2 + 1); j < n - 1; j += 2)
                y_local[i * n + j] = (temp[(i + 1) * n + j] + temp[(i - 1) * n + j] + temp[i * n + (j + 1)] + temp[i * n + (j - 1)] + hh * ff((nums_local + i) * h, j * h)) * q;

        if (it % 2 == 0)
        {
            MPI_Waitall(2, send_req1, MPI_STATUSES_IGNORE);
            MPI_Waitall(2, recv_req1, MPI_STATUSES_IGNORE);
        }
        else
        {
            MPI_Waitall(2, send_req2, MPI_STATUSES_IGNORE);
            MPI_Waitall(2, recv_req2, MPI_STATUSES_IGNORE);
        }

        // верхние строки (красные)
        if (myid != 0)
            for (int j = 2; j < n - 1; j += 2)
                y_local[j] = (temp[n + j] + y_prev_low[j] + temp[j + 1] + temp[j - 1] + hh * ff(nums_local * h, j * h)) * q;

        // нижние строки (красные)
        if (myid != np - 1)
            for (int j = 1 + str_local % 2; j < n - 1; j += 2)
                y_local[(str_local - 1) * n + j] = (y_next_top[j] + temp[(str_local - 2) * n + j] + temp[(str_local - 1) * n + (j + 1)] + temp[(str_local - 1) * n + (j - 1)] + hh * ff((nums_local + (str_local - 1)) * h, j * h)) * q;

        //MPI_Barrier;

        if (it % 2 == 0)
        {
            MPI_Startall(2, send_req2);
            MPI_Startall(2, recv_req2);
        }
        else
        {
            MPI_Startall(2, send_req1);
            MPI_Startall(2, recv_req1);
        }

        // внутренние строки (чёрные)
        for (int i = 1; i < str_local - 1; ++i)
            for (int j = (i % 2 + 1); j < n - 1; j += 2)
                y_local[i * n + j] = (y_local[(i + 1) * n + j] + y_local[(i - 1) * n + j] + y_local[i * n + (j + 1)] + y_local[i * n + (j - 1)] + hh * ff((nums_local + i) * h, j * h)) * q;

        if (it % 2 == 0)
        {
            MPI_Waitall(2, send_req2, MPI_STATUSES_IGNORE);
            MPI_Waitall(2, recv_req2, MPI_STATUSES_IGNORE);
        }
        else
        {
            MPI_Waitall(2, send_req1, MPI_STATUSES_IGNORE);
            MPI_Waitall(2, recv_req1, MPI_STATUSES_IGNORE);
        }

        // верхние строки (чёрные)
        if (myid != 0)
            for (int j = 1; j < n - 1; j += 2)
                y_local[j] = (y_local[n + j] + y_prev_low[j] + y_local[j + 1] + y_local[j - 1] + hh * ff(nums_local * h, j * h)) * q;

        // нижние строки (чёрные)
        if (myid != np - 1)
            for (int j = 1 + (str_local - 1) % 2; j < n - 1; j += 2)
                y_local[(str_local - 1) * n + j] = (y_next_top[j] + y_local[(str_local - 2) * n + j] + y_local[(str_local - 1) * n + (j + 1)] + y_local[(str_local - 1) * n + (j - 1)] + hh * ff((nums_local + (str_local - 1)) * h, j * h)) * q;


        norm_local = norm_minus(temp, y_local);

        MPI_Allreduce(&norm_local, &norm_err, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        if (norm_err < eps)
            flag = false;
    }

    t1 += MPI_Wtime();

    for (int i = 0; i < np; ++i)
    {
        str_per_proc[i] *= n;
        nums_start[i] *= n;
    }

    //MPI_Gather(y_local.data(), str_local * n, MPI_DOUBLE, y.data(), str_local * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(y_local.data(), str_local * n, MPI_DOUBLE, y.data(), str_per_proc.data(), nums_start.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (myid == 0)
    {
        // точное решение
        vector<double> u(n * n);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                u[i * n + j] = u_f(i * h, j * h);

        cout << "\n\n 3. Isend + Irecv";
        cout << "\n\t norm: " << norm_minus(y, u);
        cout << "\n\t iterations: " << it;
        printf("\n\t time: %.4f", t1);
    }

    delete[] send_req1;
    delete[] recv_req1;
    delete[] send_req2;
    delete[] recv_req2;
}

void Jacobi_Send_Recv(int myid, int np)
{
    vector<int> str_per_proc, nums_start;
    int str_local, nums_local;
    double norm_local, norm_err;

    str_split(myid, np, str_local, nums_local, str_per_proc, nums_start);

    vector<double> y_local(str_local * n);
    vector<double> y_next_top(n);
    vector<double> y_prev_low(n);

    int source_proc = myid ? myid - 1 : np - 1; // myid == 0 = np - 1
    int dest_proc = (myid != (np - 1)) ? myid + 1 : 0; // у myid == np - 1 = 0

    int scount = (myid != (np - 1)) ? n : 0; // у myid == np - 1 = 0
    int rcount = myid ? n : 0; // myid == 0 = 0

    vector<double> y;
    if (myid == 0)
        y.resize(n * n);

    double t1 = -MPI_Wtime();

    int it = 0;
    bool flag = true;
    vector<double> temp(y_local.size());
    while (flag)
    {
        it++;
        //cout << "\n it = " << it;

        std::swap(temp, y_local);

        // пересылаем нижние строки всеми процессами кроме последнего 
        MPI_Send(temp.data() + (str_local - 1) * n, scount, MPI_DOUBLE, dest_proc, 42, MPI_COMM_WORLD);
        MPI_Recv(y_prev_low.data(), rcount, MPI_DOUBLE, source_proc, 42, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

        // пересылаем верхние строки всеми процессами кроме нулевого
        MPI_Send(temp.data(), rcount, MPI_DOUBLE, source_proc, 46, MPI_COMM_WORLD);
        MPI_Recv(y_next_top.data(), scount, MPI_DOUBLE, dest_proc, 46, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

        /* пересчитываем все строки в полосе кроме верхней и нижней */
        for (int i = 1; i < str_local - 1; ++i)
            for (int j = 1; j < n - 1; ++j)
                y_local[i * n + j] = (temp[(i + 1) * n + j] + temp[(i - 1) * n + j] + temp[i * n + (j + 1)] + temp[i * n + (j - 1)] + hh * ff((nums_local + i) * h, j * h)) * q;

        /* пересчитываем верхние строки */
        if (myid != 0)
            for (int j = 1; j < n - 1; ++j)
                y_local[j] = (temp[n + j] + y_prev_low[j] + temp[j + 1] + temp[j - 1] + hh * ff(nums_local * h, j * h)) * q;

        /* пересчитываем нижние строки */
        if (myid != np - 1)
            for (int j = 1; j < n - 1; ++j)
                y_local[(str_local - 1) * n + j] = (y_next_top[j] + temp[(str_local - 2) * n + j] + temp[(str_local - 1) * n + (j + 1)] + temp[(str_local - 1) * n + (j - 1)] + hh * ff((nums_local + (str_local - 1)) * h, j * h)) * q;

        norm_local = norm_minus(temp, y_local);

        MPI_Allreduce(&norm_local, &norm_err, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        if (norm_err < eps)
            flag = false;
    }

    t1 += MPI_Wtime();

    for (int i = 0; i < np; ++i)
    {
        str_per_proc[i] *= n;
        nums_start[i] *= n;
    }

    //MPI_Gather(y_local.data(), str_local * n, MPI_DOUBLE, y.data(), str_local * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(y_local.data(), str_local * n, MPI_DOUBLE, y.data(), str_per_proc.data(), nums_start.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (myid == 0)
    {
        // точное решение
        vector<double> u(n * n);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                u[i * n + j] = u_f(i * h, j * h);

        cout << "\n\n 1. Send + Recv";
        cout << "\n\t norm: " << norm_minus(y, u);
        cout << "\n\t iterations: " << it;
        printf("\n\t time: %.4f", t1);
    }
}

void Jacobi_SendRecv(int myid, int np)
{
    //cout << "\n\n The begin of programm by proc. " << myid << endl;

    vector<int> str_per_proc, nums_start;
    int str_local, nums_local;
    double norm_local, norm_err;

    //str_split(myid, np, str_local, nums_local);
    str_split(myid, np, str_local, nums_local, str_per_proc, nums_start);

    vector<double> y_local(str_local * n);
    vector<double> y_next_top(n);
    vector<double> y_prev_low(n);

    vector<double> y;
    if (myid == 0)
        y.resize(n * n);

    double t1 = -MPI_Wtime();

    int source_proc = myid ? myid - 1 : np - 1;
    int dest_proc = (myid != (np - 1)) ? myid + 1 : 0;

    int scount = (myid != (np - 1)) ? n : 0;
    int rcount = myid ? n : 0;

    int it = 0;
    bool flag = true;
    vector<double> temp(y_local.size());
    while (flag)
    {
        it++;
        //cout << "\n it = " << it;

        std::swap(temp, y_local);

        // пересылаем нижние и верхние строки
        MPI_Sendrecv(temp.data() + (str_local - 1) * n, scount, MPI_DOUBLE, dest_proc, 42, y_prev_low.data(), rcount, MPI_DOUBLE, source_proc, 42, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
        MPI_Sendrecv(temp.data(), rcount, MPI_DOUBLE, source_proc, 46, y_next_top.data(), scount, MPI_DOUBLE, dest_proc, 46, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

        /* пересчитываем все строки в полосе кроме верхней и нижней */
        for (int i = 1; i < str_local - 1; ++i)
            for (int j = 1; j < n - 1; ++j)
                y_local[i * n + j] = (temp[(i + 1) * n + j] + temp[(i - 1) * n + j] + temp[i * n + (j + 1)] + temp[i * n + (j - 1)] + hh * ff((nums_local + i) * h, j * h)) * q;

        /* пересчитываем верхние строки */
        if (myid != 0)
            for (int j = 1; j < n - 1; ++j)
                y_local[j] = (temp[n + j] + y_prev_low[j] + temp[j + 1] + temp[j - 1] + hh * ff(nums_local * h, j * h)) * q;

        /* пересчитываем нижние строки */
        if (myid != np - 1)
            for (int j = 1; j < n - 1; ++j)
                y_local[(str_local - 1) * n + j] = (y_next_top[j] + temp[(str_local - 2) * n + j] + temp[(str_local - 1) * n + (j + 1)] + temp[(str_local - 1) * n + (j - 1)] + hh * ff((nums_local + (str_local - 1)) * h, j * h)) * q;

        norm_local = norm_minus(temp, y_local);

        MPI_Allreduce(&norm_local, &norm_err, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        if (norm_err < eps)
            flag = false;
    }

    t1 += MPI_Wtime();

    for (int i = 0; i < np; ++i)
    {
        str_per_proc[i] *= n;
        nums_start[i] *= n;
    }

    //MPI_Gather(y_local.data(), str_local * n, MPI_DOUBLE, y.data(), str_local * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(y_local.data(), str_local * n, MPI_DOUBLE, y.data(), str_per_proc.data(), nums_start.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (myid == 0)
    {
        // точное решение
        vector<double> u(n * n);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                u[i * n + j] = u_f(i * h, j * h);

        cout << "\n\n 2. SendRecv";
        cout << "\n\t norm: " << norm_minus(y, u);
        cout << "\n\t iterations: " << it;
        printf("\n\t time: %.4f", t1);
    }
}

void Jacobi_ISend_IRecv(int myid, int np)
{
    //cout << "\n\n The begin of programm by proc. " << myid << endl;

    MPI_Request* send_req1;
    MPI_Request* send_req2;
    MPI_Request* recv_req1;
    MPI_Request* recv_req2;

    vector<int> str_per_proc, nums_start;
    int str_local, nums_local;
    double norm_local, norm_err;

    str_split(myid, np, str_local, nums_local, str_per_proc, nums_start);

    send_req1 = new MPI_Request[2], recv_req1 = new MPI_Request[2];
    send_req2 = new MPI_Request[2], recv_req2 = new MPI_Request[2];

    vector<double> y_local(str_local * n);
    vector<double> y_next_top(n);
    vector<double> y_prev_low(n);

    int source_proc = myid ? myid - 1 : np - 1; // myid == 0 = np - 1
    int dest_proc = (myid != (np - 1)) ? myid + 1 : 0; // у myid == np - 1 = 0

    int scount = (myid != (np - 1)) ? n : 0; // у myid == np - 1 = 0
    int rcount = myid ? n : 0; // myid == 0 = 0

    vector<double> temp(y_local.size());

    // пересылаем верхние и нижние строки temp
    MPI_Send_init(temp.data(), rcount, MPI_DOUBLE, source_proc, 0, MPI_COMM_WORLD, send_req1);
    MPI_Recv_init(y_prev_low.data(), rcount, MPI_DOUBLE, source_proc, 1, MPI_COMM_WORLD, recv_req1);

    MPI_Send_init(temp.data() + (str_local - 1) * n, scount, MPI_DOUBLE, dest_proc, 1, MPI_COMM_WORLD, send_req1 + 1);
    MPI_Recv_init(y_next_top.data(), scount, MPI_DOUBLE, dest_proc, 0, MPI_COMM_WORLD, recv_req1 + 1);

    // пересылаем верхние и нижние строки y_local
    MPI_Send_init(y_local.data(), rcount, MPI_DOUBLE, source_proc, 0, MPI_COMM_WORLD, send_req2);
    MPI_Recv_init(y_prev_low.data(), rcount, MPI_DOUBLE, source_proc, 1, MPI_COMM_WORLD, recv_req2);

    MPI_Send_init(y_local.data() + (str_local - 1) * n, scount, MPI_DOUBLE, dest_proc, 1, MPI_COMM_WORLD, send_req2 + 1);
    MPI_Recv_init(y_next_top.data(), scount, MPI_DOUBLE, dest_proc, 0, MPI_COMM_WORLD, recv_req2 + 1);

    vector<double> y;
    if (myid == 0)
        y.resize(n * n);

    double t1 = -MPI_Wtime();

    int it = 0;
    bool flag = true;
    while (flag)
    {
        it++;
        //cout << "\n it = " << it;

        std::swap(temp, y_local);

        //// пересылаем верхние строки
        //MPI_Isend(temp.data(), rcount, MPI_DOUBLE, source_proc, 0, MPI_COMM_WORLD, send_req1);
        //MPI_Irecv(y_prev_low.data(), rcount, MPI_DOUBLE, source_proc, 1, MPI_COMM_WORLD, recv_req1 + 1);

        //// пересылаем нижние строки
        //MPI_Isend(temp.data() + (str_local - 1) * n, scount, MPI_DOUBLE, dest_proc, 1, MPI_COMM_WORLD, send_req1 + 1);
        //MPI_Irecv(y_next_top.data(), scount, MPI_DOUBLE, dest_proc, 0, MPI_COMM_WORLD, recv_req1);

        if (it % 2 == 0)
        {
            MPI_Startall(2, send_req1);
            MPI_Startall(2, recv_req1);
        }
        else
        {
            MPI_Startall(2, send_req2);
            MPI_Startall(2, recv_req2);
        }

        /* пересчитываем все строки в полосе кроме верхней и нижней пока идёт пересылка */
        for (int i = 1; i < str_local - 1; ++i)
            for (int j = 1; j < n - 1; ++j)
                y_local[i * n + j] = (temp[(i + 1) * n + j] + temp[(i - 1) * n + j] + temp[i * n + (j + 1)] + temp[i * n + (j - 1)] + hh * ff((nums_local + i) * h, j * h)) * q;

        if (it % 2 == 0)
        {
            MPI_Waitall(2, send_req1, MPI_STATUSES_IGNORE);
            MPI_Waitall(2, recv_req1, MPI_STATUSES_IGNORE);
        }
        else
        {
            MPI_Waitall(2, send_req2, MPI_STATUSES_IGNORE);
            MPI_Waitall(2, recv_req2, MPI_STATUSES_IGNORE);
        }

        //MPI_Waitall(2, send_req1, MPI_STATUSES_IGNORE);
        //MPI_Waitall(2, recv_req1, MPI_STATUSES_IGNORE);

        //MPI_Barrier;

        /* пересчитываем верхние строки */
        if (myid != 0)
            for (int j = 1; j < n - 1; ++j)
                y_local[j] = (temp[n + j] + y_prev_low[j] + temp[j + 1] + temp[j - 1] + hh * ff(nums_local * h, j * h)) * q;

        /* пересчитываем нижние строки */
        if (myid != np - 1)
            for (int j = 1; j < n - 1; ++j)
                y_local[(str_local - 1) * n + j] = (y_next_top[j] + temp[(str_local - 2) * n + j] + temp[(str_local - 1) * n + (j + 1)] + temp[(str_local - 1) * n + (j - 1)] + hh * ff((nums_local + (str_local - 1)) * h, j * h)) * q;

        norm_local = norm_minus(temp, y_local);

        MPI_Allreduce(&norm_local, &norm_err, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        if (norm_err < eps)
            flag = false;
    }

    t1 += MPI_Wtime();

    for (int i = 0; i < np; ++i)
    {
        str_per_proc[i] *= n;
        nums_start[i] *= n;
    }

    //MPI_Gather(y_local.data(), str_local * n, MPI_DOUBLE, y.data(), str_local * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(y_local.data(), str_local * n, MPI_DOUBLE, y.data(), str_per_proc.data(), nums_start.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (myid == 0)
    {
        // точное решение
        vector<double> u(n * n);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                u[i * n + j] = u_f(i * h, j * h);

        cout << "\n\n 3. Isend + Irecv";
        cout << "\n\t norm: " << norm_minus(y, u);
        cout << "\n\t iterations: " << it;
        printf("\n\t time: %.4f", t1);

        //write(y, n, n, "y.txt");
    }

    delete[] send_req1;
    delete[] recv_req1;
    delete[] send_req2;
    delete[] recv_req2;
}

void print(const vector<double>& vec, int m, int n)
{
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            cout << vec[i * n + j] << "\t";
        }
        cout << endl;
    }
}

void print(const vector<int>& vec)
{
    for (int i = 0; i < vec.size(); ++i)
        cout << vec[i] << "\t";
}

double norm_minus(const vector<double>& a, const vector<double>& b)
{
    double max = 0.0;
    double tmp = 0.0;

    for (int i = 0; i < a.size(); ++i)
    {
        tmp = fabs(a[i] - b[i]);
        if (tmp > max)
            max = tmp;
    }

    return max;
}

double norm(const vector<double>& vec)
{
    double sum = 0.0;

    for (int i = 0; i < vec.size(); ++i)
        sum += vec[i] * vec[i];

    return sqrt(h * sum);
}