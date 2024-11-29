/* Лабораторная работа № 4 - Решение задачи N тел
 *
 * Условие - в приложенном файле Microsoft Word. Некоторые замечания:
 * 1) данные о телах хранить в структуре/классе (масса, положение, скорость);
 * 2) реализовать метод Рунге - Кутты p-го порядка (p>1, чаще всего 2 или 4) для интегрирования уравнений движения;
 * 3) использовать коллективные операции передачи данных в MPI;
 * 4) данные передавать с помощью производного типа данных, пересылать только в объеме изменившейся части данных;
 * 5) с использованием приложенных данных для 4 тел произвести тестовый расчет,
 *    показать работоспособность реализации и порядок метода;
 *    решить задачу для большого числа тел с произвольными исходными данными (число тел порядка 10^3..10^4),
 *    вычислять среднее время для некоторого числа шагов (10-20-50, более не требуется)
 *    и ускорение для различного числа процессов.
 *
 * */

#include "include/N-BodyProblem.h"


int main(int argc, char* argv[]) {

    int ID = 0; // Идентификатор текущего процесса
    int NP = 1; // Кол-во процессов

    /* Запускаем MPI */
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &NP); // Получаем кол-во процессов
    MPI_Comm_rank(MPI_COMM_WORLD, &ID); // Получаем номер текущего процесса

    /* Делаем пересылку */

    /* Тела */
    int count = 3;
    int lengths[3] = { 1, 3, 3 };
    MPI_Aint offsets[3] = { offsetof(Body, m), offsetof(Body, r), offsetof(Body, v) };
    MPI_Datatype types[3] = { MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE };

    MPI_Datatype mpi_body;
    MPI_Type_create_struct(count, lengths, offsets, types, &mpi_body);
    MPI_Type_commit(&mpi_body);

    // скорость
    int count_v = 1;
    int lengths_v[1] = { 3 };
    MPI_Aint offsets_v[1] = { offsetof(Body, v) };
    MPI_Datatype types_v[1] = { MPI_DOUBLE };
    MPI_Datatype mpi_body_v;
    MPI_Type_create_struct(count_v, lengths_v, offsets_v, types_v, &mpi_body_v);
    MPI_Type_commit(&mpi_body_v);

    MPI_Datatype mpi_body_v_part;
    MPI_Type_create_resized(mpi_body_v, offsetof(Body, v), sizeof(Body), &mpi_body_v_part);
    MPI_Type_commit(&mpi_body_v_part);

    //std::string path = "4body.txt";
    std::string path = "60kbody.txt";

    int N = 0;
    std::vector<Body> input;
    if (ID == 0)
        read(path, input, N);

    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    input.resize(N);
    MPI_Bcast(input.data(), N, mpi_body, 0, MPI_COMM_WORLD);

    double tau = 1.0, countStep = 20.0, time;
    bool output;

    double T = 20.0;

    //if (myid == 0)
    //    cout << "\n N=" << N << "\n";

    if (path == "4body.txt")
    {
        T = 20.0;
        output = true;
    }
    else
    {
        T = countStep * tau;
        output = false;
    }

    if (output)
        clear_files("solution(", N);

    if (ID == 0)
        std::cout << "\n T=" << T << "\n";

    Runge_Kutta_MPI("solution(", input, tau, T, time, output, np, myid, N, mpi_body_v_part); //здесь поменять

    if (ID == 0)
        std::cout << "time: " << time / countStep << std::endl;

    MPI_Finalize();

    return 0;
}
