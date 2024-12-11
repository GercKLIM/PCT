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

void test1(int argc, char* argv[]){

    /* ### Инициализация переменных ### */
    const std::string PTS_FILENAME = "../INPUT/input.json";
    std::string TEST_FILENAME;
    std::string OUTPUT_FILEPATH;

    int ID = 0; // Идентификатор текущего процесса
    int NP = 1; // Кол-во процессов

    int N = 0;              // Кол-во тел
    double tau = 1.0;       // Шаг по времени
    double T = 20.0;        // Конечный момент времени
    double countStep = 20.0;//
    double time;            //
    double EPS;             // Допустимая погрешность
    std::vector<Body> input;//
    bool output = true;     //
    int max_iteration;




    /* ### Получаем параметры программы ### */


    input_json(PTS_FILENAME, "test_filename", TEST_FILENAME);
    input_json(PTS_FILENAME, "output_filepath", OUTPUT_FILEPATH);
    input_json(PTS_FILENAME, "EPS", EPS);
    input_json(PTS_FILENAME, "T", T);
    input_json(PTS_FILENAME, "tau", tau);
    input_json(PTS_FILENAME, "EPS", EPS);
    input_json(PTS_FILENAME, "output", output);
    input_json(PTS_FILENAME, "max_iteration", max_iteration);



    if (ID == 0) {
        clear_files(TEST_FILENAME, N);
        read(TEST_FILENAME, input, N);
    }



    /* ### Запускаем MPI ### */



    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &NP); // Получаем кол-во процессов
    MPI_Comm_rank(MPI_COMM_WORLD, &ID); // Получаем номер текущего процесса



    /* ### Создаем свою пересылку ### */



    /* Тела */
    int count = 3;
    int lengths[3] = { 1, 3, 3 };
    MPI_Aint offsets[3] = { offsetof(Body, m), offsetof(Body, r), offsetof(Body, v) };
    MPI_Datatype types[3] = { MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE };

    MPI_Datatype mpi_body;
    MPI_Type_create_struct(count, lengths, offsets, types, &mpi_body);
    MPI_Type_commit(&mpi_body);

    /* Скорость */
    int count_v = 1;
    int lengths_v[1] = { 3 };
    MPI_Aint offsets_v[1] = { offsetof(Body, v) };
    MPI_Datatype types_v[1] = { MPI_DOUBLE };
    MPI_Datatype mpi_body_v;
    MPI_Type_create_struct(count_v, lengths_v, offsets_v, types_v, &mpi_body_v);
    //MPI_Type_commit(&mpi_body_v);

    MPI_Datatype mpi_body_v_part;
    MPI_Type_create_resized(mpi_body_v, offsetof(Body, v), sizeof(Body), &mpi_body_v_part);
    MPI_Type_commit(&mpi_body_v_part);



    /* ### Решаем задачу ### */

    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    input.resize(N);
    MPI_Bcast(input.data(), N, mpi_body, 0, MPI_COMM_WORLD);

    Runge_Kutta_MPI(OUTPUT_FILEPATH, input, tau, T, time, EPS,
                    output, NP, ID, N, mpi_body_v_part, max_iteration); //здесь поменять

    MPI_Finalize();

    if (ID == 0){
        std::cout << "TIME = " << time << std::endl;
    }

}


void test_fout(int argc, char* argv[]){

    /* ### Инициализация переменных ### */
    const std::string PTS_FILENAME = "../INPUT/input.json";
    std::string TEST_FILENAME;
    std::string OUTPUT_FILEPATH;

    int ID = 0; // Идентификатор текущего процесса
    int NP = 1; // Кол-во процессов

    int N = 0;              // Кол-во тел
    double tau = 1.0;       // Шаг по времени
    double T = 20.0;        // Конечный момент времени
    double countStep = 20.0;//
    double time;            //
    double EPS;             // Допустимая погрешность
    std::vector<Body> input;//
    bool output = true;     //
    int max_iteration;




    /* ### Получаем параметры программы ### */


    input_json(PTS_FILENAME, "test_filename", TEST_FILENAME);
    input_json(PTS_FILENAME, "output_filepath", OUTPUT_FILEPATH);
    input_json(PTS_FILENAME, "EPS", EPS);
    input_json(PTS_FILENAME, "T", T);
    input_json(PTS_FILENAME, "tau", tau);
    input_json(PTS_FILENAME, "EPS", EPS);
    input_json(PTS_FILENAME, "output", output);
    input_json(PTS_FILENAME, "max_iteration", max_iteration);

    output = false;

    if (ID == 0) {
        clear_files(TEST_FILENAME, N);
        read(TEST_FILENAME, input, N);
    }



    /* ### Запускаем MPI ### */



    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &NP); // Получаем кол-во процессов
    MPI_Comm_rank(MPI_COMM_WORLD, &ID); // Получаем номер текущего процесса



    /* ### Создаем свою пересылку ### */



    /* Тела */
    int count = 3;
    int lengths[3] = { 1, 3, 3 };
    MPI_Aint offsets[3] = { offsetof(Body, m), offsetof(Body, r), offsetof(Body, v) };
    MPI_Datatype types[3] = { MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE };

    MPI_Datatype mpi_body;
    MPI_Type_create_struct(count, lengths, offsets, types, &mpi_body);
    MPI_Type_commit(&mpi_body);

    /* Скорость */
    int count_v = 1;
    int lengths_v[1] = { 3 };
    MPI_Aint offsets_v[1] = { offsetof(Body, v) };
    MPI_Datatype types_v[1] = { MPI_DOUBLE };
    MPI_Datatype mpi_body_v;
    MPI_Type_create_struct(count_v, lengths_v, offsets_v, types_v, &mpi_body_v);
    //MPI_Type_commit(&mpi_body_v);

    MPI_Datatype mpi_body_v_part;
    MPI_Type_create_resized(mpi_body_v, offsetof(Body, v), sizeof(Body), &mpi_body_v_part);
    MPI_Type_commit(&mpi_body_v_part);



    /* ### Решаем задачу ### */

    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    input.resize(N);
    MPI_Bcast(input.data(), N, mpi_body, 0, MPI_COMM_WORLD);

    Runge_Kutta_MPI(OUTPUT_FILEPATH, input, tau, T, time, EPS,
                    output, NP, ID, N, mpi_body_v_part, max_iteration); //здесь поменять

    MPI_Finalize();

    if (ID == 0) {
        std::ofstream file("../OUTPUT/RESULT.txt", std::ios::app);
        if (!file.is_open()) {
            std::cout << "[LOG]: File is NOT open.";
        }

        file << NP << " " << time << "\n";

        file.close();
    }

}


int main(int argc, char* argv[]) {


    test_fout(argc,argv);


    std::cout << "Complete!" << std::endl;
    return EXIT_SUCCESS;
}
