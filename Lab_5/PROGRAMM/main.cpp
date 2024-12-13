/* ### Лабораторная работа № 5- Решение задачи N тел на CUDA ###
 *
 * 1) время измерять с помощью событий CUDA;
 *
 * 2) программу реализовать с учетом возможности быстрого переключения типа данных float и double;
 *
 * 3) для вычисления ускорения каждого тела с использованием всех остальных использовать shared-память
 *    (учесть возможность количества тел не кратного число потоков);
 *
 * 4) использовать специальные математические и intrinsic-функции (см. CUDA C Programming Guide, Appendix H)
 *    такие, как __fdividef, rsqrtf и др.
 *
 */


#include "include/N-body-problem-cuda.cuh"


void test1(){
    /* ### Инициализация переменных ### */



    const std::string PTS_FILENAME = "../INPUT/input.json";
    std::string TEST_FILENAME;
    std::string OUTPUT_FILEPATH;

    int N = 0;              // Кол-во тел
    double tau = 1.0;       // Шаг по времени
    double T = 20.0;        // Конечный момент времени
    double countStep = 20.0;
    double time;            // Время работы алгоритма
    double EPS;             // Допустимая погрешность
    bool output = false;     // Опция записи в файл
    int max_iterations;      // Ограничение итераций

    std::vector<mytype> global_m; // Вектор всех масс
    std::vector<mytype> global_r; // Вектор всех координат
    std::vector<mytype> global_v; // Вектор всех скоростей



    /* ### Получаем параметры программы ### */



    input_parametres(PTS_FILENAME, TEST_FILENAME, OUTPUT_FILEPATH,
                          T, tau, EPS, output, max_iterations);


    read(TEST_FILENAME, global_m, global_r, global_v, N);
    if (output) {
        clear_files(OUTPUT_FILEPATH + "body-(", N);
    }





    /* ### РЕШЕНИЕ ЗАДАЧИ ###*/

    // Вывод кол-ва операций
    //double op = ((BS + BS + 21 * BS + 16) * N + N + 24 + 8 + N) * 4 + 29 * 6 + 113;
    //std::cout << "[LOG]: Num of operations = " << op << std::endl;


    time = Runge_Kutta(OUTPUT_FILEPATH + "body-(", global_m, global_r, global_v, tau, T, output);


    std::ofstream F(OUTPUT_FILEPATH + "RESULT.txt", std::ios::app);
    if (!F.is_open()){
        std::cout << "[LOG]: File " << OUTPUT_FILEPATH + "RESULT.txt" << " is NOT open." << std::endl;
    }

    F << "Type = "<< typeid(mytype).name() << ", N = " << N  << ", T = " << T << ", tau = " << tau <<
    ", TIME = " << time * T / tau << ", TIME per iter = " << time << std::endl;

    typeid(mytype);

    std::cout << "[LOG]: Time per step: " << time / 1000. << std::endl;
    //std::cout << "[LOG]: Profit: " << op * N * N / time * 1000. << std::endl;
}


int main(int argc, char* argv[]) {

    test1();
    std::cout << " \n[LOG]: Complete! \n" << std::endl;

    return 0;

}