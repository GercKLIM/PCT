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



    const std::string PTS_FILENAME = "INPUT/input.json";
    std::string TEST_FILENAME;
    std::string OUTPUT_FILEPATH;

    int N = 0;              // Кол-во тел
    mytype tau = 1.0;       // Шаг по времени
    mytype T = 20.0;        // Конечный момент времени
    mytype countStep = 20.0;
    mytype time;            // Время работы алгоритма
    mytype EPS;             // Допустимая погрешность
    bool output = false;     // Опция записи в файл
    int max_iterations;      // Ограничение итераций

    std::vector<mytype> global_m; // Вектор всех масс
    std::vector<mytype3> global_r; // Вектор всех координат
    std::vector<mytype3> global_v; // Вектор всех скоростей



    /* ### Получаем параметры программы ### */



    input_parametres(PTS_FILENAME, TEST_FILENAME, OUTPUT_FILEPATH,
                          T, tau, EPS, output, max_iterations);


    read(TEST_FILENAME, global_m, global_r, global_v, N);
    if (output) {
        clear_files(OUTPUT_FILEPATH + "body-(", N);
    }





    /* ### РЕШЕНИЕ ЗАДАЧИ ###*/


    time = Runge_Kutta(OUTPUT_FILEPATH + "body-(", global_m, global_r, global_v, tau, T, output);


//    std::ofstream F(OUTPUT_FILEPATH + "RESULT.txt", std::ios::app);
//    if (!F.is_open()){
//        std::cout << "[LOG]: File " << OUTPUT_FILEPATH + "RESULT.txt" << " is NOT open." << std::endl;
//    }
//
//    F << "Type = "<< typeid(mytype).name() << ", N = " << N  << ", T = " << T << ", tau = " << tau <<
//    ", TIME = " << time * T / tau << ", TIME per iter = " << time << std::endl;
//
//    std::cout << "Type = "<< typeid(mytype).name() << ", N = " << N  << ", T = " << T << ", tau = " << tau <<
//      ", TIME = " << time * T / tau << ", TIME per iter = " << time << std::endl;
//    typeid(mytype);

    //std::cout << "[LOG]: Time per step: " << time / 1000. << std::endl;
    //std::cout << "[LOG]: Profit: " << op * N * N / time * 1000. << std::endl;
}


void res_for_tables(){
    /* ### Инициализация переменных ### */

    const std::string PTS_FILENAME = "INPUT/input.json";
    std::string TEST_FILENAME;
    std::string OUTPUT_FILEPATH;

    int N = 0;              // Кол-во тел
    mytype tau = 1.0;       // Шаг по времени
    mytype T = 20.0;        // Конечный момент времени
    mytype countStep = 20.0;
    mytype time;            // Время работы алгоритма
    mytype EPS;             // Допустимая погрешность
    bool output = false;     // Опция записи в файл
    int max_iterations;      // Ограничение итераций

    std::vector<mytype> global_m; // Вектор всех масс
    std::vector<mytype3> global_r; // Вектор всех координат
    std::vector<mytype3> global_v; // Вектор всех скоростей


    /* ### Получаем параметры программы ### */

    input_parametres(PTS_FILENAME, TEST_FILENAME, OUTPUT_FILEPATH,
                     T, tau, EPS, output, max_iterations);



//    if (output) {
//        clear_files(OUTPUT_FILEPATH + "body-(", N);
//    }

    /* ### РЕШЕНИЕ ЗАДАЧИ ###*/
    for (int i = 1; i <= 5; ++i){
        read(TEST_FILENAME, global_m, global_r, global_v, N);
        if (output) {
          clear_files(OUTPUT_FILEPATH + std::to_string(i) + "_body-(", N);
        }
        time = Runge_Kutta(OUTPUT_FILEPATH + std::to_string(i) + "_body-(", global_m, global_r, global_v, tau, T, output);
        tau /= 2.;
    }
}


int main(int argc, char* argv[]) {

    //test1();
    res_for_tables();
    std::cout << " \n[LOG]: Complete! \n" << std::endl;

    return 0;

}