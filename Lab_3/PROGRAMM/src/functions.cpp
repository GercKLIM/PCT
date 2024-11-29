
/* --------------------------------------------------- */
/* ### РЕАЛИЗАЦИЯ ДОПОЛНИТЕЛЬНЫх ФУНКЦИЙ ПРОГРАММЫ ### */
/* --------------------------------------------------- */

#include "../include/Helmholtz_Solver.h"
#include <iostream>



/** Функция Нормы разности векторов */
double NOD(const int& size, const double& h, const std::vector<double>& A, const std::vector<double>& B) {
    double sum = 0.0;
    double tmp = 0.;

/**/#pragma omp parallel for default(none) shared(size,A,B) private(tmp) reduction(+:sum)
    for (int i = 0; i < size; ++i){
        tmp = A[i] - B[i];
        sum += tmp * tmp;
    }
    return sqrt(sum * h);
}



/** Функция получения параметров из .json файла */
bool input_parametres(const std::string& filename, int& N, double& K,
                      int& max_iterations, double& EPS, std::string& test_name){

    // Открываем файл
    std::ifstream config_file(filename);

    // Проверяем открытие файла
    if (!config_file.is_open()) {
        std::cout << Logs::LOG_ERROR << "ERROR: Don't open " << filename  << std::endl;
        return false;
    }

    // Загружаем JSON
    nlohmann::json config;
    try {
        config_file >> config;
    } catch (const nlohmann::json::parse_error& e) {
        std::cout << Logs::LOG_ERROR << "ERROR: Parsing error with " << e.what() << "\n";
        return false;
    }

    // Получаем значения из JSON
    try {

        //NP = config.at("NP").get<int>();
        N = config.at("N").get<int>();
        K = config.at("K").get<double>();
        max_iterations = config.at("max_iterations").get<int>();
        EPS = config.at("EPS").get<double>();
        test_name = config.at("test_name").get<std::string>();

    } catch (const nlohmann::json::out_of_range& e) {
        std::cout << Logs::LOG_ERROR << "ERROR: " <<  e.what() << "\n";
        return false;
    }

    return true;
}



/** Функция для проверки корректности решения уравнения */
double test_sol(const int& N, const std::vector<double>& y, std::function<double(double, double)>& True_sol_func) {

    double h = 1. / (N - 1);
    std::vector<double> y_true(N*N, 0.0);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            y_true[i * N + j] = True_sol_func(i * h, j * h);
        }
    }

    double norm_res = NOD(N*N, h, y, y_true);
    return norm_res;
}



/** Функция раздача работы */
void Work_Distribution(int NP, int N, int& str_local, int& nums_local,
                       std::vector<int>& str_per_proc, std::vector<int>& nums_start){

    str_per_proc.resize(NP, N / NP);
    nums_start.resize(NP, 0);

    for (int i = 0; i < N % NP; ++i)
        ++str_per_proc[i];

    for (int i = 1; i < NP; ++i)
        nums_start[i] = nums_start[i - 1] + str_per_proc[i - 1];

    // Передача работы процессам
    int ID = 0; // Процесс, который раздает работы
    MPI_Scatter(str_per_proc.data(), 1, MPI_INT, &str_local, 1, MPI_INT, ID, MPI_COMM_WORLD);
    MPI_Scatter(nums_start.data(), 1, MPI_INT, &nums_local, 1, MPI_INT, ID, MPI_COMM_WORLD);
}

/* Вывод результатов метода */
void print_MethodResultInfo(const MethodResultInfo& MR){

    std::cout << "<----------------------------------->"  << std::endl;
    std::cout << " ### "      << MR.method_name << " ### "<< std::endl << std::endl;
    std::cout << "Norm    = " << MR.norm_sol              << std::endl;
    std::cout << "Iter    = " << MR.iterations            << std::endl;
    std::cout << "Time    = " << MR.time                  << std::endl;
    std::cout << "|Y-Yp|  = " << MR.norm_iter             << std::endl;
    std::cout << "Threads = " << MR.NP                    << std::endl;
    std::cout << "<----------------------------------->"  << std::endl;
}


/* Вывод результатов метода в файл */
void print_MethodResultInfoFile(const MethodResultInfo& MR, std::ofstream& Fout){


    Fout << "<----------------------------------->"  << std::endl;
    Fout << " ### "      << MR.method_name << " ### "<< std::endl << std::endl;
    Fout << "Norm    = " << MR.norm_sol              << std::endl;
    Fout << "Iter    = " << MR.iterations            << std::endl;
    Fout << "Time    = "<< MR.time                   << std::endl;
    Fout << "|Y-Yp|  = " << MR.norm_iter             << std::endl;
    Fout << "Threads = " << MR.NP                    << std::endl;
    Fout << "<----------------------------------->"  << std::endl;

}


