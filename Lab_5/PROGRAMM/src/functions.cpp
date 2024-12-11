#include "../include/N-body-problem-cuda.cuh"

bool read(const std::string& path, std::vector<mytype>& global_m, std::vector<mytype>& global_r, std::vector<mytype>& global_v, int& N) {
    std::ifstream file(path);

    if (!file.is_open()) {
        std::cout << "[LOG]: File " << path << " is NOT open. \n";
        return false;
    }

    file >> N;
    global_m.resize(N);
    global_r.resize(N * 3);
    global_v.resize(N * 3);

    for (size_t i = 0; i < N; ++i) {
        file >> global_m[i] >> global_r[3 * i] >> global_r[3 * i + 1] >> global_r[3 * i + 2]
             >> global_v[3 * i] >> global_v[3 * i + 1] >> global_v[3 * i + 2];
    }

    file.close();

    return true;
}

/**
 * @brief Записывает текущие координаты тела в файл.
 * @param path Базовый путь к файлу.
 * @param r Вектор координат тела.
 * @param t Текущее время.
 * @param number Номер тела.
 */
__host__ bool write(const std::string& path, const std::vector<mytype>& r, mytype t, int number) {
    std::ofstream file(path + std::to_string(number) + ").txt", std::ios::app);

    if (!file.is_open()) {
        std::cout << "[LOG]: File " << path + std::to_string(number) + ").txt" << " is NOT open." << std::endl;
        return false;
    }
    file << t << "\t\t\t" << std::fixed << std::setprecision(12)
         << r[0] << "\t\t\t" << r[1] << "\t\t\t" << r[2] << std::endl;
    file.close();
    return true;
}

/**
 * @brief Очищает файлы вывода перед началом записи.
 * @param path Базовый путь к файлам.
 * @param N Число тел.
 */
__host__ void clear_files(const std::string& path, int N) {
    for (int i = 1; i <= N; ++i) {
        std::ofstream output(path + std::to_string(i) + ").txt");
        output.close();
    }
}



