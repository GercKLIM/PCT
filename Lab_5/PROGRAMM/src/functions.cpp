#include "../include/N-body-problem-cuda.cuh"

bool read(const std::string& path, std::vector<mytype>& global_m, std::vector<mytype3>& global_r, std::vector<mytype3>& global_v, int& N) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cout << "[LOG]: File " << path << " is NOT open. \n";
        return false;
    }

    file >> N;
    global_m.resize(N);
    global_r.resize(N);
    global_v.resize(N);

    for (int i = 0; i < N; ++i) {
        mytype3 r, v;
        file >> global_m[i] >> r.x >> r.y >> r.z >> v.x >> v.y >> v.z;
        global_r[i] = r;
        global_v[i] = v;
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
__host__ bool write(const std::string& path, const mytype3& r, mytype t, int number) {
    std::ofstream file(path + std::to_string(number) + ").txt", std::ios::app);

    if (!file.is_open()) {
        std::cout << "[LOG]: File " << path + std::to_string(number) + ").txt" << " is NOT open." << std::endl;
        return false;
    }
    file << t << "\t\t\t" << std::fixed << std::setprecision(12)
         << r.x << "\t\t\t" << r.y << "\t\t\t" << r.z << std::endl;
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

/** Функция получения параметров из .json файла */
bool input_parametres(const std::string& filename, std::string& test_filename, std::string& output_filename,
                      mytype& T, mytype&tau, mytype& EPS, bool& output, int& max_iterations){

    // Открываем файл
    std::ifstream config_file(filename);

    // Проверяем открытие файла
    if (!config_file.is_open()) {
        std::cout << "ERROR: Don't open " << filename  << std::endl;
        return false;
    }

    // Загружаем JSON

    Json::Reader json_reader;
    Json::Value json_root;

    bool read_succeeded = json_reader.parse(config_file, json_root);
    if (!read_succeeded) {
        std::cout << "ERROR: Parsing error" << std::endl;
        return false;
    }

    //t_seconds = json_root.get("dt", 1.0 / 300.0).asDouble();


    //NP = config.at("NP").get<int>();
    test_filename = json_root.get("test_filename", " ").asString();
    output_filename = json_root.get("output_filepath", " ").asString();
    T = json_root.get("T", 0).asDouble();
    tau = json_root.get("tau", 1).asDouble();
    EPS = json_root.get("EPS", 1).asDouble();
    output = json_root.get("output", false).asBool();
    max_iterations = json_root.get("max_iterations", 0).asInt();

    return true;
}



