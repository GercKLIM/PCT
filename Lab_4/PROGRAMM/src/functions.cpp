/*  РЕАЛИЗАЦИЯ ДОПОЛНИТЕЛЬНЫХ ФУНКЦИЙ
 *
 * */

#include "../include/N-BodyProblem.h"



/** Функция получения параметров из .json файла */
bool input_parametres(const std::string& filename, std::string& test_filename, std::string& output_filename,
                      double& T, double&tau, double& EPS, bool& output, int& max_iterations){

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

    for (int i = 0; i < a.size(); ++i) {

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

    std::ofstream output(path + "body-(" + std::to_string(num) + ").txt", std::ios::app);

    output << t << "\t" << std::fixed << std::setprecision(16) <<
           body.r[0] << "\t" << body.r[1] << "\t" << body.r[2] << std::endl;

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


///** Функция получения параметров из .json файла */
//template <typename type>
//bool input_json(const std::string& filename, const std::string& name, type& pts){
//
//    // Открываем файл
//    std::ifstream config_file(filename);
//
//    // Проверяем открытие файла
//    if (!config_file.is_open()) {
//        std::cout << "[LOG]: " <<  "ERROR: Don't open " << filename  << std::endl;
//        return false;
//    }
//
//    // Загружаем JSON
//    nlohmann::json config;
//    try {
//        config_file >> config;
//    } catch (const nlohmann::json::parse_error& e) {
//        std::cout <<  "[LOG]: " << "ERROR: Parsing error with " << e.what() << "\n";
//        return false;
//    }
//
//    // Получаем значения из JSON
//    try {
//        //NP = config.at("NP").get<int>();
//        pts = config.at(name).get<type>();
//
//    } catch (const nlohmann::json::out_of_range& e) {
//        std::cout << "[LOG]: " <<  "ERROR: " <<  e.what() << "\n";
//        return false;
//    }
//
//    return true;
//}

