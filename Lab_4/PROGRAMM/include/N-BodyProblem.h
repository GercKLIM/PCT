/*  ОБЪЯВЛЕНИЕ ФУНКЦИЙ РЕШЕНИЯ ЗАДАЧИ N-ТЕЛ
 *
 * */

#pragma once

#include <iostream>
#include <vector>
#include <array>
#include <string>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include "mpi.h"

#include <nlohmann/json.hpp> // Для работы с json файлами


const double G = 6.67 * 1e-11; // Гравитационная постоянная



/** Структура Тела
 *   - масса
 *   - 3 координаты,
 *   - 3 скорости
 * */
struct Body {
    double m;                // Масса тела
    std::array<double, 3> r; // Начальные положение тела
    std::array<double, 3> v; // Начальные скорость тела

    Body() : m(0), r{ 0,0,0 }, v{ 0,0,0 } {};
};


/* ### Операции над array и vector ### */

/* Операция умножения массива на число */
std::array<double, 3>& operator*= (std::array<double, 3>& arr, double alpha);


/* Операция сложения массивов */
std::array<double, 3>& operator+= (std::array<double, 3>& a, const std::array<double, 3> b);


/* Операция домножения координат и скоростей на число */
std::vector<Body>& operator*= (std::vector<Body>& bodies, double alpha);


/* Функция сложения двух массивов тел */
std::vector<Body>& operator+= (std::vector<Body>& a, const std::vector<Body>& b);

/* Функция разности массивов */
// res = a - b
void array_diff(const std::array<double, 3>& a, const std::array<double, 3>& b,
                std::array<double, 3>& res);

/* Функция сложения? */
// res = a + alpha * b;
void vec_add_mul(const std::vector<Body>& a, const std::vector<Body>& b,
                 double alpha, std::vector<Body>& res);


/* Функция нормы R^3 */
double norm(const std::array<double, 3>& arr);



/* ### Функции для работы с файлами ### */



/** Функция чтения из файла
 * @param path    - Файл для чтения
 * @param bodies  - Массив Тел
 * @param N       - Кол-во Тел
 */
void read(const std::string& path, std::vector<Body>& bodies, int& N);


/** Функция записи в файл
 * @param path   - Файл для записи
 * @param body   - Массив тел для записи
 * @param t      - Момент времени
 * @param num    - номер?
 */
void write(const std::string& path, const Body& body, double t, int num);


/** Функция очистки файла
 *
 * @param path - Путь к файлам для очищения
 * @param N    - Кол-во файлов
 */
void clear_files(const std::string& path, int N);


/** Функция получения параметров из .json файла */
template <typename type>
bool input_json(const std::string& filename, const std::string& name, type& pts) {
    std::ifstream config_file(filename);
    if (!config_file.is_open()) {
        std::cerr << "[LOG]: ERROR: Unable to open " << filename << std::endl;
        return false;
    }
    nlohmann::json config;
    try {
        config_file >> config;
    } catch (const nlohmann::json::parse_error& e) {
        std::cerr << "[LOG]: ERROR: Parsing error: " << e.what() << "\n";
        return false;
    }
    try {
        pts = config.at(name).get<type>();
    } catch (const nlohmann::json::out_of_range& e) {
        std::cerr << "[LOG]: ERROR: Key \"" << name << "\" not found: " << e.what() << "\n";
        return false;
    }
    return true;
}



/* ### Методы Рунге-Кутты ### */



/** Функция, описывающая систему диффуров, по формулам из файла
 *
 * @param left
 * @param right
 * @param start
 * @param end
 */
void f(std::vector<Body>& left, const std::vector<Body>& right, int start, int end);


/** Функция метода Рунге-Кутты 4-го порядка
 *
 * @param path
 * @param init
 * @param tau
 * @param T
 * @param t
 * @param output
 */
void Runge_Kutta(const std::string& path, const std::vector<Body>& init, double tau,
                 double T, double& t, const double& EPS, bool output);


/** Функция метода Рунге-Кутты 4-го порядка с MPI
 *
 * @param path
 * @param init
 * @param tau
 * @param T
 * @param t
 * @param output
 * @param NP
 * @param myid
 * @param N
 * @param MPI_BODY_VPART
 */
void Runge_Kutta_MPI(const std::string& path, const std::vector<Body>& init, double tau,
                     double T, double& t, const double& EPS, bool output,
                     int NP, int myid, int N, MPI_Datatype MPI_BODY_VPART, const int& max_iteration);













