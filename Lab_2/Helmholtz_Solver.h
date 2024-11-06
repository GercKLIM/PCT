/* --------------------------------------------------------- */
/* ### ОБЪЯВЛЕНИЕ ФУНКЦИЙ РЕШЕНИЯ УРАВНЕНИЯ ГЕЛЬМГОЛЬЦА  ### */
/* --------------------------------------------------------- */


#pragma once
#include <iostream>
#include <vector> // Для доступа к типу std::vector
#include "omp.h"  // Для распараллеливания вычислений
#include <ctime>  // Для инициализации генератора
#include <fstream>// Для работы с файлами
#include <cmath>  // Для математических функций
#include <functional> // Для передачи функторов


/* Функция нормы разности векторов */
double DifferentNorm(const int& size, const double& h, const std::vector<double>& A, const std::vector<double>& B);


/** Метод решения двумерного уравнения Гельмгольца методом Якоби
 * @param y - массив решения
 * @param f - функция правой части
 * @param k - коэффициент в уравнении
 * @param N - число разбиений
 * @param max_num_iterations - Макс. кол-во итераций
 */
void Method_Jacobi(std::vector<double>& y, std::function<double(double, double)>&f, const double& k, const int& N,
                   const double& eps, const int& max_num_iterations = 1000);
