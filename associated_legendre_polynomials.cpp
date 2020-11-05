#include <cmath>
#include <iostream>
#include <iomanip>
#include <functional>
#include <string>

#ifdef USE_NUMERICAL_RECIPES
#include "plegendre.h"
#endif

#ifdef USE_GSL
#include <gsl/gsl_sf.h>
#endif

#define _USE_MATH_DEFINES

// calculate \tilde{P}_m^m
double tildePmm(const int m, const double x) {
    double factor = ((m % 2) == 0) ? 1.0 : -1.0;
    const double tmp = 1.0 - x * x;
    double tmp2 = 1.0;
    for (int i = 1; i <= m * 2 - 1; i += 2) {
        tmp2 *= tmp * static_cast<double>(i) / static_cast<double>(i + 1);
    }
    factor *= std::sqrt((2 * m + 1) * tmp2 / (4.0 * M_PI));
    return factor;
}

double tildePmm_derivative(const int m, const double x) {
    const double tilde_pmm = tildePmm(m, x);
    const double result = 0.5 * m * (-2.0) * x / (1 - x * x) * tilde_pmm;
    return result;
}

// calculate \tilde{P}_{(m+1)}^m
double tildePmm1(const int m, const double x) {
    return x * std::sqrt(2.0 * m + 3) * tildePmm(m, x);
}

double tildePmm1_derivative(const int m, const double x) {
    return x * std::sqrt(2.0 * m + 3) * tildePmm_derivative(m, x) + std::sqrt(2.0 * m + 3) * tildePmm(m, x);
}

// calculate \tilde{P}_m^l
double tildePlm(const int l, const int m, const double x) {
    if (m < 0 || m > l || std::abs(x) > 1.0) {
        std::cerr << "Bad arguments in tildePlm.\n";
        return 0;
    }
    const double factor1 = std::sqrt((4.0 * l * l - 1) / (l * l - m * m));
    const double factor2 = std::sqrt(((l - 1) * (l - 1) - m * m) / (4.0 * (l - 1) * (l - 1) - 1));
    if (l == m) return tildePmm(m, x);
    if (l == m + 1) return tildePmm1(m, x);
    return factor1 * (x * tildePlm(l-1, m, x) - factor2 * tildePlm(l-2, m, x));
}

double tildePlm_derivative(const int l, const int m, const double x) {
    if (m < 0 || m > l || std::abs(x) > 1.0) {
        std::cerr << "Bad arguments in tildePlm.\n";
        return 0;
    }
    const double factor1 = std::sqrt((4.0 * l * l - 1) / (l * l - m * m));
    const double factor2 = std::sqrt(((l - 1) * (l - 1) - m * m) / (4.0 * (l - 1) * (l - 1) - 1));
    if (l == m) return tildePmm_derivative(m, x);
    if (l == m + 1) return tildePmm1_derivative(m, x);
    return factor1 * (tildePlm(l-1, m, x) + x * tildePlm_derivative(l-1, m, x) - factor2 * tildePlm_derivative(l-2, m, x));
}

double Plm(const int l, const int m, const double x) {
    double tmp1 = 1.0;
    double tmp2 = 1.0;
    if (m < 0 || m > l || std::abs(x) > 1.0) {
        std::cerr << "Bad arguments in Plm.\n";
        return 0;
    }
    for (int i = 1; i <= (l - m); ++i) {
        tmp1 *= static_cast<double>(i);
    }
    for (int i = 1; i <= (l + m); ++i) {
        tmp2 *= static_cast<double>(i);
    }
    const double factor = std::sqrt((4.0 * M_PI * tmp2) / ((2.0 * l + 1) * tmp1));
    return factor * tildePlm(l, m, x);
}

double Plm_derivative(const int l, const int m, const double x) {
    double tmp1 = 1.0;
    double tmp2 = 1.0;
    if (m < 0 || m > l || std::abs(x) > 1.0) {
        std::cerr << "Bad arguments in Plm.\n";
        return 0;
    }
    for (int i = 1; i <= (l - m); ++i) {
        tmp1 *= static_cast<double>(i);
    }
    for (int i = 1; i <= (l + m); ++i) {
        tmp2 *= static_cast<double>(i);
    }
    const double factor = std::sqrt((4.0 * M_PI * tmp2) / ((2.0 * l + 1) * tmp1));
    return factor * tildePlm_derivative(l, m, x);
}

double numericalDerivative(std::function<double(double)> f, double x, double width) {
    return (f(x + width) - f(x - width)) / (width * 2.0);
}

void checkDerivative(std::function<double(double)> f, std::function<double(double)> df, double x, const std::string& function_name) {
    static const int magnitude = 5;
    double delta_x = 0.1;
    const double analytical_derivative = df(x);
    std::cout << "Checking derivative of " << function_name << " for x = " << x << '\n';
    std::cout << "Analytical derivative: " << analytical_derivative << std::endl;
    for (int i = 1; i <= magnitude; ++i) {
        const double numerical_derivative = numericalDerivative(f, x, delta_x);
        const double rmse = std::sqrt((numerical_derivative - analytical_derivative) * (numerical_derivative - analytical_derivative));
        std::cout << "Delta_x = " << delta_x << " ; "
                  << "numerical derivative = " << numerical_derivative << " ; "
                  << "RMSE = " << rmse << '\n';
        delta_x = delta_x / 10;
    }
    std::cout << "=========================================\n";
}

int main() {
    int l = 6;
    int m = 2;
    double x = 0.5;
    std::cin >> x;
    std::cout << std::fixed << std::setprecision(9);
    std::cout << tildePlm(l, m, x) << std::endl;
#ifdef USE_NUMERICAL_RECIPES
    std::cout << plegendre(l, m, x) << std::endl;
#endif
    std::cout << Plm(l, m, x) << std::endl;
#ifdef USE_NUMERICAL_RECIPES
    std::cout << plgndr(l, m, x) << std::endl;
#endif
    std::cout << "STL (no Condon-Shortley phase term): " << std::assoc_legendre(l, m, x) << std::endl;
#ifdef USE_GSL
    std::cout << "GSL: " << gsl_sf_legendre_Plm(l, m, x) << std::endl;
#endif
    using namespace std::placeholders;
    auto f1 = std::bind(tildePmm, m, _1);
    auto df1 = std::bind(tildePmm_derivative, m, _1);
    checkDerivative(f1, df1, x, "tildePmm(m, x)");
    auto f2 = std::bind(tildePmm1, m, _1);
    auto df2 = std::bind(tildePmm1_derivative, m, _1);
    checkDerivative(f2, df2, x, "tildePmm1(m, x)");
    auto f3 = std::bind(tildePlm, l, m, _1);
    auto df3 = std::bind(tildePlm_derivative, l, m, _1);
    checkDerivative(f3, df3, x, "tildePlm(l, m, x)");
    auto f4 = std::bind(Plm, l, m, _1);
    auto df4 = std::bind(Plm_derivative, l, m, _1);
    checkDerivative(f4, df4, x, "Plm(l, m, x)");
    return 0;
}
