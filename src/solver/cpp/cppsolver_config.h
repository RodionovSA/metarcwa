#pragma once
#include <string>

struct CppSolverConfig {
    std::string inverse_matrix_method;
    std::string factorization;
    std::string modes_solver;
    bool h_simplify;
    bool sm_layer;
};