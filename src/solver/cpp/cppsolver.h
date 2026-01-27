#pragma once
#include "cppsolver_config.h"

class CppSolver {
public:
    explicit CppSolver(const CppSolverConfig& cfg);
    // main entry point
    // torch::Tensor solve(...);

private:
    CppSolverConfig cfg_;
};