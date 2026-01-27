#include "cppsolver.h"

CppSolver::CppSolver(const CppSolverConfig& cfg)
    : cfg_(cfg)
{
    // optional sanity checks
    // if (cfg_.factorization != "tangential") ...
}