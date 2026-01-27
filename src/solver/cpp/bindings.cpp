#include <pybind11/pybind11.h>
#include "cppsolver.h"
#include "cppsolver_config.h"

namespace py = pybind11;

PYBIND11_MODULE(rcwa_cpp, m) {
    py::class_<CppSolverConfig>(m, "CppSolverConfig")
        .def(py::init<>())
        .def_readwrite("factorization", &CppSolverConfig::factorization)
        .def_readwrite("inverse_matrix_method", &CppSolverConfig::inverse_matrix_method)
        .def_readwrite("modes_solver", &CppSolverConfig::modes_solver)
        .def_readwrite("h_simplify", &CppSolverConfig::h_simplify)
        .def_readwrite("sm_layer", &CppSolverConfig::sm_layer);

    py::class_<CppSolver>(m, "CppSolver")
        .def(py::init<const CppSolverConfig&>());
}