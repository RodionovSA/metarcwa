# tests/pml_scattered_validation_bench.py
"""Validation of the PML scattered-field formulation (PLAN.md phase 7).

Not a pytest module (``*_bench.py`` is not collected): run it directly,

    python tests/pml_scattered_validation_bench.py

Test structure: a dielectric ridge (eps 4, width 0.5, height 0.3) in air on a
glass substrate (eps 2.25), wavelength 0.8, uniform along y, PML along x.

Checks:

1. Normal incidence: scattered-field = total-field formulation
   ({eig, matexp} x {no TVF, TVF}), expected ~1e-14.
2. Tilt along the periodic (non-PML) axis y: both formulations are valid,
   expected ~1e-14.
3. Tilt along the PML axis x (theta = 0.4): the specular scattering amplitude
   per unit length, ``L * (J - J_bg)``, converges with the cell size towards
   the large periodic supercell result (``Lp * (J_00 - J_bg)``, Lp = 24).
"""

import math
import time

import torch

from metarcwa import (Config, Factorization, IsotropicMedium, Lattice, Layer, Model,
                      Observables, PML, Solver, Source, Stack)
from metarcwa.model.nn_helpers import CallableModule

torch.set_default_dtype(torch.float64)
WL, EPS_R, W_R, H_R, EPS_SUB = 0.8, 4.0, 0.5, 0.3, 2.25


def medium(eps):
    return IsotropicMedium(CallableModule(lambda wl: torch.full_like(wl, eps, dtype=torch.complex128)))


def ridge(L):
    def shape_fn(lattice, nx, ny):
        x = ((torch.arange(nx) + 0.5) / nx - 0.5) * L
        return (x.abs() < W_R / 2).double()[None, :].expand(ny, nx)
    return CallableModule(shape_fn)


def model(L, theta=0.0, phi=0.0, pml=None, film=False):
    layers = [Layer(medium_solid=medium(EPS_R), medium_void=medium(1.0), shape_fn=ridge(L), thickness=H_R)]
    if film:
        layers.append(Layer(medium_solid=medium(2.0), thickness=0.1))
    stack = Stack(incidence=medium(1.0), layers=layers, transmission=medium(EPS_SUB),
                  lattice=Lattice.rectangular(L, 0.5), pml=pml)
    return Model(stack, Source(wavelength=torch.tensor(WL), theta=theta, phi=phi))


def config(L, m=None, **kw):
    m = m or round(5 * L)
    base = dict(dtype=torch.float64, nx=max(8 * m, 64), ny=4, m=m, n=0,
                truncation="rectangular", factorization=None)
    base.update(kw)
    return Config(**base)


def check_normal_incidence():
    L = 2.0
    for ms in ("matexp", "eig"):
        for tvf in (None, Factorization()):
            kw = dict(m=12, nx=128, modesolver=ms, factorization=tvf)
            tot = Observables(Solver(model(L, pml=PML(width=(0.5, 0.0)), film=True),
                                     config(L, pml_formulation="total", **kw)).run())
            sca = Observables(Solver(model(L, pml=PML(width=(0.5, 0.0)), film=True),
                                     config(L, pml_formulation="scattered", **kw)).run())
            for kind in ("reflection", "transmission"):
                err = (tot.jones(kind) - sca.jones(kind)).abs().max().item()
                print(f"normal incidence  {ms:6s} tvf={tvf is not None!s:5s} {kind:12s} |dJ| = {err:.1e}")


def check_tilt_along_periodic_axis():
    L = 2.0
    for theta in (0.3, 0.7):
        tot = Observables(Solver(model(L, theta, math.pi / 2, PML(width=(0.5, 0.0))),
                                 config(L, m=14, pml_formulation="total")).run())
        sca = Observables(Solver(model(L, theta, math.pi / 2, PML(width=(0.5, 0.0))),
                                 config(L, m=14)).run())
        for kind in ("reflection", "transmission"):
            err = (tot.jones(kind) - sca.jones(kind)).abs().max().item()
            print(f"tilt along y theta={theta}  {kind:12s} |dJ| = {err:.1e}")


def check_tilt_along_pml_axis(theta=0.4):
    Lp = 24.0
    per = Observables(Solver(model(Lp, theta), config(Lp)).run())
    ref = {}
    for kind in ("reflection", "transmission"):
        bg = Observables(Solver(model(4.0, theta, pml=PML(width=(0.75, 0.0))), config(4.0)).run())
        ref[kind] = (per.jones(kind) - bg.background_jones(kind)) * Lp
    for L, w in ((4.0, 0.75), (8.0, 1.5), (12.0, 1.5), (16.0, 2.0), (16.0, 3.0)):
        t = time.time()
        obs = Observables(Solver(model(L, theta, pml=PML(width=(w, 0.0))), config(L)).run())
        errs = []
        for kind in ("reflection", "transmission"):
            d = (obs.jones(kind) - obs.background_jones(kind)) * L
            errs.append((d - ref[kind]).diagonal(dim1=-2, dim2=-1).abs().max().item())
        print(f"tilt along x theta={theta} L={L:4.1f} w={w}: |L dJ - Lp dJ_periodic| "
              f"R {errs[0]:.4f}  T {errs[1]:.4f}  ({time.time() - t:.1f}s)")


if __name__ == "__main__":
    check_normal_incidence()
    check_tilt_along_periodic_axis()
    check_tilt_along_pml_axis()
