# Phase 5 — Background solver: comments

## What was done
- New `src/metarcwa/solver/background.py`:
  - `solve_background(model_spec, grazing_eps_reg)` → `BackgroundField`.
  - One-harmonic S-matrix solve of the background stack (patterned layers → `medium_void`),
    for a unit s and a unit p incident wave (the `Observables` s/p basis, incl. its
    `rho → 0` fallback), at the *true* incident `(kx0, ky0)` (not zeroed along PML axes).
  - Per element: `lam`, the 2x2 `V`, forward amplitude `c_plus` referenced to the left face,
    backward amplitude `c_minus` referenced to the right face (each wave referenced to the
    face where it enters, so all exponentials are bounded inside a layer).
  - `BackgroundField.waves(i)` returns the full `(Ex, Ey, Ez, Ux, Uy, Uz)` amplitudes of
    both waves (`Ez` is needed by the Phase 6 source term).
  - Same mode construction as the main solver (`homogeneous_modes`) and the same
    `grazing_eps_reg` regularization, so the background is the exact solution of the
    regularized problem.
- Amplitudes inside layer i come from the left/right partial S-matrices:
  `c+(0) = (I − R_L X R_R X)^{-1} t`, `c−(d) = R_R X c+(0)` — stable for thick/evanescent layers.
- Not wired into `run()` yet; that happens in Phase 6 together with the source vectors.

## Checks run (scratch script, not the test suite)
- Homogeneous 3-film stack, lossy film, wavelengths × θ ∈ {0, 0.4, 1.2} × φ: background
  reflection/transmission Jones matrices vs the existing periodic solver + `Observables`:
  max abs difference ~2e-15.
- Tangential field continuity at every interface: ~5e-16.
- Single interface vs Fresnel rs, rp: agree to 1e-16; `Ez` of the incident p-wave equals `sin θ / n`.

## Notes / open points
- No test files were added (per your instruction). The tests listed in PLAN.md for this phase
  (Fresnel, agreement with the homogeneous-stack solver) correspond to the checks above and
  are still to be written as pytest tests.
- `_branch_select_direction` (PLAN.md Phase 3) is not in the code: it was removed on purpose
  (PLAN.md now says so).
