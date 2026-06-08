# MetaRCWA

A flexible framework for Rigorous Coupled-Wave Analysis (RCWA) simulations built on PyTorch.

## Overview

MetaRCWA provides a modular framework for 3D electromagnetic simulations using the RCWA method. It leverages PyTorch for GPU acceleration and automatic differentiation.

## Features

- **PyTorch Backend**
    - GPU acceleration
    - Automatic differentiation for inverse design and optimization

- **Advanced Numerical Methods**
    - Li factorization rules
    - Subpixel smoothing
    - Multiple eigenvalue solver options

- **Machine Learning Integration**
    - Gradient-based optimization via PyTorch autograd

## Installation

```bash
uv add git+https://github.com/RodionovSA/metarcwa.git
```

Or with pip:

```bash
pip install git+https://github.com/RodionovSA/metarcwa.git
```

### Optional extras

For dispersion models (wavelength-dependent refractive indices):

```bash
uv add git+https://github.com/RodionovSA/metarcwa.git --extra dispertorch
```

For geometry and shape utilities:

```bash
uv add git+https://github.com/RodionovSA/metarcwa.git --extra metashapes
```

Install both:

```bash
uv add git+https://github.com/RodionovSA/metarcwa.git --extra dispertorch --extra metashapes
```

### Development installation

Clone the repo and install all extras and dev dependencies:

```bash
git clone https://github.com/RodionovSA/metarcwa.git
cd metarcwa
uv sync --all-extras --dev
```

## Requirements

- Python 3.12+
- PyTorch 2.1+

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
