# Neural IVP Examples

This directory contains example applications of the neural IVP solver to various dynamical systems.

## Available Examples

### `harmonic_oscillator.ipynb`

The simplest example: a 1D harmonic oscillator with equation d^2x/dt^2 = -x.

With initial conditions x(0) = 0, v(0) = 1, the analytical solution is x(t) = sin(t).

### `henon_heiles.ipynb`

Describes the motion of a star about its galactic centre, akin to a 3-body problem.

The example includes:

- **Chaotic trajectory** (E=1/6)
- **Quasi-periodic trajectory** (E=1/12)
- Configurable training parameters and network architecture
- Real-time visualization during training
- Results saved to `./outputs/`

### `landau_lifschitz.py`

Models magnetisation dynamics in ferromagnetic materials using the Landau-Lifschitz equation.

This is a **first-order** system (unlike the above second-order systems) with equation:

`dM/dt = -M x H - alpha M x (M x H)`

where:

- M is the dimensionless magnetisation vector (mx, my, mz)
- H = H_0 e_z is the applied magnetic field along the z-axis
- alpha is the damping parameter

With initial condition M(0) = e_x, the magnetisation precesses around the z-axis while damping toward equilibrium.

Features:

- Three separate neural networks for mx, my, mz components
- 3D phase space visualization
- Comparison with RK4 ground truth

## Output Structure

Results are automatically organised in `./outputs/`:

```text
outputs/
├── figures/      # Training progress plots
├── paths/        # Loss histories (JSON)
└── models/       # Trained model checkpoints
```

Subdirectories are named with the system name and a parameter hash to identify unique runs based on the `params` dictionary:

- Example: `henon_heiles_a1b2c3d4/`

## Adding New Examples

To add a new dynamical system:

1. Define the system directly in the example file that uses it
2. Create a new notebook in this directory
3. Reuse the generic helpers from `../helpers.py` for RK4 references and residual calculations

Example template:

```python
from helpers import rk4_integrate, expected_path_tensor

class MyNewSystem:
    dim = 1

    def acceleration(self, position):
        return -position

system = MyNewSystem()
x0 = torch.tensor([...])
v0 = torch.tensor([...])
expected = expected_path_tensor(system, x0, v0, t_min, t_max, dt)
```

Keep system-specific physics definitions in the example that uses them.
