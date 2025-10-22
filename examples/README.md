# Neural IVP Examples

This directory contains example applications of the neural IVP solver to various dynamical systems.

## Available Examples

### `harmonic_oscillator.py`

The simplest example: a 1D harmonic oscillator with equation d²x/dt² = -x.

With initial conditions x(0) = 0, v(0) = 1, the analytical solution is x(t) = sin(t).

**To run:**

```bash
uv run marimo edit harmonic_oscillator.py
```

### `henon_heiles.py`

Describes the motion of a star about its galactic centre, akin to a 3-body problem.

The example includes:

- **Chaotic trajectory** (E=1/6)
- **Quasi-periodic trajectory** (E=1/12)
- Configurable training parameters and network architecture
- Real-time visualization during training
- Results saved to `./outputs/`

**To run:**

```bash
uv run marimo edit henon_heiles.py
```

Or batch run:

```bash
uv run marimo run henon_heiles.py
```

### `landau_lifschitz.py`

Models magnetization dynamics in ferromagnetic materials using the Landau-Lifschitz equation.

This is a **first-order** system (unlike the above second-order systems) with equation:

**dM/dt = -M × H - α M × (M × H)**

where:
- M is the dimensionless magnetization vector (mx, my, mz)
- H = H₀ẑ is the applied magnetic field along the z-axis
- α is the Gilbert damping parameter

With initial condition M(0) = x̂, the magnetization precesses around the z-axis while damping toward equilibrium.

Features:
- Three separate neural networks for mx, my, mz components
- 3D phase space visualization
- Comparison with RK4 ground truth

**To run:**

```bash
uv run marimo edit landau_lifschitz.py
```

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

1. Define the system in `../systems.py` by subclassing `DynamicalSystem`
2. Create a new notebook in this directory
3. Import the system and use it with the generic helpers

Example template:

```python
from systems import MyNewSystem
from helpers import rk4_integrate, expected_path_tensor

system = MyNewSystem()
x0 = torch.tensor([...])
v0 = torch.tensor([...])
expected = expected_path_tensor(system, x0, v0, t_min, t_max, dt)
```

