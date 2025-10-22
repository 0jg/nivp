"""
Helper utilities for neural IVP solving.

Provides:
- Filesystem management (directory structure, run caching)
- Numerical integration (RK4 for ground truth solutions)
- Torch utilities (derivatives, coupled optimisers)
- Neural network architectures with custom activations
"""

from __future__ import annotations

import os
import json
import math
import hashlib
import torch
from typing import Tuple, Dict, Any
import torch.nn as nn
import torch.nn.functional as F

DATA_TYPE: torch.dtype = torch.float
DEVICE: torch.device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

# ---------- Filesystem helpers ----------
def ensure_run_dirs(
    base: str, 
    params: Dict[str, Any], 
    system_name: str
) -> Dict[str, str]:
    """Create run-specific directories with a hash of the params dictionary.
    
    Args:
        base: Base output directory path
        params: Parameter dictionary to hash for run identification
        system_name: Name of the dynamical system (e.g., 'henon', 'lorenz')
        
    Returns:
        Dictionary with keys 'figures', 'paths', 'models' pointing to subdirectories
    """
    base = base.rstrip("/")
    # Create a hash of the params dictionary
    params_str: str = json.dumps(params, sort_keys=True)
    params_hash: str = hashlib.md5(params_str.encode()).hexdigest()[:8]
    run_name: str = f"{system_name}_{params_hash}"
    
    figures: str = os.path.join(base, "figures", run_name)
    paths: str = os.path.join(base, "paths", run_name)
    models: str = os.path.join(base, "models", run_name)
    os.makedirs(figures, exist_ok=True)
    os.makedirs(paths, exist_ok=True)
    os.makedirs(models, exist_ok=True)
    return {"figures": figures, "paths": paths, "models": models}


# ---------- Physics helpers ----------
def rk4_integrate(
    system: Any,  # DynamicalSystem
    x0: torch.Tensor,
    v0: torch.Tensor,
    t_min: float,
    t_max: float,
    dt: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Integrate a second-order ODE system using RK4.

    Solves d^2x/dt^2 = a(x) with initial conditions x(t_min) = x0, dx/dt(t_min) = v0.

    Args:
        system: DynamicalSystem instance with acceleration() method
        x0: Initial position, shape (dim,)
        v0: Initial velocity, shape (dim,)
        t_min: Start time
        t_max: End time
        dt: Time step
        
    Returns:
        Tuple of (t, positions, velocities) all shape compatible with time dimension T
    """
    dim: int = system.dim
    T: int = int(round((t_max - t_min) / dt)) + 1
    t: torch.Tensor = torch.linspace(t_min, t_max, T, dtype=DATA_TYPE, device=DEVICE)
    A: torch.Tensor = torch.zeros((dim, T), dtype=DATA_TYPE, device=DEVICE)
    A_t: torch.Tensor = torch.zeros((dim, T), dtype=DATA_TYPE, device=DEVICE)
    A[:, 0] = x0.to(DEVICE)
    A_t[:, 0] = v0.to(DEVICE)

    for i in range(0, T - 1):
        k1y: torch.Tensor = dt * A_t[:, i]
        k1v: torch.Tensor = dt * system.acceleration(A[:, i])

        k2y: torch.Tensor = dt * (A_t[:, i] + 0.5 * k1v)
        k2v: torch.Tensor = dt * system.acceleration(A[:, i] + 0.5 * k1y)

        k3y: torch.Tensor = dt * (A_t[:, i] + 0.5 * k2v)
        k3v: torch.Tensor = dt * system.acceleration(A[:, i] + 0.5 * k2y)

        k4y: torch.Tensor = dt * (A_t[:, i] + k3v)
        k4v: torch.Tensor = dt * system.acceleration(A[:, i] + k3y)

        A[:, i+1] = A[:, i] + (k1y + 2*k2y + 2*k3y + k4y) / 6.0
        A_t[:, i+1] = A_t[:, i] + (k1v + 2*k2v + 2*k3v + k4v) / 6.0

    return t, A, A_t


def rk4_integrate_first_order(
    system: Any,  # FirstOrderSystem
    M0: torch.Tensor,
    t_min: float,
    t_max: float,
    dt: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Integrate a first-order ODE system using RK4.

    Solves dM/dt = F(M) with initial condition M(t_min) = M0.

    Args:
        system: FirstOrderSystem instance with time_derivative() method
        M0: Initial state, shape (dim,)
        t_min: Start time
        t_max: End time
        dt: Time step
        
    Returns:
        Tuple of (t, state) where state has shape (dim, T)
    """
    dim: int = system.dim
    T: int = int(round((t_max - t_min) / dt)) + 1
    t: torch.Tensor = torch.linspace(t_min, t_max, T, dtype=DATA_TYPE, device=DEVICE)
    M: torch.Tensor = torch.zeros((dim, T), dtype=DATA_TYPE, device=DEVICE)
    M[:, 0] = M0.to(DEVICE)

    for i in range(0, T - 1):
        k1: torch.Tensor = dt * system.time_derivative(M[:, i])
        k2: torch.Tensor = dt * system.time_derivative(M[:, i] + 0.5 * k1)
        k3: torch.Tensor = dt * system.time_derivative(M[:, i] + 0.5 * k2)
        k4: torch.Tensor = dt * system.time_derivative(M[:, i] + k3)

        M[:, i+1] = M[:, i] + (k1 + 2*k2 + 2*k3 + k4) / 6.0

    return t, M


def expected_path_tensor(
    system: Any,  # DynamicalSystem
    x0: torch.Tensor,
    v0: torch.Tensor,
    t_min: float,
    t_max: float,
    dt: float,
) -> torch.Tensor:
    """Compute ground-truth solution via RK4 integration.
    
    Args:
        system: DynamicalSystem instance
        x0: Initial position
        v0: Initial velocity  
        t_min: Start time
        t_max: End time
        dt: Time step
        
    Returns:
        Solution tensor reshaped to (1, T, dim) for batch processing
    """
    t: torch.Tensor
    A: torch.Tensor
    A_t: torch.Tensor
    t, A, A_t = rk4_integrate(system, x0, v0, t_min, t_max, dt)
    # Reshape to (1, T, dim) for compatibility with batch predictions
    solution: torch.Tensor = A.transpose(0, 1).unsqueeze(0)
    return solution


def expected_path_tensor_first_order(
    system: Any,  # FirstOrderSystem
    M0: torch.Tensor,
    t_min: float,
    t_max: float,
    dt: float,
) -> torch.Tensor:
    """Compute ground-truth solution via RK4 integration for first-order systems.
    
    Args:
        system: FirstOrderSystem instance
        M0: Initial state
        t_min: Start time
        t_max: End time
        dt: Time step
        
    Returns:
        Solution tensor reshaped to (1, T, dim) for batch processing
    """
    t: torch.Tensor
    M: torch.Tensor
    t, M = rk4_integrate_first_order(system, M0, t_min, t_max, dt)
    # Reshape to (1, T, dim) for compatibility with batch predictions
    solution: torch.Tensor = M.transpose(0, 1).unsqueeze(0)
    return solution



# ---------- Torch helpers ----------
def gradient(
    y: torch.Tensor, 
    x: torch.Tensor, 
    derivative: int
) -> torch.Tensor:
    """Compute derivatives of y with respect to x using autograd.
    
    Args:
        y: Output tensor (typically model output)
        x: Input tensor with requires_grad=True
        derivative: Order of derivative (1 for dy/dx, 2 for d^2y/dx^2)
        
    Returns:
        Tensor of same shape as y containing the derivatives
        
    Raises:
        NotImplementedError: If derivative > 2
    """
    weights: torch.Tensor = torch.ones_like(y)
    f_x: torch.Tensor = torch.autograd.grad(
        outputs=y, inputs=x, grad_outputs=weights,
        retain_graph=True, create_graph=True, allow_unused=True
    )[0]

    if derivative == 1:
        return f_x
    elif derivative == 2:
        f_xx: torch.Tensor = torch.autograd.grad(
            outputs=f_x, inputs=x, grad_outputs=weights,
            retain_graph=True, create_graph=True, allow_unused=True
        )[0]
        return f_xx
    else:
        raise NotImplementedError("Only 1st and 2nd derivatives supported.")


class CoupledOptimiser:
    """Manager for multiple optimisers to be stepped together.
    
    Useful when training coupled networks (e.g., separate networks for each coordinate).
    """
    
    def __init__(self, *optimisers: torch.optim.Optimizer) -> None:
        """Initialise with one or more optimisers.
        
        Args:
            *optimisers: Variable number of torch.optim.Optimizer instances
        """
        self.optimisers: Tuple[torch.optim.Optimizer, ...] = optimisers

    def zero_grad(self) -> None:
        """Zero gradients in all optimisers."""
        for op in self.optimisers:
            op.zero_grad()

    def step(self) -> None:
        """Perform optimisation step in all optimisers."""
        for op in self.optimisers:
            op.step()


def second_order_dynamics(
    system: Any,
    positions: torch.Tensor,
    t: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute velocities, model accelerations, and residuals for q'' = a(q).
    
    Args:
        system: DynamicalSystem with acceleration() method
        positions: Predicted positions q(t) with trailing dimension equal to system.dim
        t: Input time tensor matching positions except for the last dimension
    
    Returns:
        Tuple of tensors (velocities, model_accelerations, residuals), each shaped like positions.
        Residuals follow d^2q/dt^2 - system.acceleration(q).
    """
    if positions.shape[-1] != system.dim:
        raise ValueError(
            f"Expected positions last dimension {system.dim}, got {positions.shape[-1]}"
        )
    if positions.shape[:-1] != t.shape[:-1]:
        raise ValueError("positions and t must share the same leading dimensions.")

    system_accel: torch.Tensor = system.acceleration(positions)
    velocities: list[torch.Tensor] = []
    model_accels: list[torch.Tensor] = []
    residuals: list[torch.Tensor] = []

    for idx in range(system.dim):
        coord: torch.Tensor = positions[..., idx:idx+1]
        vel: torch.Tensor = gradient(coord, t, 1)
        acc_model: torch.Tensor = gradient(coord, t, 2)
        velocities.append(vel)
        model_accels.append(acc_model)
        residuals.append(acc_model - system_accel[..., idx:idx+1])

    return (
        torch.cat(velocities, dim=-1),
        torch.cat(model_accels, dim=-1),
        torch.cat(residuals, dim=-1),
    )


def first_order_dynamics(
    system: Any,
    state: torch.Tensor,
    t: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute model time derivatives and residuals for dM/dt = F(M).
    
    Args:
        system: FirstOrderSystem with time_derivative() method
        state: Predicted state M(t) with trailing dimension equal to system.dim
        t: Input time tensor matching state except for the last dimension
    
    Returns:
        Tuple of tensors (model_derivatives, residuals), each shaped like state.
        Residuals follow dM/dt - system.time_derivative(M).
    """
    if state.shape[-1] != system.dim:
        raise ValueError(
            f"Expected state last dimension {system.dim}, got {state.shape[-1]}"
        )
    if state.shape[:-1] != t.shape[:-1]:
        raise ValueError("state and t must share the same leading dimensions.")

    system_deriv: torch.Tensor = system.time_derivative(state)
    model_derivs: list[torch.Tensor] = []
    residuals: list[torch.Tensor] = []

    for idx in range(system.dim):
        component: torch.Tensor = state[..., idx:idx+1]
        deriv_model: torch.Tensor = gradient(component, t, 1)
        model_derivs.append(deriv_model)
        residuals.append(deriv_model - system_deriv[..., idx:idx+1])

    return (
        torch.cat(model_derivs, dim=-1),
        torch.cat(residuals, dim=-1),
    )


# ---------- Activations & model ----------


def sechlu(x: torch.Tensor) -> torch.Tensor:
    """Sech-LU activation: sigmoid(2x) * x.
    
    A smooth activation function combining sigmoid gating with linear term.
    """
    return torch.sigmoid(2*x) * x


def sechlu_k(x: torch.Tensor, k: float) -> torch.Tensor:
    """Scaled Sech-LU activation: sigmoid(2x/k) * x.
    
    Args:
        x: Input tensor
        k: Scaling factor for sigmoid steepness
    """
    return torch.sigmoid(2*x/float(k)) * x


def cauchylu(x: torch.Tensor, x0: float = 0.0, gamma: float = 0.5) -> torch.Tensor:
    """Cauchy-LU activation: x * (1/π * arctan((x-x0)/gamma) + 1/2).
    
    Smooth activation based on Cauchy distribution CDF.
    
    Args:
        x: Input tensor
        x0: Center of Cauchy distribution
        gamma: Scale parameter
    """
    return x * ((1.0/math.pi) * torch.atan((x - x0)/gamma) + 0.5)


class SimpleMLP(nn.Module):
    """Simple multi-layer perceptron with configurable activation.
    
    Architecture: in_features -> hidden -> hidden -> hidden -> out_features
    with specified activation function applied between hidden layers.
    """
    
    def __init__(
        self, 
        in_features: int, 
        hidden: int, 
        out_features: int, 
        activation: str
    ) -> None:
        """Initialize MLP.
        
        Args:
            in_features: Input dimension
            hidden: Hidden layer dimension (same for all hidden layers)
            out_features: Output dimension
            activation: Name of activation function (e.g., 'sechlu', 'tanh', 'relu')
        """
        super().__init__()
        self.L1: nn.Linear = nn.Linear(in_features, hidden)
        self.L2: nn.Linear = nn.Linear(hidden, hidden)
        self.L3: nn.Linear = nn.Linear(hidden, hidden)
        self.L4: nn.Linear = nn.Linear(hidden, out_features)
        self.activation: str = activation

    def _act(self, x: torch.Tensor) -> torch.Tensor:
        """Apply activation function.
        
        Args:
            x: Input tensor
            
        Returns:
            Activated tensor
        """
        a: str = self.activation.lower()
        if a == "sigmoid":
            return torch.sigmoid(x)
        if a == "tanh":
            return torch.tanh(x)
        if a == "relu":
            return F.relu(x)
        if a == "gelu":
            return F.gelu(x)
        if a == "sechlu":
            return sechlu(x)
        if a == "sechlu8":
            return sechlu_k(x, 8.0)
        if a == "sechlu4":
            return sechlu_k(x, 4.0)
        if a == "sechlu05":
            return sechlu_k(x, 0.5)
        if a == "sechlu01":
            return sechlu_k(x, 0.1)
        if a == "cauchylu":
            return cauchylu(x)
        return F.relu(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through network.
        
        Args:
            x: Input tensor of shape (..., in_features)
            
        Returns:
            Output tensor of shape (..., out_features)
        """
        x = self._act(self.L1(x))
        x = self._act(self.L2(x))
        x = self._act(self.L3(x))
        x = self.L4(x)
        return x
