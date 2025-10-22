"""
Dynamical systems for the neural IVP codebase
"""

from __future__ import annotations

import torch
from abc import ABC, abstractmethod


class DynamicalSystem(ABC):
    """Abstract base class (ABC) for dynamical systems.
    
    A dynamical system is defined by:
    - State dimension
    - Acceleration function (for second-order systems)
    - Optional Lagrangian and energy functions
    """
    
    @property
    @abstractmethod
    def dim(self) -> int:
        """Spatial dimension of the system (e.g., 2 for 2D Hénon-Heiles)."""
        pass
    
    @abstractmethod
    def acceleration(self, position: torch.Tensor) -> torch.Tensor:
        """Compute acceleration given position.
        
        Args:
            position: Tensor whose trailing dimension equals `dim`
                (e.g., shape (dim,) or (..., dim))
            
        Returns:
            Tensor of shape (dim,) with acceleration components
        """
        pass
    
    def lagrangian(
        self, 
        position: torch.Tensor, 
        velocity: torch.Tensor
    ) -> torch.Tensor:
        """Compute Lagrangian L = T - V at given state.
        
        Args:
            position: Tensor of shape (dim,)
            velocity: Tensor of shape (dim,)
            
        Returns:
            Scalar tensor
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement lagrangian()"
        )
    
    def energy(
        self, 
        position: torch.Tensor, 
        velocity: torch.Tensor
    ) -> torch.Tensor:
        """Compute total energy at given state.
        
        Args:
            position: Tensor of shape (dim,)
            velocity: Tensor of shape (dim,)
            
        Returns:
            Scalar tensor
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement energy()"
        )


class HarmonicOscillator(DynamicalSystem):
    """The 1D harmonic oscillator system.
    
    A simple 1-degree-of-freedom system with equation:
    d^2x/dt^2 = -x
    
    This represents a mass on a spring with potential V(x) = 1/2 * x^2.
    With initial conditions x(0) = 0, v(0) = 1, the analytical solution is x(t) = sin(t).
    """
    
    @property
    def dim(self) -> int:
        return 1
    
    def acceleration(self, position: torch.Tensor) -> torch.Tensor:
        """Acceleration in harmonic oscillator system.

        Returns: d^2x/dt^2 = -x
        """
        return -position
    
    def lagrangian(
        self,
        position: torch.Tensor,
        velocity: torch.Tensor
    ) -> torch.Tensor:
        """Lagrangian of harmonic oscillator system.

        L = T - V = 1/2 * \dot{x}^2 - 1/2 * x^2
        """
        x = position[0]
        vx = velocity[0]
        
        kinetic_energy = 0.5 * vx**2
        potential_energy = 0.5 * x**2
        return kinetic_energy - potential_energy
    
    def energy(
        self,
        position: torch.Tensor,
        velocity: torch.Tensor
    ) -> torch.Tensor:
        """Total energy of harmonic oscillator system.

        E = T + V = 1/2 * \dot{x}^2 + 1/2 * x^2
        """
        x = position[0]
        vx = velocity[0]
        
        kinetic_energy = 0.5 * vx**2
        potential_energy = 0.5 * x**2
        return kinetic_energy + potential_energy


class HenonHeiles(DynamicalSystem):
    """The Hénon-Heiles system.
    
    A 2-degree-of-freedom Hamiltonian system with potential:
    V(x,y) = 1/2(x^2 + y^2) + x^2y - y^3/3

    The system exhibits chaotic and quasi-periodic behavior depending on energy.
    """
    
    @property
    def dim(self) -> int:
        return 2
    
    def acceleration(self, position: torch.Tensor) -> torch.Tensor:
        """Acceleration in Hénon-Heiles system.

        Returns: d^2x/dt^2 = -dV/dx, d^2y/dt^2 = -dV/dy
        """
        x = position[..., 0]
        y = position[..., 1]
        ax = -x - 2.0 * x * y
        ay = -y - x**2 + y**2
        return torch.stack([ax, ay], dim=-1)
    
    def lagrangian(
        self,
        position: torch.Tensor,
        velocity: torch.Tensor
    ) -> torch.Tensor:
        """Lagrangian of Hénon-Heiles system.

        L = T - V = 1/2(\dot{x}^2 + \dot{y}^2) - V(x,y)
        """
        q1, q2 = position[0], position[1]
        dq1, dq2 = velocity[0], velocity[1]
        
        kinetic_energy = 0.5 * (dq1**2 + dq2**2)
        potential_energy = 0.5 * (q1**2 + q2**2) + q1**2 * q2 - q2**3 / 3.0
        return kinetic_energy - potential_energy
    
    def energy(
        self,
        position: torch.Tensor,
        velocity: torch.Tensor
    ) -> torch.Tensor:
        """Total energy of Hénon-Heiles system.

        E = T + V = 1/2(\dot{x}^2 + \dot{y}^2) + V(x,y)
        """
        q1, q2 = position[0], position[1]
        dq1, dq2 = velocity[0], velocity[1]
        
        kinetic_energy = 0.5 * (dq1**2 + dq2**2)
        potential_energy = 0.5 * (q1**2 + q2**2) + q1**2 * q2 - q2**3 / 3.0
        return kinetic_energy + potential_energy


class FirstOrderSystem(ABC):
    """Abstract base class for first-order dynamical systems.
    
    A first-order system is defined by:
    - State dimension
    - Time derivative function dM/dt = F(M, t)
    """
    
    @property
    @abstractmethod
    def dim(self) -> int:
        """Dimension of the state vector."""
        pass
    
    @abstractmethod
    def time_derivative(self, state: torch.Tensor) -> torch.Tensor:
        """Compute time derivative dM/dt given current state.
        
        Args:
            state: Tensor whose trailing dimension equals `dim`
                (e.g., shape (dim,) or (..., dim))
            
        Returns:
            Tensor of shape (..., dim) with time derivatives
        """
        pass


class LandauLifschitz(FirstOrderSystem):
    """The Landau-Lifschitz equation for magnetisation dynamics.
    
    In dimensionless form:
    dM/dt = -M x H - alpha M x (M x H)
    
    where M is the dimensionless magnetisation vector, H is the applied field, and α is the damping parameter.
    
    For H = H0 e_z (field along z-axis) and initial condition M(0) = e_x, this describes precession and damping of the magnetisation vector.
    """
    
    def __init__(self, alpha: float = 0.1, H0: float = 1.0):
        """Initialise Landau-Lifschitz system.
        
        Args:
            alpha: damping parameter (dimensionless)
            H0: Applied field strength in z-direction (dimensionless)
        """
        self.alpha = alpha
        self.H0 = H0
    
    @property
    def dim(self) -> int:
        return 3
    
    def time_derivative(self, M: torch.Tensor) -> torch.Tensor:
        """Compute dM/dt for the LL equation.

        dM/dt = -M x H - alpha M x (M x H)

        where H = H0 e_z
        
        Args:
            M: Magnetisation vector, shape (..., 3)
            
        Returns:
            Time derivative dM/dt, shape (..., 3)
        """
        # H = H0 * ẑ = (0, 0, H0)
        H = torch.zeros_like(M)
        H[..., 2] = self.H0
        
        # Compute M x H using PyTorch's cross product
        M_cross_H = torch.cross(M, H, dim=-1)
        
        # Compute M x (M x H)
        M_cross_MH = torch.cross(M, M_cross_H, dim=-1)

        # dM/dt = -M x H - alpha M x (M x H)
        dM_dt = -M_cross_H - self.alpha * M_cross_MH
        
        return dM_dt
