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
            position: Tensor of shape (dim,) representing position vector
            
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
        x, y = position[0], position[1]
        ax = -x - 2.0 * x * y
        ay = -y - x**2 + y**2
        return torch.stack([ax, ay])
    
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
