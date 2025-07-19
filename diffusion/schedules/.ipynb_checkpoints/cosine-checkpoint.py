import torch
import numpy as np
from typing import Callable, Tuple


def get_cosine_schedule_functions(
    T: float = 1.0, s: float = 0.008
) -> Tuple[Callable, Callable, Callable]:
    """
    Returns functions for beta(t), alpha_bar(t), and std_dev(t) for a cosine schedule.
    Used for the Variance Preserving (VP) SDE.
    Based on the paper "Improved Denoising Diffusion Probabilistic Models" (https://arxiv.org/abs/2102.09672).

    Args:
        T: Maximum time step (typically 1.0).
        s: Small offset to prevent beta_t from being zero at t=0.

    Returns:
        Tuple[Callable, Callable, Callable]: beta_fn, alpha_bar_fn, std_dev_fn
    """
    # Precompute f(0) for normalization
    f_0 = np.cos((s / (1 + s)) * np.pi / 2) ** 2

    def alpha_bar_fn(t: torch.Tensor) -> torch.Tensor:
        """Calculates alpha_bar(t) based on the cosine schedule formula."""
        scaled_time = torch.clamp(
            t / T, 0.0, 1.0
        )  # Ensure t is scaled to [0, 1]
        f_t = torch.cos(((scaled_time + s) / (1 + s)) * (np.pi / 2)) ** 2
        # Clamp to avoid issues near t=0 and t=T, ensure monotonicity is preserved as much as possible
        alpha_bar = torch.clamp(f_t / f_0, min=1e-8, max=1.0)
        return alpha_bar

    def beta_fn(t: torch.Tensor) -> torch.Tensor:
        """
        Calculates beta(t) using the relation β(t) ≈ -d/dt ln(ᾱ(t)).
        We use numerical differentiation for robustness.
        Alternatively, the discrete definition β̃ₜ = min(1 - ᾱₜ / ᾱₜ₋₁, 0.999) can be used.
        Let's use numerical differentiation.
        """
        dt = 1e-6  # Small step for numerical differentiation
        # Clamp input t to be within valid range [0, T]
        t_clamped = torch.clamp(t, 0.0, T)

        alpha_bar_t = alpha_bar_fn(t_clamped)
        # Calculate alpha_bar at t+dt, ensuring not to exceed T
        alpha_bar_t_plus_dt = alpha_bar_fn(torch.clamp(t_clamped + dt, max=T))

        # Approximate beta using log derivative: beta(t) approx - (log(alpha_bar(t+dt)) - log(alpha_bar(t))) / dt
        # Use clamp to prevent log(0)
        log_alpha_bar_t = torch.log(torch.clamp(alpha_bar_t, min=1e-10))
        log_alpha_bar_t_plus_dt = torch.log(
            torch.clamp(alpha_bar_t_plus_dt, min=1e-10)
        )

        # Handle potential division by zero if dt is too small (shouldn't happen with 1e-6)
        beta = -(log_alpha_bar_t_plus_dt - log_alpha_bar_t) / max(dt, 1e-10)

        # Clamp beta to prevent instability, following DDPM practice [0.0001, 0.999]
        # Using a slightly wider range based on common impls [1e-7, 0.999]
        beta = torch.clamp(beta, min=1e-7, max=0.999)
        return beta

    def std_dev_fn(t: torch.Tensor) -> torch.Tensor:
        """Calculates the standard deviation σ(t) = sqrt(1 - ᾱ(t))."""
        alpha_bar_t = alpha_bar_fn(t)
        # Add epsilon for numerical stability, esp. near t=0
        return torch.sqrt(torch.clamp(1.0 - alpha_bar_t, min=1e-8))

    return beta_fn, alpha_bar_fn, std_dev_fn
