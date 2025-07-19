import torch
import numpy as np
from typing import Callable, Tuple


def get_cosine_schedule_functions(
    T: float = 1.0,
    s: float = 0.008,
    beta_min: float = 1e-7,
    beta_max: float = 0.999,
) -> Tuple[Callable, Callable, Callable]:
    """
    Returns functions for beta(t), alpha_bar(t), and std_dev(t) for a cosine schedule.
    Used for the Variance Preserving (VP) SDE.
    Based on the paper "Improved Denoising Diffusion Probabilistic Models"
    (https://arxiv.org/abs/2102.09672).
    Restored standard normalized alpha_bar, analytical beta, T scaling.

    Args:
        T: Maximum time step (typically 1.0).
        s: Small offset to prevent beta_t from being zero at t=0.
        beta_min: Minimum value to clamp beta(t) to.
        beta_max: Maximum value to clamp beta(t) to.

    Returns:
        Tuple[Callable, Callable, Callable]: beta_fn, alpha_bar_fn, std_dev_fn
    """
    # Restore f(0) calculation for normalization
    f_0_angle = (s / (1 + s)) * (np.pi / 2)
    f_0 = np.cos(f_0_angle) ** 2
    pi = np.pi  # Use numpy pi

    def alpha_bar_fn(t: torch.Tensor) -> torch.Tensor:
        """Calculates alpha_bar(t) based on the cosine schedule formula (Normalized)."""
        # Restore T scaling and clamping logic
        t_clamped = torch.clamp(t, 0.0, T)
        scaled_time = t_clamped / T  # Ensure t is scaled to [0, 1]

        angle = ((scaled_time + s) / (1 + s)) * (pi / 2)
        angle = torch.clamp(
            angle, max=pi / 2 - 1e-7
        )  # Restore internal angle clamp

        f_t = torch.cos(angle) ** 2
        # Restore normalization and final alpha_bar clamp
        alpha_bar = torch.clamp(f_t / f_0, min=1e-8, max=1.0)
        return alpha_bar

    def beta_fn(t: torch.Tensor) -> torch.Tensor:
        """
        Calculates beta(t) using the analytical derivative of log(alpha_bar(t)).
        beta(t) = tan(f(t)) * (pi / (T * (1 + s)))
        where f(t) = ((t/T + s) / (1 + s)) * (pi / 2)
        """
        # Restore analytical tan-based calculation with T scaling
        t_clamped = torch.clamp(t, 0.0, T)
        scaled_time = t_clamped / T  # Ensure t is scaled to [0, 1]

        angle = ((scaled_time + s) / (1 + s)) * (pi / 2)
        angle = torch.clamp(
            angle, min=1e-7, max=pi / 2 - 1e-7
        )  # Restore angle clamp

        tan_angle = torch.tan(angle)
        factor = pi / (T * (1 + s))
        beta = tan_angle * factor

        # Apply final beta clamp using provided args
        beta = torch.clamp(beta, min=beta_min, max=beta_max)
        return beta

    def std_dev_fn(t: torch.Tensor) -> torch.Tensor:
        """Calculates the standard deviation \sigma(t) = sqrt(1 - \bar{\alpha}(t))."""
        alpha_bar_t = alpha_bar_fn(t)
        # Add epsilon for numerical stability, esp. near t=0
        return torch.sqrt(torch.clamp(1.0 - alpha_bar_t, min=1e-8))

    return beta_fn, alpha_bar_fn, std_dev_fn
