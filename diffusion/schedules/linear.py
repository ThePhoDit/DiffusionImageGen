import torch
from typing import Callable, Tuple


def get_linear_schedule_functions(
    beta_min: float = 0.01, beta_max: float = 0.95, T: float = 1.0
) -> Tuple[Callable, Callable, Callable]:
    """
    Returns functions for beta(t), alpha_bar(t), and std_dev(t) for a linear schedule.
    Used for the Variance Preserving (VP) SDE.

    Args:
        beta_min: Starting value of beta.
        beta_max: Ending value of beta.
        T: Maximum time step (typically 1.0).

    Returns:
        Tuple[Callable, Callable, Callable]: beta_fn, alpha_bar_fn, std_dev_fn
    """

    def beta_fn(t: torch.Tensor) -> torch.Tensor:
        """Linearly increasing beta schedule:
        \beta(t) = \beta_min + (\beta_max - \beta_min) * t / T"""
        # Ensure t is scaled correctly if T is not 1.0
        scaled_t = torch.clamp(t / T, 0.0, 1.0)
        return beta_min + (beta_max - beta_min) * scaled_t

    def alpha_bar_fn(t: torch.Tensor) -> torch.Tensor:
        """
        Calculates alpha_bar(t) = exp(-\int_{0}^{t} \beta(s) ds).
        Uses the integral approximation for continuous time.
        \int_{0}^{t} \beta(s) ds = \int_{0}^{t} (\beta_min +
                    (\beta_max - \beta_min) * s / T) ds
                    = \beta_min * t + (\beta_max - \beta_min) / T * (t^2/2)
        \bar{\alpha}(t) = exp(-\int_{0}^{t} \beta(s) ds)
        """
        # Ensure t is clamped [0, T]
        t_clamped = torch.clamp(t, 0.0, T)
        integral_beta = (
            beta_min * t_clamped
            + 0.5 * (beta_max - beta_min) * (t_clamped**2) / T
        )
        # Clamp the exponent input to avoid potential overflow/underflow with large integrals
        integral_beta_clamped = torch.clamp(
            integral_beta, -20, 20
        )  # Heuristic clamp
        alpha_bar = torch.exp(-integral_beta_clamped)
        # Ensure alpha_bar stays within a reasonable range, especially near t=0 and t=T
        return torch.clamp(alpha_bar, min=1e-8, max=1.0)

    def std_dev_fn(t: torch.Tensor) -> torch.Tensor:
        """Calculates the standard deviation \sigma(t) = sqrt(1 - \bar{\alpha}(t))."""
        alpha_bar_t = alpha_bar_fn(t)
        # Add epsilon for numerical stability near t=0 where alpha_bar might be close to 1
        return torch.sqrt(torch.clamp(1.0 - alpha_bar_t, min=1e-8))

    return beta_fn, alpha_bar_fn, std_dev_fn
