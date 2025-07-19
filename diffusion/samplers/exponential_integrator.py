# -*- coding: utf-8 -*-
"""
Exponential Integrator Sampler (specifically for VP-like ODEs).
"""

import torch
from torch import Tensor
from tqdm.notebook import tqdm  # Use standard tqdm if not in notebook
import numpy as np
import warnings


def exponential_integrator_sampler(
    diffusion_process,  # Instance of VPDiffusionProcess
    score_model: torch.nn.Module,  # The trained score network
    x_T: Tensor,  # Initial noise sample at t=T, shape (batch, C, H, W)
    t_0: float,  # Starting integration time (usually T)
    t_end: float,  # End integration time (usually eps > 0)
    n_steps: int,  # Number of discretization steps
    class_labels=None,  # Optional class labels for conditional generation
    eps_alpha: float = 1e-7,  # Small epsilon for stability
    use_notebook_tqdm: bool = False,  # <-- Add flag
) -> tuple[Tensor, Tensor]:
    """
    Implements a first-order Exponential Integrator (Exponential Euler) sampler.
    Designed for VP SDEs where the ODE is dx = [-1/2*beta*x - 1/2*beta*score] dt

    Update Rule (Exponential Euler):
    x_{i+1} = exp_term * x_i + non_linear_term * dt
    where exp_term = sqrt(alpha_bar(t_{i+1}) / alpha_bar(t_i))
    and non_linear_term = -1/2 * beta(t_i) * score(x_i, t_i)

    Args:
        diffusion_process: An object providing VP schedule methods (beta, alpha_bar, score_fn).
                           Must be VPDiffusionProcess or have compatible attributes.
        score_model: The trained neural network model.
        x_T: Initial state tensor at time t_0 (typically T).
        t_0: Starting time of the reverse process.
        t_end: Ending time of the reverse process.
        n_steps: Number of discretization steps.
        class_labels: Optional tensor of class labels for conditional models.
        eps_alpha: Small value added to alpha_bar in denominator for stability.
        use_notebook_tqdm: If True, use tqdm.notebook, otherwise use standard tqdm.

    Returns:
        A tuple (times, samples) where times is a tensor of time points and samples
        is a tensor containing the trajectory of samples shape (*x_T.shape, n_steps+1).
    """
    # Check if the process is suitable (has beta and alpha_bar and score_fn)
    if not (
        hasattr(diffusion_process, "beta")
        and hasattr(diffusion_process, "alpha_bar")
        and hasattr(diffusion_process, "score_fn")
    ):
        raise TypeError(
            "Exponential Integrator requires a diffusion process with 'beta', 'alpha_bar', and 'score_fn' functions (e.g., VPDiffusionProcess)."
        )
    if getattr(diffusion_process, "kind", "") != "VP":
        warnings.warn(
            "Exponential Integrator is designed for VP processes. Results with other processes may be unexpected."
        )

    device = x_T.device
    # Create time discretization
    times = torch.linspace(t_0, t_end, n_steps + 1, device=device)
    dt = (t_end - t_0) / n_steps  # Negative dt

    # Get necessary functions from the VP process
    beta_fn = diffusion_process.beta
    alpha_bar_fn = diffusion_process.alpha_bar
    score_fn = diffusion_process.score_fn

    # Initialize tensor to store trajectory
    x_t_traj = torch.zeros(
        (*x_T.shape, n_steps + 1), device=device, dtype=x_T.dtype
    )
    x_t = x_T.clone()
    x_t_traj[..., 0] = x_t

    print("Running Exponential Integrator Sampler (for VP)")

    # Iterate backwards in time
    time_steps_iter = range(n_steps)
    desc = "Exp. Integrator Sampling Steps"
    if use_notebook_tqdm:
        from tqdm.notebook import tqdm

        time_steps_iter = tqdm(time_steps_iter, desc=desc)
    else:
        try:
            from tqdm import tqdm

            time_steps_iter = tqdm(time_steps_iter, desc=desc)
        except ImportError:
            pass  # Run without tqdm

    for n in time_steps_iter:
        t_current = times[n]
        t_next = times[n + 1]

        # --- Calculate terms for the update ---
        vec_t_current = torch.ones(x_t.shape[0], device=device) * t_current
        vec_t_next = torch.ones(x_t.shape[0], device=device) * t_next

        # Exponential term: sqrt(alpha_bar(t_next) / alpha_bar(t_current))
        alpha_bar_t_curr = alpha_bar_fn(vec_t_current).view(
            -1, *([1] * (x_t.dim() - 1))
        )
        alpha_bar_t_next = alpha_bar_fn(vec_t_next).view(
            -1, *([1] * (x_t.dim() - 1))
        )
        exp_term = torch.sqrt(
            alpha_bar_t_next / torch.clamp(alpha_bar_t_curr, min=eps_alpha)
        )
        # Clamping might not be strictly necessary as alpha_bar decreases, but safe
        exp_term = torch.clamp(exp_term, 0.0, 1.0)

        # Non-linear term N(x_i, t_i) = -1/2 * beta(t_i) * score(x_i, t_i)
        beta_t_curr = beta_fn(vec_t_current).view(-1, *([1] * (x_t.dim() - 1)))
        score = score_fn(score_model, x_t, vec_t_current, class_labels)
        # Clamp score? Optional, depends on model stability.
        # score = torch.clamp(score, -1000, 1000)
        non_linear_term = -0.5 * beta_t_curr * score

        # --- Apply Exponential Integrator Update ---
        # x_{i+1} = exp_term * x_i + non_linear_term * dt
        x_t = exp_term * x_t + non_linear_term * dt

        # Store the state in the trajectory
        x_t_traj[..., n + 1] = x_t

        # Optional: Check for NaNs
        if torch.isnan(x_t).any():
            print(
                f"!!! NaN detected at step {n}, t={t_current:.4f}. Stopping early. !!!"
            )
            return times, x_t_traj

    # Return times and trajectory
    return times, x_t_traj
