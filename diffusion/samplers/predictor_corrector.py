# -*- coding: utf-8 -*-
"""
Predictor-Corrector Samplers for Score-Based Generative Models.
"""

from typing import Callable
import torch
from torch import Tensor
import numpy as np
from tqdm.notebook import tqdm  # Use standard tqdm if not in notebook


def pc_sampler(
    diffusion_process,  # Instance of VEDiffusionProcess or VPDiffusionProcess
    score_model: torch.nn.Module,  # The trained score network
    x_T: Tensor,  # Initial noise sample at t=T, shape (batch, C, H, W)
    t_0: float,  # Starting integration time (usually T)
    t_end: float,  # End integration time (usually eps > 0)
    n_steps: int,  # Number of discretization steps
    snr: float,  # Signal-to-noise ratio for corrector step size
    num_corrector_steps: int,  # Number of corrector steps per predictor step
    class_labels=None,  # Optional class labels for conditional generation
    use_notebook_tqdm: bool = False,
) -> tuple[Tensor, Tensor]:
    """
    Implements a Predictor-Corrector sampler using Reverse SDE + Langevin Corrector.

    Args:
        diffusion_process: An object providing SDE methods (drift, diffusion, score, marginal_prob).
        score_model: The trained neural network model.
        x_T: Initial state tensor at time t_0 (typically T).
        t_0: Starting time of the reverse process.
        t_end: Ending time of the reverse process.
        n_steps: Number of discretization steps.
        snr: Signal-to-noise ratio for determining Langevin step size.
        num_corrector_steps: Number of Langevin corrector steps per time step.
        class_labels: Optional tensor of class labels for conditional models.
        use_notebook_tqdm: If True, use tqdm.notebook, otherwise use standard tqdm.

    Returns:
        A tuple (times, samples) where times is a tensor of time points and samples
        is a tensor containing the trajectory of samples shape (*x_T.shape, n_steps+1).
    """
    device = x_T.device
    # Create time discretization
    times = torch.linspace(t_0, t_end, n_steps + 1, device=device)
    dt = (t_end - t_0) / n_steps  # Negative dt since we integrate backwards

    # Get the reverse SDE functions from the process
    drift_fn = diffusion_process.sde_drift_reverse(score_model)
    diffusion_fn = diffusion_process.sde_diffusion_reverse()

    # Initialize tensor to store trajectory
    x_t_traj = torch.zeros(
        (*x_T.shape, n_steps + 1), device=device, dtype=x_T.dtype
    )
    x_t = x_T.clone()  # Start with the initial noise
    x_t_traj[..., 0] = x_t

    # Create a dummy tensor for marginal_prob (needed for std_dev calculation)
    dummy_x0 = torch.zeros_like(x_T)

    print(
        f"Running PC Sampler: snr={snr}, corrector_steps={num_corrector_steps}"
    )

    # Iterate backwards in time
    time_steps_iter = range(n_steps)
    if use_notebook_tqdm:
        from tqdm.notebook import tqdm

        time_steps_iter = tqdm(time_steps_iter, desc="PC Sampling Steps")
    else:
        # Only import standard tqdm if needed
        try:
            from tqdm import tqdm

            time_steps_iter = tqdm(time_steps_iter, desc="PC Sampling Steps")
        except ImportError:
            pass  # tqdm not installed, run without progress bar

    for n in time_steps_iter:
        t_current = times[n]
        t_next = times[n + 1]

        # --- Predictor Step (Reverse Euler-Maruyama) ---
        # Time tensor for function calls
        vec_t_current = torch.ones(x_t.shape[0], device=device) * t_current

        # Calculate drift and diffusion
        f = drift_fn(x_t, vec_t_current, class_labels)
        g = diffusion_fn(vec_t_current).view(-1, *([1] * (x_t.dim() - 1)))

        # Generate random noise for predictor step
        z_p = torch.randn_like(x_t)

        # Apply Euler step: x_{t-dt} = x_t + f*dt + g*sqrt(|dt|)*z_p
        # Note: dt is negative here, so f*dt handles the sign
        x_mean_pred = x_t + f * dt
        x_pred = (
            x_mean_pred + g * torch.sqrt(torch.tensor(-dt, device=device)) * z_p
        )

        # --- Corrector Step (Langevin MCMC) ---
        x_corrected = x_pred.clone()
        vec_t_next = (
            torch.ones(x_t.shape[0], device=device) * t_next
        )  # Use the target time for score

        for _ in range(num_corrector_steps):
            # Calculate score at the current corrected state and target time
            score = diffusion_process.score_fn(
                score_model, x_corrected, vec_t_next, class_labels
            )

            # Calculate Langevin step size eta
            # Need standard deviation at t_next
            _, std_dev_t_next = diffusion_process.marginal_prob(
                dummy_x0, vec_t_next
            )
            std_dev_t_next = std_dev_t_next.view(
                -1, *([1] * (x_corrected.dim() - 1))
            )

            # Step size: eta = 2 * (snr * std_dev)^2
            # Note: Some implementations use a different factor or scaling.
            # Clamping std_dev to avoid issues if it's near zero.
            eta = 2.0 * (snr * torch.clamp(std_dev_t_next, min=1e-5)) ** 2

            # Generate random noise for corrector step
            z_c = torch.randn_like(x_corrected)

            # Apply Langevin step: x_new = x_old + eta * score + sqrt(2*eta) * z_c
            x_corrected = (
                x_corrected + eta * score + torch.sqrt(2.0 * eta) * z_c
            )

        # Update the state for the next iteration
        x_t = x_corrected
        # Store the corrected state in the trajectory
        x_t_traj[..., n + 1] = x_t

    # Return times and trajectory
    return times, x_t_traj
