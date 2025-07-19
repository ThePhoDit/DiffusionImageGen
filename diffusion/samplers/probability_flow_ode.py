# -*- coding: utf-8 -*-
"""
Probability Flow ODE Sampler for Score-Based Generative Models.
"""

import torch
from torch import Tensor
from tqdm.notebook import tqdm  # Use standard tqdm if not in notebook
import numpy as np


def probability_flow_ode_sampler(
    diffusion_process,  # Instance of VEDiffusionProcess or VPDiffusionProcess
    score_model: torch.nn.Module,  # The trained score network
    x_T: Tensor,  # Initial noise sample at t=T, shape (batch, C, H, W)
    t_0: float,  # Starting integration time (usually T)
    t_end: float,  # End integration time (usually eps > 0)
    n_steps: int,  # Number of discretization steps
    class_labels=None,  # Optional class labels for conditional generation
    early_stop_time: float = None,  # Optional: Stop integration early at this time
    use_rk4: bool = True,  # Use RK4 integrator by default
    use_notebook_tqdm: bool = False,  # <-- Add flag
) -> tuple[Tensor, Tensor]:
    """
    Implements a Probability Flow ODE sampler using Euler or RK4 integration.

    ODE: dx = [f(x, t) - 1/2 * g(t)^2 * score(x, t)] dt = F(x, t) dt

    Args:
        diffusion_process: An object providing SDE methods (forward drift, diffusion^2, score).
        score_model: The trained neural network model.
        x_T: Initial state tensor at time t_0 (typically T).
        t_0: Starting time of the reverse process.
        t_end: Ending time of the reverse process.
        n_steps: Number of discretization steps.
        class_labels: Optional tensor of class labels for conditional models.
        early_stop_time: Optional: Stop integration early at this time.
        use_rk4: If True, use RK4 integrator, otherwise use Euler.
        use_notebook_tqdm: If True, use tqdm.notebook, otherwise use standard tqdm.

    Returns:
        A tuple (times, samples) where times is a tensor of time points and samples
        is a tensor containing the trajectory of samples shape (*x_T.shape, n_steps+1).
    """
    device = x_T.device
    # Create time discretization
    times = torch.linspace(t_0, t_end, n_steps + 1, device=device)
    dt = (t_end - t_0) / n_steps  # Negative dt since we integrate backwards

    # Get the necessary functions from the diffusion process
    drift_forward_fn = diffusion_process.sde_drift_forward()
    # Get g^2(t) function, which is named differently in VE vs VP
    # VE uses g2_t (takes t directly), VP uses diffusion_squared (returns a function that takes t)
    if hasattr(diffusion_process, "g2_t"):  # VE case
        diffusion_sq_fn_or_method = diffusion_process.g2_t
        is_method_returning_fn = False
    elif hasattr(diffusion_process, "diffusion_squared"):  # VP case
        diffusion_sq_fn_or_method = diffusion_process.diffusion_squared
        is_method_returning_fn = True  # VP returns a function
    else:
        raise AttributeError(
            f"{type(diffusion_process).__name__} object has no 'g2_t' or 'diffusion_squared' method needed for ODE sampler."
        )

    score_fn = (
        diffusion_process.score_fn
    )  # score_fn already takes score_model as first arg

    # Helper function to compute the full ODE drift F(x, t)
    def _get_ode_drift(x_state, t_val_scalar):
        vec_t = torch.ones(x_state.shape[0], device=device) * t_val_scalar
        drift_f = drift_forward_fn(x_state, vec_t)

        # Get the actual function to calculate g^2(t)
        if is_method_returning_fn:
            actual_diffusion_sq_fn = (
                diffusion_sq_fn_or_method()
            )  # Call the VP method to get the function
        else:
            actual_diffusion_sq_fn = (
                diffusion_sq_fn_or_method  # Use the VE function directly
            )

        g2_t = actual_diffusion_sq_fn(vec_t).view(
            -1, *([1] * (x_state.dim() - 1))
        )
        score = score_fn(score_model, x_state, vec_t, class_labels)

        # Calculate full drift before clamping
        ode_drift_unclamped = drift_f - 0.5 * g2_t * score

        # Return the unclamped drift
        ode_drift_final = ode_drift_unclamped

        return ode_drift_final

    # Initialize tensor to store trajectory
    x_t_traj = torch.zeros(
        (*x_T.shape, n_steps + 1), device=device, dtype=x_T.dtype
    )
    x_t = x_T.clone()  # Start with the initial noise
    x_t_traj[..., 0] = x_t

    sampler_name = "RK4" if use_rk4 else "Euler"
    print(f"Running Probability Flow ODE Sampler ({sampler_name})")

    # Iterate backwards in time
    time_steps_iter = range(n_steps)
    desc = f"ODE Sampling Steps ({sampler_name})"
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

        # --- Early Stopping Check ---
        if early_stop_time is not None and t_current < early_stop_time:
            print(
                f"!!! Early stopping ODE integration at t={t_current:.4f} (requested stop < {early_stop_time}) !!!"
            )
            stop_idx = n  # Current step index where condition was met
            # Return trajectory up to the *previous* time step (index n-1)
            return times[:stop_idx], x_t_traj[..., :stop_idx]
        # --- End Early Stopping Check ---

        if use_rk4:
            # RK4 Integration Step
            k1 = _get_ode_drift(x_t, t_current)
            k2 = _get_ode_drift(x_t + 0.5 * dt * k1, t_current + 0.5 * dt)
            k3 = _get_ode_drift(x_t + 0.5 * dt * k2, t_current + 0.5 * dt)
            k4 = _get_ode_drift(
                x_t + dt * k3, t_next
            )  # t_next = t_current + dt
            x_t = x_t + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        else:
            # Euler Integration Step (kept for comparison/debugging)
            ode_drift = _get_ode_drift(x_t, t_current)
            if torch.isnan(ode_drift).any():
                print(
                    f"!!! NaN detected in final ode_drift at step {n}, t={t_current:.4f}. Stopping early. !!!"
                )
                return times, x_t_traj
            x_t = x_t + ode_drift * dt

        # Store the state in the trajectory
        x_t_traj[..., n + 1] = x_t

    # Return times and trajectory
    return times, x_t_traj
