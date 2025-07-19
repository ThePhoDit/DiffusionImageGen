# -*- coding: utf-8 -*-
"""
Imputation Sampler using RePaint strategy.
"""
import torch
from torch import Tensor
from tqdm.notebook import tqdm
import warnings


def repaint_sampler(
    diffusion_process,  # Instance of VPDiffusionProcess or VEDiffusionProcess
    score_model: torch.nn.Module,  # Trained score network
    x_masked: Tensor,  # Original image tensor with known values [-1, 1]
    mask: Tensor,  # Binary mask (1=known, 0=unknown)
    t_0: float,  # Starting integration time (usually T)
    t_end: float,  # End integration time (usually eps > 0)
    n_steps: int,  # Total number of reverse steps
    jump_length: int,  # How many steps back to jump (N)
    jump_n_sample: int,  # How many times to resample for each jump (R)
    class_labels=None,  # Optional class labels
    use_notebook_tqdm: bool = False,
) -> tuple[Tensor, Tensor]:
    """
    Implements image imputation using the RePaint strategy:
    Alternates reverse diffusion steps with forward noising and resampling
    of the known image regions.

    Args:
        diffusion_process: Diffusion process object (VP or VE).
        score_model: Trained score model.
        x_masked: Original known image data (normalized to [-1, 1]), shape (B, C, H, W).
        mask: Binary mask tensor (1 for known, 0 for unknown), shape (B, C, H, W).
        t_0: Starting time (T).
        t_end: Ending time (eps).
        n_steps: Total number of discretization steps for the full reverse process.
        jump_length: Number of steps for the RePaint jump (N).
        jump_n_sample: Number of resampling steps within each jump (R).
        class_labels: Optional class labels.
        use_notebook_tqdm: If True, use tqdm.notebook, otherwise use standard tqdm.

    Returns:
        Tuple (times, samples trajectory)
    """
    device = x_masked.device
    batch_size = x_masked.shape[0]

    # Time steps
    times = torch.linspace(t_0, t_end, n_steps + 1, device=device)

    # Get SDE components (only reverse is needed for generation part)
    drift_fn = diffusion_process.sde_drift_reverse(score_model)
    diffusion_fn = diffusion_process.sde_diffusion_reverse()

    # Initialize with random noise
    x_t = torch.randn_like(x_masked)

    # Trajectory storage
    x_t_traj = torch.zeros(
        (*x_masked.shape, n_steps + 1), device=device, dtype=x_masked.dtype
    )
    x_t_traj[..., 0] = x_t

    print(
        f"Running RePaint Sampler: n_steps={n_steps}, jump_length={jump_length}, jump_n_sample={jump_n_sample}"
    )

    time_steps_iter = range(n_steps)
    desc = "RePaint Sampling Steps"
    if use_notebook_tqdm:
        time_steps_iter = tqdm(time_steps_iter, desc=desc)
    else:
        try:
            from tqdm import tqdm

            time_steps_iter = tqdm(time_steps_iter, desc=desc)
        except ImportError:
            pass  # Run without tqdm

    for n in time_steps_iter:
        t_current_idx = n
        t_next_idx = n + 1
        t_current = times[t_current_idx]
        t_next = times[t_next_idx]
        dt = t_next - t_current  # dt is negative

        # --- Create time tensors ---
        vec_t_current = torch.ones(batch_size, device=device) * t_current
        vec_t_next = (
            torch.ones(batch_size, device=device) * t_next
        )  # <-- Define vec_t_next here
        # -------------------------

        # --- Standard Reverse Diffusion Step (Euler-Maruyama) ---
        f = drift_fn(x_t, vec_t_current, class_labels)
        g = diffusion_fn(vec_t_current).view(-1, *([1] * (x_t.dim() - 1)))
        z_p = torch.randn_like(x_t)
        x_mean_next = x_t + f * dt
        x_t_next_noised = x_mean_next + g * torch.sqrt(-dt) * z_p
        # ------------------------------------------------------

        # --- RePaint Resampling Step ---
        if (
            n % jump_length == 0 and n > 0
        ):  # Perform jump at intervals (skip first step)
            for _ in range(jump_n_sample):
                # 1. Calculate known image data at t_next using analytical formula
                x_known_t_next, _ = diffusion_process.sample_forward(
                    x_masked, vec_t_next
                )

                # 2. Noise x_t back to t_current (using forward SDE)
                # Need forward diffusion/drift, simplified here assuming VE/VP structure
                if diffusion_process.kind == "VP":
                    # VP forward: x_curr = sqrt(a_curr/a_next) * x_next + sqrt(1 - a_curr/a_next) * noise
                    a_curr = diffusion_process.alpha_bar(vec_t_current).view(
                        -1, 1, 1, 1
                    )
                    a_next = diffusion_process.alpha_bar(vec_t_next).view(
                        -1, 1, 1, 1
                    )
                    alpha_ratio = torch.clamp(
                        a_curr / (a_next + 1e-7), 0.0, 1.0
                    )
                    std_dev_fwd = torch.sqrt(
                        torch.clamp(1.0 - alpha_ratio, min=1e-7)
                    )
                    z_fwd = torch.randn_like(x_t)
                    x_t_noised_back = (
                        torch.sqrt(alpha_ratio) * x_t_next_noised
                        + std_dev_fwd * z_fwd
                    )
                elif diffusion_process.kind == "VE":
                    # VE forward: x_curr = x_next + sqrt(sigma_curr^2 - sigma_next^2) * noise
                    var_curr = (
                        diffusion_process.sigma_t(vec_t_current)
                        .pow(2)
                        .view(-1, 1, 1, 1)
                    )
                    var_next = (
                        diffusion_process.sigma_t(vec_t_next)
                        .pow(2)
                        .view(-1, 1, 1, 1)
                    )
                    std_dev_fwd = torch.sqrt(
                        torch.clamp(var_curr - var_next, min=1e-7)
                    )
                    z_fwd = torch.randn_like(x_t)
                    x_t_noised_back = x_t_next_noised + std_dev_fwd * z_fwd
                else:
                    # Fallback/Error for unsupported process
                    warnings.warn(
                        f"RePaint forward noise step not implemented for process type {diffusion_process.kind}. Using previous state."
                    )
                    x_t_noised_back = x_t_next_noised  # Simple fallback

                # 3. Combine: Use noised-back sample in unknown regions, known data in known regions
                x_t_next_noised = (
                    x_t_noised_back * (1.0 - mask) + x_known_t_next * mask
                )
        # -----------------------------

        # Update state for next iteration
        x_t = x_t_next_noised
        x_t_traj[..., t_next_idx] = x_t

    return times, x_t_traj
