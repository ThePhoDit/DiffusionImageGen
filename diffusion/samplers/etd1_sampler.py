import torch
from torch import Tensor
from tqdm.notebook import tqdm  # Use standard tqdm if not in notebook
import warnings

# Ensure VPDiffusionProcess methods are available
# from diffusion.processes.variance_preserving import VPDiffusionProcess


def etd1_sampler(
    diffusion_process,  # Instance of VPDiffusionProcess
    score_model: torch.nn.Module,  # The trained score network
    x_T: Tensor,  # Initial noise sample at t=T
    t_0: float,  # Starting integration time (usually T)
    t_end: float,  # End integration time (usually eps > 0)
    n_steps: int,  # Number of discretization steps
    class_labels=None,  # Optional class labels
    eps_stable: float = 1e-6,  # Threshold for Taylor approximation
    use_notebook_tqdm: bool = False,
) -> tuple[Tensor, Tensor]:
    """
    Implements the ETD1 (Exponential Time Differencing Euler) sampler for the
    Probability Flow ODE associated with a VP process.

    ODE: dx = [-1/2*beta*x - 1/2*beta*score] dt = L*x + N(x, t) dt
    ETD1 Update: x_{i+1} = exp(L*dt)*x_i + (exp(L*dt) - 1)/(L*dt) * N(x_i, t_i) * dt

    Args:
        diffusion_process: An object providing VP schedule methods (beta, score_fn).
                           Must be VPDiffusionProcess or compatible.
        score_model: The trained neural network model.
        x_T: Initial state tensor at time t_0 (typically T).
        t_0: Starting time of the reverse process.
        t_end: Ending time of the reverse process.
        n_steps: Number of discretization steps.
        class_labels: Optional tensor of class labels for conditional models.
        eps_stable: Threshold below which Taylor approx is used for integrating factor.
        use_notebook_tqdm: If True, use tqdm.notebook, otherwise use standard tqdm.

    Returns:
        A tuple (times, samples) where times is a tensor of time points and samples
        is a tensor containing the trajectory.
    """
    # Check if the process is suitable (has beta and score_fn)
    if not (
        hasattr(diffusion_process, "beta")
        and hasattr(diffusion_process, "score_fn")
    ):
        raise TypeError(
            "ETD1 Sampler requires a diffusion process with 'beta' and 'score_fn' functions (e.g., VPDiffusionProcess)."
        )
    if getattr(diffusion_process, "kind", "") != "VP":
        warnings.warn(
            "ETD1 Sampler adapted for VP processes. Results with other processes may be unexpected."
        )

    device = x_T.device
    # Create time discretization
    times = torch.linspace(t_0, t_end, n_steps + 1, device=device)
    dt = (t_end - t_0) / n_steps  # Negative dt

    # Get necessary functions from the VP process
    beta_fn = diffusion_process.beta
    score_fn = diffusion_process.score_fn

    # Initialize tensor to store trajectory (match other samplers' format)
    x_t_traj = torch.zeros(
        (*x_T.shape, n_steps + 1), device=device, dtype=x_T.dtype
    )
    x_t = x_T.clone()
    x_t_traj[..., 0] = x_t

    print("Running ETD1 Sampler (for VP ODE)")

    # Iterate backwards in time
    time_steps_iter = range(n_steps)
    desc = "ETD1 Sampling Steps"
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
        # t_next = times[n+1] # Not needed for ETD1 explicit calculation

        vec_t_current = torch.ones(x_t.shape[0], device=device) * t_current

        # Calculate terms for the update
        beta_t = beta_fn(vec_t_current).view(-1, *([1] * (x_t.ndim - 1)))
        score = score_fn(score_model, x_t, vec_t_current, class_labels)

        # Linear coefficient L = -1/2 * beta(t)
        L = -0.5 * beta_t
        # Non-linear part N(x,t) = -1/2 * beta(t) * score(x,t)
        N = L * score  # Since L already has -0.5*beta

        z = L * dt

        # Calculate exp(L*dt)
        exp_Ldt = torch.exp(z)

        # Calculate integrating factor phi_1 = (exp(z) - 1) / z using Taylor for stability
        # Taylor approx for (e^z - 1)/z around z=0 is 1 + z/2 + z^2/6 + z^3/24 + ...
        taylor_approx = 1.0 + 0.5 * z + (1.0 / 6.0) * z**2 + (1.0 / 24.0) * z**3
        numerador = exp_Ldt - 1.0

        # Use Taylor where |z| is small
        use_taylor = z.abs() < eps_stable
        # Avoid division by zero in denominator when not using Taylor
        z_safe = torch.where(use_taylor, torch.ones_like(z), z)
        phi_1 = torch.where(use_taylor, taylor_approx, numerador / z_safe)

        # Apply ETD1 Update: x_{i+1} = exp(L*dt)*x_i + phi_1(L*dt) * N(x_i, t_i) * dt
        x_t = exp_Ldt * x_t + phi_1 * N * dt

        # Store the state
        x_t_traj[..., n + 1] = x_t

        # Optional: Check for NaNs
        if torch.isnan(x_t).any():
            print(
                f"!!! NaN detected at step {n}, t={t_current:.4f}. Stopping early. !!!"
            )
            # Return partial trajectory for debugging
            return times, x_t_traj

    # Return times and full trajectory
    return times, x_t_traj
