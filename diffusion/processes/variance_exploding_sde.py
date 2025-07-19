"""
Implements the Variance Exploding (VE) Stochastic Differential Equation (SDE)
process, including its definition, score calculation, loss function, and
associated training and sampling helper functions.

Based on the paper "Score-Based Generative Modeling through Stochastic
Differential Equations" (Song et al., 2021), specifically the VE SDE variant.
"""

import numpy as np
from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch import Tensor
from torch.optim import Adam
import warnings

# Conditional tqdm import is handled within functions requiring progress bars

from diffusion.samplers.euler_maruyama import euler_maruyama_integrator
from diffusion.samplers.predictor_corrector import pc_sampler
from diffusion.samplers.probability_flow_ode import probability_flow_ode_sampler
from diffusion.samplers.exponential_integrator import (
    exponential_integrator_sampler,
)
from diffusion.samplers.etd1_sampler import etd1_sampler
from diffusion.models.score_model import ScoreNet

import os

device = "cuda" if torch.cuda.is_available() else "cpu"
n_threads = torch.get_num_threads()


class VEDiffusionProcess:
    """
    Encapsulates the Variance Exploding (VE) SDE diffusion process.

    This process models data generation through a stochastic differential equation
    where the variance of the noise added to the data increases (explodes) over time.

    Forward SDE:
        dxt = g(t) * dWt
        where drift is zero, and g(t) is the diffusion coefficient.

    Reverse SDE:
        dx\bar{t} = -g(t)^2 * \nabla log p_t(x\bar{t}) dt + g(t) * dW\bar{t}
        where \nabla log p_t(x\bar{t}) is the score function estimated by a neural network.

    The specific functional form of sigma(t) (and thus g(t)) follows the definition
    commonly used in implementations based on the original paper.

    Attributes:
        kind (str): Identifier for the SDE type ("VE").
        sigma_param (float): Controls the maximum standard deviation (sigma_max).
        T (float): The total integration time.
        log_sigma (float): Precomputed log of sigma_param.
        sigma_t (Callable): Function sigma(t) -> std dev at time t.
        g_t (Callable): Function g(t) -> diffusion coefficient.
        g2_t (Callable): Function g(t)^2.
        mu_t (Callable): Function mu(x_0, t) -> mean of p(xt | x0).
    """

    kind = "VE"

    def __init__(
        self,
        sigma_param: float = 25.0,
        T: float = 1.0,
    ):
        """Initializes the VE diffusion process parameters and functions.

        Args:
            sigma_param: The parameter controlling the scale of noise variance.
                         Corresponds to `σ_max` in the paper.
            T: The final time step for the diffusion process (default: 1.0).
        """
        self.sigma_param = sigma_param
        self.T = T
        self.log_sigma = np.log(sigma_param)
        print(
            f"Initializing VE diffusion process with sigma={sigma_param}, T={T}"
        )

        def _sigma_t_internal(t: torch.Tensor) -> torch.Tensor:
            """Calculates the standard deviation σ(t) for the VE SDE perturbation kernel."""
            t_scaled = torch.clamp(t / self.T, 0.0, 1.0).to(device).float()
            log_sigma_tensor = torch.tensor(
                self.log_sigma, device=t_scaled.device, dtype=torch.float32
            )
            # Variance derived from σ(t) = σ_min * (σ_max/σ_min)^t. Assuming σ_min -> 0 leads
            # to σ(t) ~ σ_max^t. The std dev here is σ_eff(t)^2 = ∫ g(s)^2 ds.
            # σ(t) = sqrt( [σ_max^(2t) - 1] / [2 * log(σ_max)] )
            var = (
                0.5
                * (self.sigma_param ** (2 * t_scaled) - 1.0 + 1e-8)
                / (log_sigma_tensor + 1e-8)
            )
            return torch.sqrt(torch.clamp(var, min=1e-8))

        self.sigma_t = _sigma_t_internal

        def _g_t_internal(t: torch.Tensor) -> torch.Tensor:
            """Calculates the diffusion coefficient g(t) = σ_max^t * sqrt(2 * log(σ_max))."""
            t_scaled = torch.clamp(t / self.T, 0.0, 1.0).to(device).float()
            # --- OLD VERSION ---
            return (
                torch.tensor(
                    self.sigma_param,
                    device=t_scaled.device,
                    dtype=torch.float32,
                )
                ** t_scaled
            )
            # --- END OLD VERSION ---
            # log_sigma_tensor = torch.tensor(
            #     self.log_sigma, device=t_scaled.device, dtype=torch.float32
            # )
            # g_t_val = (
            #     torch.tensor(
            #         self.sigma_param,
            #         device=t_scaled.device,
            #         dtype=torch.float32,
            #     )
            #     ** t_scaled
            # ) * torch.sqrt(2 * log_sigma_tensor + 1e-8)
            # return g_t_val

        self.g_t = _g_t_internal

        def _g2_t_internal(t: torch.Tensor) -> torch.Tensor:
            """Calculates g(t)^2."""
            return self.g_t(t) ** 2

        self.g2_t = _g2_t_internal

        def _mu_t_internal(x_0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            """Calculates the mean mu(t) for the VE SDE (which is just x_0)."""
            return x_0  # Mean is just x_0 for VE SDE with zero drift

        self.mu_t = _mu_t_internal

    # --- Core Process Methods ---

    def marginal_prob(
        self, x_0: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the marginal probability parameters for p(xt | x0).

        For the VE SDE, the marginal distribution is:
        - Mean \mu(t) = x_0
        - Standard Deviation \sigma(t) is given by self.sigma_t(t).

        Args:
            x_0: The initial data point (batch) at t=0.
            t: The time step(s) (scalar or batch tensor).

        Returns:
            A tuple (mean, std_dev) where:
            - mean: Tensor of the same shape as x_0.
            - std_dev: Tensor broadcastable to x_0's shape containing sigma(t).
        """
        mean = self.mu_t(x_0, t)
        std_dev_t = self.sigma_t(t).view(-1, *([1] * (x_0.dim() - 1)))
        return mean, std_dev_t

    def sample_forward(
        self, x_0: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Samples from the perturbation kernel p(xt | x0) at time t.

        Formula: x_t = \mu(t) + \sigma(t) * \epsilon, where \epsilon ~ N(0, I).
        For VE SDE, this simplifies to: x_t = x_0 + \sigma(t) * \epsilon.

        Args:
            x_0: The initial data point (batch) at t=0.
            t: The time step(s) (scalar or batch tensor).

        Returns:
            A tuple (x_t, noise) where:
            - x_t: The noisy sample(s) at time t.
            - noise: The standard Gaussian noise \epsilon used to generate the sample.
        """
        mean, std = self.marginal_prob(x_0, t)
        noise = torch.randn_like(x_0)
        x_t = mean + std * noise
        return x_t, noise

    def score_fn(
        self,
        score_model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        class_labels=None,
    ) -> torch.Tensor:
        """
        Calculates the score function \nabla log p_t(xt) using the trained neural network.

        This relies on the ScoreNet model being configured correctly. For VE SDE,
        ScoreNet internally divides its output by sigma(t), so the model's direct
        output is the score.

        Args:
            score_model: The trained score network (potentially wrapped in DataParallel).
            x_t: The noisy data point(s) at time t.
            t: The time step(s).
            class_labels: Optional class labels for conditional models.

        Returns:
            The estimated score tensor, \nabla log p_t(xt).
        """
        t_clamped = torch.clamp(t, 0.0, self.T).to(x_t.device)

        # Unwrap model if DataParallel
        model_module = (
            score_model.module
            if isinstance(score_model, nn.DataParallel)
            else score_model
        )

        # Pass class labels if the model is conditional and labels are provided
        if (
            getattr(model_module, "use_class_condition", False)
            and class_labels is not None
        ):
            score = score_model(x_t, t_clamped, class_labels)
        else:
            score = score_model(x_t, t_clamped)

        # Handle potential NaNs for numerical stability
        if torch.isnan(score).any():
            warnings.warn(
                f"NaN detected in predicted score at t={t.mean().item():.4f}. Replacing with zeros.",
                RuntimeWarning,
            )
            score = torch.nan_to_num(score, nan=0.0)

        return score

    def loss_fn(
        self,
        score_model: nn.Module,
        x_0: torch.Tensor,
        y=None,
        eps: float = 1e-5,
    ) -> torch.Tensor:
        """
        Computes the score matching loss for the VE SDE.

        The loss is based on denoising score matching principles, adapted for the VE SDE:
        Loss = E_t [ \lambda(t) * || s_\theta(x_t, t) - \nabla log p_0t(x_t | x_0) ||^2 ]
        where \lambda(t) is a weighting function, often chosen as
        E[||\nabla log p_0t(x_t | x_0)||^2]^-1.
        For VE SDE, \nabla log p_0t(x_t | x_0) = - (x_t - x_0) / sigma(t)^2
        = - noise / sigma(t).
        The common loss formulation simplifies to:
        Loss = E_t [ || sigma(t) * s_\theta(x_t, t) + noise ||^2 ]
        where s_\theta(x_t, t) is the score predicted by the model.

        Args:
            score_model: The score network model.
            x_0: The original clean data batch.
            y: Optional class labels for conditional training.
            eps: A small epsilon to avoid sampling t exactly at 0 or T.

        Returns:
            The computed scalar loss value for the batch.
        """
        # Sample time uniformly in [eps, T]
        t = torch.rand(x_0.shape[0], device=x_0.device) * (self.T - eps) + eps

        # Get noisy sample x_t and the noise \epsilon used
        x_t, noise = self.sample_forward(x_0, t)

        # Get standard deviation sigma(t)
        sigma_t_val = self.sigma_t(t).view(-1, *([1] * (x_0.dim() - 1)))
        sigma_t_val = torch.clamp(
            sigma_t_val, min=1e-5
        )  # Avoid division by zero

        # Get model's score prediction
        model_module = (
            score_model.module
            if isinstance(score_model, nn.DataParallel)
            else score_model
        )
        if (
            getattr(model_module, "use_class_condition", False)
            and y is not None
        ):
            predicted_score = score_model(x_t, t, y)
        else:
            predicted_score = score_model(x_t, t)

        # Handle potential NaNs in prediction
        if torch.isnan(predicted_score).any():
            warnings.warn(
                f"NaN detected in predicted score during loss calculation at t={t.mean().item():.4f}. Replacing with zeros.",
                RuntimeWarning,
            )
            predicted_score = torch.nan_to_num(predicted_score, nan=0.0)

        # Calculate the loss term: (sigma(t) * score_prediction + noise)^2
        # Sum over spatial/channel dimensions, then average over batch
        loss_per_example = torch.sum(
            (sigma_t_val * predicted_score + noise) ** 2,
            dim=list(range(1, x_0.dim())),
        )
        loss = torch.mean(loss_per_example)
        return loss

    # --- SDE Coefficients (Forward and Reverse) ---

    def sde_drift_forward(self) -> Callable[..., torch.Tensor]:
        """
        Returns the drift function f(x, t) for the forward VE SDE.

        For the VE SDE defined here, the drift f(x, t) = 0.

        Returns:
            A function `drift(x_t, t)` that returns a zero tensor shaped like x_t.
        """

        def _drift(x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            # Drift is zero
            return torch.zeros_like(x_t)

        return _drift

    def sde_diffusion(self) -> Callable[..., torch.Tensor]:
        """
        Returns the diffusion function g(t) for the VE SDE.

        Returns:
            The function `self.g_t(t)`.
        """
        return self.g_t

    def sde_drift_reverse(
        self, score_model: nn.Module
    ) -> Callable[..., torch.Tensor]:
        """
        Returns the drift function \tilde{f}(x, t) for the reverse VE SDE.

        Formula: \tilde{f}(x, t) = -g(t)^2 * score_fn(x, t)
        This corresponds to the drift used in the sampling process.

        Args:
            score_model: The trained score network.

        Returns:
            A function `drift_reverse(x_t, t, class_labels=None)` that computes the reverse drift.
        """
        g2_fn = self.g2_t  # Function g(t)^2

        def _drift_reverse(
            x_t: torch.Tensor, t: torch.Tensor, class_labels=None
        ) -> torch.Tensor:
            t_clamped = torch.clamp(t, 0.0, self.T).to(x_t.device)
            g2_t_val = g2_fn(t_clamped).view(-1, *([1] * (x_t.dim() - 1)))
            g2_t_val = torch.clamp(g2_t_val, min=1e-7)  # Numerical stability

            # Calculate the score using the model
            score = self.score_fn(score_model, x_t, t_clamped, class_labels)
            score = torch.clamp(
                score, -1000, 1000
            )  # Clamp score to prevent explosion

            # Calculate reverse drift
            drift = -g2_t_val * score
            # Clamp final drift to prevent numerical issues in sampler
            return torch.clamp(drift, -1000.0, 1000.0)

        return _drift_reverse

    def sde_diffusion_reverse(self) -> Callable[..., torch.Tensor]:
        """
        Returns the diffusion function \bar{g}(t) for the reverse VE SDE.

        This is the same as the forward diffusion function g(t) for the
        variance exploding SDE, as per the theory of SDEs.

        Returns:
            The function `self.g_t(t)`.
        """
        return self.g_t


# --- Generic Helper Functions ---


def get_score_model(
    diffusion_process: VEDiffusionProcess,
    image_channels: int = 3,
    num_classes: int = None,
    scorenet_channels: list[int] = [64, 128, 256, 512],
    scorenet_embed_dim: int = 256,
) -> nn.Module:
    """
    Creates and initializes a ScoreNet model configured for the given diffusion process.

    Automatically uses the appropriate marginal probability standard deviation function
    from the provided diffusion process instance and handles DataParallel setup.

    Args:
        diffusion_process: An instance of a diffusion process class (e.g., VEDiffusionProcess).
        image_channels: Number of channels in the input/output images.
        num_classes: Number of classes for conditional modeling. If None, model is unconditional.
        scorenet_channels: List of channel counts for U-Net blocks in ScoreNet.
        scorenet_embed_dim: Embedding dimension for time/class in ScoreNet.

    Returns:
        The initialized ScoreNet model (possibly wrapped in nn.DataParallel), moved to the appropriate device.

    Raises:
        ValueError: If diffusion_process is None.
    """
    if diffusion_process is None:
        raise ValueError("Diffusion process instance must be provided.")

    print(
        f"Getting ScoreModel configured for {diffusion_process.kind} process."
    )

    # Wrapper to provide the marginal_prob_std function required by ScoreNet
    def _std_dev_wrapper(t: torch.Tensor) -> torch.Tensor:
        """Extracts std deviation from the process's marginal_prob method."""
        # Needs a dummy x_0 to call marginal_prob. Shape/content don't matter.
        t_dev = t.to(device)
        batch_size = t_dev.shape[0] if t_dev.dim() > 0 else 1
        dummy_x0 = torch.zeros(
            (batch_size, 1, 1, 1), device=device  # Minimal shape
        )
        _, std_dev = diffusion_process.marginal_prob(dummy_x0, t_dev)
        # Ensure output shape is [batch_size]
        return std_dev.view(batch_size)

    # Create the ScoreNet instance
    score_model_instance = ScoreNet(
        marginal_prob_std=_std_dev_wrapper,
        image_channels=image_channels,
        channels=scorenet_channels,
        embed_dim=scorenet_embed_dim,
        num_classes=num_classes,
        # For VE, ScoreNet output is the score, so disable_final_scaling should be False.
        # For VP, if model predicts noise ε, set disable_final_scaling=True.
        disable_final_scaling=(diffusion_process.kind == "VP"),
    )

    # Use DataParallel if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        print(
            f"Using {torch.cuda.device_count()} GPUs! Wrapping model in DataParallel."
        )
        score_model = nn.DataParallel(score_model_instance)
    else:
        score_model = score_model_instance

    return score_model.to(device)


def load_model(
    diffusion_process: VEDiffusionProcess,
    model_path: str,
    image_channels: int = 3,
    num_classes: int = None,
) -> nn.Module:
    """
    Loads a pre-trained ScoreNet model state dictionary from a file.

    Configures the model structure based on the provided diffusion process and
    parameters, then loads the weights. Handles potential mismatches in keys
    due to saving/loading with or without nn.DataParallel.

    Args:
        diffusion_process: The diffusion process instance the model was trained for.
        model_path: Path to the saved state dictionary (.pth file).
        image_channels: Number of image channels the model expects.
        num_classes: Number of classes if the model is conditional.

    Returns:
        The loaded model, set to evaluation mode and moved to the appropriate device.
    """
    print(f"Loading model from: {model_path}")
    # Get the base model structure
    model = get_score_model(
        diffusion_process,
        image_channels=image_channels,
        num_classes=num_classes,
    )

    # Load the saved state dictionary
    state_dict = torch.load(model_path, map_location=torch.device(device))

    # Adjust keys if DataParallel wrapper status differs between saving and loading
    is_parallel_model = isinstance(model, nn.DataParallel)
    is_parallel_state = list(state_dict.keys())[0].startswith("module.")

    if is_parallel_model and not is_parallel_state:
        print(
            "Adding 'module.' prefix to state_dict keys for DataParallel model."
        )
        state_dict = {"module." + k: v for k, v in state_dict.items()}
    elif not is_parallel_model and is_parallel_state:
        print(
            "Removing 'module.' prefix from state_dict keys for non-DataParallel model."
        )
        state_dict = {
            k.partition("module.")[2]: v for k, v in state_dict.items()
        }

    # Load the potentially adjusted state dictionary
    model.load_state_dict(state_dict)
    model.eval()  # Set to evaluation mode after loading
    print("Model loaded successfully.")
    return model


def train(
    diffusion_process: VEDiffusionProcess,
    data_train: Dataset,
    batch_size: int = 32,
    n_epochs: int = 10,
    learning_rate: float = 1e-4,
    save_model_to: str = "checkpoint.pth",
    grad_clip_val: float = 1.0,
    use_class_condition: bool = False,
    num_classes: int = None,
    image_channels: int = 3,
    save_checkpoint_every: int = 10,
    use_notebook_tqdm: bool = False,
) -> nn.Module:
    """
    Trains the ScoreNet model using the specified diffusion process and dataset.

    Handles the training loop, optimization, loss calculation (delegated to the
    diffusion_process instance), gradient clipping, progress reporting, and
    saving the final model and periodic checkpoints.

    Args:
        diffusion_process: The diffusion process instance defining the SDE and loss.
        data_train: The training dataset (torch.utils.data.Dataset).
        batch_size: Training batch size.
        n_epochs: Number of training epochs.
        learning_rate: Optimizer learning rate.
        save_model_to: Path to save the final trained model state dictionary.
        grad_clip_val: Maximum gradient norm for clipping (set <= 0 to disable).
        use_class_condition: Whether to train a class-conditional model.
        num_classes: Number of classes (required if use_class_condition is True).
        image_channels: Number of channels in the training images.
        save_checkpoint_every: Save a checkpoint every N epochs.
        use_notebook_tqdm: If True, use tqdm.notebook for progress bars (for Jupyter).

    Returns:
        The trained ScoreNet model.

    Raises:
        ValueError: If use_class_condition is True but num_classes is not provided.
    """

    # --- Conditional tqdm import ---
    if use_notebook_tqdm:
        from tqdm.notebook import trange, tqdm
    else:
        from tqdm import trange, tqdm
    # -----------------------------

    if use_class_condition and num_classes is None:
        raise ValueError(
            "num_classes must be provided if use_class_condition is True."
        )

    # Initialize model and optimizer
    score_model = get_score_model(
        diffusion_process,
        image_channels=image_channels,
        num_classes=num_classes if use_class_condition else None,
    )
    optimizer = Adam(score_model.parameters(), lr=learning_rate)

    # Ensure model save directory exists
    os.makedirs(os.path.dirname(save_model_to), exist_ok=True)

    # Setup DataLoader
    data_loader = DataLoader(
        data_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=min(n_threads, 4),
        pin_memory=(device == "cuda"),
        drop_last=True,
    )

    tqdm_epoch = trange(n_epochs, desc="Training Progress")
    nan_batches_total = 0

    print(f"--- Starting Training ---")
    print(f"Process: {diffusion_process.kind}")
    print(f"Class conditioning: {use_class_condition}")
    if use_class_condition:
        print(f"Num classes: {num_classes}")
    print(f"Epochs: {n_epochs}, Batch size: {batch_size}, LR: {learning_rate}")
    print(
        f"Gradient Clipping: {grad_clip_val if grad_clip_val > 0 else 'Disabled'}"
    )
    print(f"Saving final model to: {save_model_to}")
    print(f"Saving checkpoints every {save_checkpoint_every} epochs.")
    print(f"Using device: {device}")
    print(f"-------------------------")

    for epoch in tqdm_epoch:
        score_model.train()
        epoch_loss = 0.0
        num_items = 0
        nan_batches_epoch = 0

        batch_progress = tqdm(
            data_loader, desc=f"Epoch {epoch + 1}/{n_epochs}", leave=False
        )

        for batch_idx, batch_data in enumerate(batch_progress):
            # Handle datasets yielding (image, label) or just image
            if isinstance(batch_data, (list, tuple)):
                x, y = batch_data
                y = y.to(device) if use_class_condition else None
            else:
                x = batch_data
                y = None

            x = x.to(device)

            # Calculate loss using the diffusion process's loss function
            loss = diffusion_process.loss_fn(score_model, x, y)

            # --- Gradient Update --- #
            optimizer.zero_grad()

            # Check for NaN/Inf loss before backward pass
            if torch.isnan(loss) or torch.isinf(loss):
                warnings.warn(
                    f"NaN/Inf loss detected in batch {batch_idx}, epoch {epoch}. Skipping backward pass and optimizer step.",
                    RuntimeWarning,
                )
                nan_batches_epoch += 1
                nan_batches_total += 1
                # No optimizer step needed if loss is invalid
                continue

            loss.backward()

            # Gradient Clipping (apply before optimizer step)
            if grad_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(
                    score_model.parameters(), grad_clip_val
                )

            optimizer.step()
            # --------------------- #

            batch_loss_item = loss.item()
            epoch_loss += batch_loss_item * x.shape[0]
            num_items += x.shape[0]
            current_avg_loss = epoch_loss / max(num_items, 1)

            batch_progress.set_postfix(
                {
                    "batch_loss": f"{batch_loss_item:.4f}",
                    "avg_epoch_loss": f"{current_avg_loss:.4f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.6f}",
                    "nans_epoch": nan_batches_epoch,
                }
            )

        # End of Epoch
        final_epoch_loss = epoch_loss / max(num_items, 1)
        nan_batches_total += nan_batches_epoch
        tqdm_epoch.set_postfix(
            {
                "epoch_loss": f"{final_epoch_loss:.4f}",
                "total_nan_batches": nan_batches_total,
            }
        )

        # --- Save Checkpoint --- #
        if (epoch + 1) % save_checkpoint_every == 0 or (epoch + 1) == n_epochs:
            chkpt_dir = os.path.dirname(save_model_to)
            base_name = os.path.basename(save_model_to)
            file_root, file_ext = os.path.splitext(base_name)
            chkpt_name = os.path.join(
                chkpt_dir, f"{file_root}_epoch{epoch + 1}{file_ext}"
            )
            # Save the state dict, handling DataParallel wrapper correctly
            state_dict_to_save = (
                score_model.module.state_dict()
                if isinstance(score_model, nn.DataParallel)
                else score_model.state_dict()
            )
            torch.save(state_dict_to_save, chkpt_name)
            print(f"\nCheckpoint saved: {chkpt_name}")

    # --- Final Model Save --- #
    # (Already saved in the last checkpoint loop if save_checkpoint_every=1 or last epoch)
    if n_epochs % save_checkpoint_every != 0:
        final_state_dict = (
            score_model.module.state_dict()
            if isinstance(score_model, nn.DataParallel)
            else score_model.state_dict()
        )
        torch.save(final_state_dict, save_model_to)
        print(f"\nFinal model state saved to {save_model_to}")

    print(f"\n--- Training completed --- ")
    print(f"Final Epoch Loss: {final_epoch_loss:.4f}")
    print(f"Total NaN batches encountered: {nan_batches_total}")
    return score_model


def generate_images(
    diffusion_process: VEDiffusionProcess,
    score_model: nn.Module,
    n_images: int = 16,
    target_class: int = None,
    image_size: tuple = (32, 32),
    n_channels: int = 3,
    n_steps: int = 1000,
    sampler_type: str = "euler_maruyama",
    eps: float = 1e-3,
    pc_snr: float = 0.1,
    pc_num_corrector_steps: int = 1,
    ode_early_stop_time: float = None,
    ode_use_rk4: bool = True,
    clamp_final: bool = True,
    clamp_range: tuple = (-1.0, 1.0),
    use_notebook_tqdm: bool = False,
) -> Tensor:
    """
    Generates images by solving the reverse SDE associated with the diffusion process.

    Starts from random noise at time T and integrates backward to time eps (~0)
    using a specified numerical SDE solver (sampler).

    Args:
        diffusion_process: The diffusion process instance defining the reverse SDE.
        score_model: The trained score network.
        n_images: Number of images to generate.
        target_class: If generating conditionally, the target class index. If None,
                      generation is unconditional (or uses default if model requires it).
        image_size: Tuple representing the desired (height, width) of the images.
        n_channels: Number of channels for the generated images.
        n_steps: Number of discretization steps for the sampler.
        sampler_type: Name of the sampler to use ('euler_maruyama', 'pc', 'ode', 'ei').
        eps: The final time step (close to 0) for the reverse integration.
        pc_snr: Signal-to-noise ratio for the Predictor-Corrector sampler's corrector step.
        pc_num_corrector_steps: Number of corrector steps in the PC sampler.
        ode_early_stop_time: Optional time < T to stop ODE integration early.
        ode_use_rk4: If True, use RK4 method for ODE sampler; otherwise, use Euler.
        clamp_final: Whether to clamp the final generated image pixel values.
        clamp_range: Tuple (min, max) for final image clamping.
        use_notebook_tqdm: If True, use tqdm.notebook for progress bars.

    Returns:
        A tensor containing the generated images, shape [n_images, n_channels, height, width].

    Raises:
        ValueError: If an unknown sampler_type is provided.
    """
    score_model.eval()

    # Initial random sample at time T (from N(0, I))
    # The sampler will handle scaling/adjustment based on the process
    x_T = torch.randn(
        (n_images, n_channels, image_size[0], image_size[1]), device=device
    )
    T_start = diffusion_process.T

    # --- Class Conditioning Setup --- #
    class_labels = None
    model_module = (
        score_model.module
        if isinstance(score_model, nn.DataParallel)
        else score_model
    )
    model_is_conditional = getattr(model_module, "use_class_condition", False)

    if target_class is not None:
        if model_is_conditional:
            print(
                f"Generating {n_images} images conditionally for class {target_class}."
            )
            class_labels = torch.full(
                (n_images,), target_class, dtype=torch.long, device=device
            )
        else:
            warnings.warn(
                f"Target class {target_class} requested, but the loaded model is unconditional. Ignoring class label.",
                RuntimeWarning,
            )
    elif model_is_conditional:
        warnings.warn(
            "Model is conditional, but no target class specified. Generating unconditionally (may use zero embedding internally).",
            RuntimeWarning,
        )
        # class_labels remains None; ScoreNet handles this case

    # --- Get SDE Components from the Process --- #
    # Note: These functions return callables
    reverse_drift_fn_base = diffusion_process.sde_drift_reverse(score_model)
    reverse_diffusion_fn = diffusion_process.sde_diffusion_reverse()

    # --- Prepare Sampler Arguments --- #
    # Wrap drift to handle class labels consistently for all samplers
    def reverse_drift_wrapper(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return reverse_drift_fn_base(x, t, class_labels=class_labels)

    # Diffusion usually only needs time `t`
    def reverse_diffusion_wrapper(t: torch.Tensor) -> torch.Tensor:
        # Add a batch dimension if t is scalar for broadcasting in sampler
        t_batched = t if t.dim() > 0 else t.unsqueeze(0)
        diff_val = reverse_diffusion_fn(t_batched)
        # Ensure output has a batch dim, even if input t was scalar
        return diff_val.view(-1, *([1] * (x_T.dim() - 1)))  # Broadcastable

    # --- Select and Run Sampler --- #
    sampler_name = sampler_type.lower().replace("-", "_")  # Normalize name
    print(
        f"Generating {n_images} images using {diffusion_process.kind} reverse SDE..."
    )
    print(f"Sampler: {sampler_name}, Steps: {n_steps}, T_end={eps}")

    final_images = None
    with torch.no_grad():
        if sampler_name in ["euler_maruyama", "euler"]:
            times, synthetic_images_t = euler_maruyama_integrator(
                x_0=x_T,
                t_0=T_start,
                t_end=eps,
                n_steps=n_steps,
                drift_coefficient=reverse_drift_wrapper,
                diffusion_coefficient=reverse_diffusion_wrapper,
            )
            final_images = synthetic_images_t[..., -1]  # Sample at t_end

        elif sampler_name in ["pc", "predictor_corrector"]:
            print(
                f"PC Sampler Params: SNR={pc_snr}, Corrector Steps={pc_num_corrector_steps}"
            )
            times, synthetic_images_t = pc_sampler(
                diffusion_process=diffusion_process,
                score_model=score_model,
                x_T=x_T,
                t_0=T_start,
                t_end=eps,
                n_steps=n_steps,
                snr=pc_snr,
                num_corrector_steps=pc_num_corrector_steps,
                class_labels=class_labels,
                use_notebook_tqdm=use_notebook_tqdm,
            )
            final_images = synthetic_images_t[..., -1]

        elif sampler_name == "ode":
            print(
                f"ODE Sampler Params: RK4={'Yes' if ode_use_rk4 else 'No'}, Early Stop={ode_early_stop_time}"
            )
            times, synthetic_images_t = probability_flow_ode_sampler(
                diffusion_process=diffusion_process,
                score_model=score_model,
                x_T=x_T,
                t_0=T_start,
                t_end=eps,
                n_steps=n_steps,
                class_labels=class_labels,
                early_stop_time=ode_early_stop_time,
                use_rk4=ode_use_rk4,
                use_notebook_tqdm=use_notebook_tqdm,
            )
            final_images = synthetic_images_t[..., -1]

        elif sampler_name in ["ei", "exponential_integrator"]:
            print("Using Exponential Integrator sampler.")
            times, synthetic_images_t = exponential_integrator_sampler(
                diffusion_process=diffusion_process,
                score_model=score_model,
                x_T=x_T,
                t_0=T_start,
                t_end=eps,
                n_steps=n_steps,
                class_labels=class_labels,
                use_notebook_tqdm=use_notebook_tqdm,
            )
            final_images = synthetic_images_t[..., -1]

        elif sampler_name == "etd1":
            print("Using ETD1 sampler...")
            # ETD1 sampler takes the process object directly
            _, samples_traj = etd1_sampler(
                **{
                    "diffusion_process": diffusion_process,
                    "score_model": score_model,
                    "x_T": x_T,
                    "t_0": T_start,
                    "t_end": eps,
                    "n_steps": n_steps,
                    "class_labels": class_labels,
                    "use_notebook_tqdm": use_notebook_tqdm,
                },
            )
            final_images = samples_traj[..., -1]

        else:
            raise ValueError(f"Unknown sampler_type: {sampler_type}")

    # --- Final Processing --- #
    if clamp_final:
        final_images = torch.clamp(
            final_images, min=clamp_range[0], max=clamp_range[1]
        )
        print(f"Final images clamped to range {clamp_range}.")
    else:
        print("Final image clamping skipped.")

    print("Image generation finished.")
    return final_images
