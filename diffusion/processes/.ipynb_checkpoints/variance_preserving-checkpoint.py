import torch
import torch.nn as nn
from typing import Callable, Tuple

# Import schedule functions
from diffusion.schedules.linear import get_linear_schedule_functions
from diffusion.schedules.cosine import get_cosine_schedule_functions

# Assuming ScoreNet might be needed, but loss/score calculation is internal
# from diffusion.models.score_model import ScoreNet

device = "cuda" if torch.cuda.is_available() else "cpu"


class VPDiffusionProcess:
    """
    Implements the Variance Preserving (VP) SDE diffusion process,
    also known as the Ornstein-Uhlenbeck process in this context.
    Uses configurable noise schedules (linear or cosine).

    Forward SDE: dxt = -1/2 * β(t) * xt * dt + sqrt(β(t)) * dWt
    Reverse SDE: dx̄t = [-1/2 * β(t) * x̄t - β(t) * ∇log p_t(x̄t)] dt + sqrt(β(t)) * dW̄t
    """

    kind = "VP"

    def __init__(
        self,
        schedule: str = "linear",
        T: float = 1.0,
        beta_min: float = 0.01,  # Linear schedule param
        beta_max: float = 0.95,  # Linear schedule param
        cosine_s: float = 0.008,  # Cosine schedule param
    ):
        self.schedule_type = schedule
        self.T = T
        print(
            f"Initializing VP diffusion process with {schedule} schedule, T={T}."
        )

        if schedule == "linear":
            self.beta, self.alpha_bar, self.std_dev = (
                get_linear_schedule_functions(
                    beta_min=beta_min, beta_max=beta_max, T=T
                )
            )
            self.beta_min = beta_min
            self.beta_max = beta_max
        elif schedule == "cosine":
            self.beta, self.alpha_bar, self.std_dev = (
                get_cosine_schedule_functions(T=T, s=cosine_s)
            )
            self.cosine_s = cosine_s
        else:
            raise ValueError(
                f"Unknown schedule: {schedule}. Choose 'linear' or 'cosine'."
            )

    def marginal_prob(
        self, x_0: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the mean and standard deviation of p(xt | x0) for the VP SDE.
        μ(t) = sqrt(ᾱ(t)) * x₀
        σ(t) = sqrt(1 - ᾱ(t))

        Returns:
            Tuple[mean, std_dev]
        """
        # Ensure t is on the correct device and clamped
        t_clamped = torch.clamp(t, 0.0, self.T).to(x_0.device)
        alpha_bar_t = self.alpha_bar(t_clamped).view(
            -1, *([1] * (x_0.dim() - 1))
        )
        std_dev_t = self.std_dev(t_clamped).view(-1, *([1] * (x_0.dim() - 1)))

        # Clamp alpha_bar_t slightly away from 1.0 to avoid mean being exactly x_0 at t=0
        alpha_bar_t_sqrt = torch.sqrt(torch.clamp(alpha_bar_t, max=1.0 - 1e-8))
        mean = alpha_bar_t_sqrt * x_0
        std = std_dev_t
        return mean, std

    def sample_forward(
        self, x_0: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample x_t from x_0 at time t using the forward process analytical solution:
        x_t = sqrt(ᾱ(t)) * x₀ + sqrt(1 - ᾱ(t)) * ε , where ε ~ N(0, I)

        Returns:
            Tuple[noisy_sample_x_t, noise_added_epsilon]
        """
        mean, std = self.marginal_prob(x_0, t)
        noise = torch.randn_like(x_0)
        x_t = mean + std * noise
        return x_t, noise  # Return sample and the noise used

    def score_fn(
        self,
        score_model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        class_labels=None,
    ) -> torch.Tensor:
        """
        Calculate the score ∇log p_t(xt) assuming the score_model predicts x_0.
        Score = -(x_t - sqrt(alpha_bar_t) * x_0_predicted) / (1 - alpha_bar_t)
        Requires the model output *not* to be scaled by sigma(t).
        """
        t_clamped = torch.clamp(t, 0.0, self.T).to(x_t.device)
        alpha_bar_t = self.alpha_bar(t_clamped).view(
            -1, *([1] * (x_t.dim() - 1))
        )
        alpha_bar_t = torch.clamp(
            alpha_bar_t, min=0.0, max=1.0 - 1e-7
        )  # Ensure 1 - alpha_bar is not zero
        sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
        one_minus_alpha_bar_t = 1.0 - alpha_bar_t

        # Get x_0 prediction from model
        # Assumes model output is x_0_hat (disable_final_scaling=True in get_score_model)
        model_module = (
            score_model.module
            if isinstance(score_model, nn.DataParallel)
            else score_model
        )
        if (
            getattr(model_module, "use_class_condition", False)
            and class_labels is not None
        ):
            x_0_predicted = score_model(x_t, t_clamped, class_labels)
        else:
            x_0_predicted = score_model(x_t, t_clamped)

        # Handle potential NaNs
        if torch.isnan(x_0_predicted).any():
            print(
                f"Warning: NaN in predicted x_0 at t={t.mean().item():.4f}. Replacing with zeros."
            )
            x_0_predicted = torch.nan_to_num(x_0_predicted, nan=0.0)

        # Calculate score using the derived formula
        score = (
            -(x_t - sqrt_alpha_bar_t * x_0_predicted) / one_minus_alpha_bar_t
        )
        return score

    def loss_fn(
        self,
        score_model: nn.Module,
        x_0: torch.Tensor,
        y=None,
        eps: float = 1e-5,
    ) -> torch.Tensor:
        """
        Compute the x_0 prediction loss.
        Assumes the score_model is trained to predict the original clean image x_0.
        Loss = E_t [ || model_output(x_t, t) - x_0 ||^2 ]
        Requires model output *not* to be scaled by sigma(t).
        """
        # Sample time uniformly in [eps, T]
        t = torch.rand(x_0.shape[0], device=x_0.device) * (self.T - eps) + eps
        x_t, noise = self.sample_forward(x_0, t)  # Get noisy sample x_t

        # Get x_0 prediction from model
        # Assumes model output is x_0_hat (disable_final_scaling=True in get_score_model)
        model_module = (
            score_model.module
            if isinstance(score_model, nn.DataParallel)
            else score_model
        )
        if (
            getattr(model_module, "use_class_condition", False)
            and y is not None
        ):
            x_0_predicted = score_model(x_t, t, y)
        else:
            x_0_predicted = score_model(x_t, t)

        # Handle potential NaNs
        if torch.isnan(x_0_predicted).any():
            print(
                f"Warning: NaN in predicted x_0 at t={t.mean().item():.4f} during loss. Replacing with zeros."
            )
            x_0_predicted = torch.nan_to_num(x_0_predicted, nan=0.0)

        # Calculate MSE loss between predicted x_0 and actual x_0
        loss = torch.mean(
            torch.sum((x_0_predicted - x_0) ** 2, dim=list(range(1, x_0.dim())))
        )
        return loss

    # --- SDE Coefficients (Forward and Reverse) ---

    def sde_drift_forward(self) -> Callable:
        """
        Returns the drift function f(x, t) for the forward VP-SDE:
        f(x, t) = -1/2 * β(t) * x
        """
        beta_fn = self.beta

        def _drift(x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            t_clamped = torch.clamp(t, 0.0, self.T).to(x_t.device)
            beta_t = beta_fn(t_clamped).view(-1, *([1] * (x_t.dim() - 1)))
            beta_t = torch.clamp(beta_t, 1e-7, 0.999)  # Stability clamp
            return -0.5 * beta_t * x_t

        return _drift

    def diffusion_squared(self) -> Callable:
        """
        Returns the squared diffusion function g(t)^2 for the VP-SDE:
        g(t)^2 = β(t)
        """
        beta_fn = self.beta

        def _diffusion_sq(t: torch.Tensor) -> torch.Tensor:
            t_clamped = torch.clamp(t, 0.0, self.T).to(
                device
            )  # Assuming t might come from integrator
            beta_t = beta_fn(t_clamped)
            beta_t = torch.clamp(beta_t, min=1e-7)  # Ensure positive
            return beta_t

        return _diffusion_sq

    def sde_drift_reverse(self, score_model: nn.Module) -> Callable:
        """
        Returns the drift function f̄(x, t) for the reverse VP-SDE:
        f̄(x, t) = [-1/2 * β(t) * x - β(t) * ∇log p_t(x)]
                 = [-1/2 * β(t) * x - β(t) * score_fn(x, t)]
        """
        beta_fn = self.beta

        def _drift(
            x_t: torch.Tensor, t: torch.Tensor, class_labels=None
        ) -> torch.Tensor:
            # Ensure t is on the correct device and clamped
            t_clamped = torch.clamp(t, 0.0, self.T).to(x_t.device)
            beta_t = beta_fn(t_clamped).view(-1, *([1] * (x_t.dim() - 1)))
            beta_t = torch.clamp(beta_t, 1e-7, 0.999)  # Stability clamp

            # Calculate score using the score_fn method
            score = self.score_fn(score_model, x_t, t_clamped, class_labels)
            # Clamp score magnitude for stability during integration
            score = torch.clamp(score, -1000, 1000)

            drift_term1 = -0.5 * beta_t * x_t
            drift_term2 = -beta_t * score

            result = drift_term1 + drift_term2
            return torch.clamp(result, -20.0, 20.0)

        return _drift

    def sde_diffusion_reverse(self) -> Callable:
        """
        Returns the diffusion function ḡ(t) for the reverse VP-SDE:
        ḡ(t) = sqrt(β(t))
        """
        beta_fn = self.beta

        def _diffusion(t: torch.Tensor) -> torch.Tensor:
            # Ensure t is on the correct device and clamped
            t_clamped = torch.clamp(t, 0.0, self.T).to(
                device
            )  # Assuming t might come from integrator
            beta_t = beta_fn(t_clamped)
            beta_t = torch.clamp(
                beta_t, min=1e-7
            )  # Ensure positive before sqrt
            return torch.sqrt(beta_t)

        return _diffusion

    # --- Methods for Sampling ---

    # def predict_x0(self, score_model: nn.Module, x_t: torch.Tensor, t: torch.Tensor, class_labels=None) -> torch.Tensor:
    #     """
    #     Directly predict x_0 using the score model.
    #     Assumes the model is configured to output x_0_hat.
    #     """
    #     t_clamped = torch.clamp(t, 0.0, self.T).to(x_t.device)
    #     model_module = score_model.module if isinstance(score_model, nn.DataParallel) else score_model
    #
    #     # Get x_0 prediction from model
    #     if getattr(model_module, 'use_class_condition', False) and class_labels is not None:
    #         x_0_predicted = score_model(x_t, t_clamped, class_labels)
    #     else:
    #         x_0_predicted = score_model(x_t, t_clamped)
    #
    #     # Handle potential NaNs
    #     if torch.isnan(x_0_predicted).any():
    #         print(f"Warning: NaN in predicted x_0 at t={t.mean().item():.4f}. Replacing with zeros.")
    #         x_0_predicted = torch.nan_to_num(x_0_predicted, nan=0.0)
    #
    #     return x_0_predicted
