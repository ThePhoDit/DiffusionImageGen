import torch
import torch.nn as nn
from typing import Callable

# Import schedule functions (same as VP)
from diffusion.schedules.linear import get_linear_schedule_functions
from diffusion.schedules.cosine import get_cosine_schedule_functions

device = "cuda" if torch.cuda.is_available() else "cpu"


class SubVariancePreservingDiffusionProcess:
    """
    Implements the Sub-Variance Preserving (Sub-VP) SDE diffusion process.
    Uses configurable noise schedules (linear or cosine).

    Based on the provided image:
    Forward SDE: dxt = -1/2 * \beta(t) * xt * dt + g(t) * dWt
        where g(t) = sqrt[ \beta(t) * (1 - exp(-2*Integral[0,t](\beta(s)ds))) ]
            = sqrt[ \beta(t) * (1 - \bar{\alpha}(t)^2) ]
    Reverse SDE: dx\bar{t} = [f(x,t) - g(t)^2 * \nabla log p_t(x\bar{t})] dt
            + g(t) * dW\bar{t}
            = [-1/2 * \beta(t) * x\bar{t} - g(t)^2 * \nabla log p_t(x\bar{t})] dt
            + g(t) * dW\bar{t}
    """

    kind = "Sub-VP"

    def __init__(
        self,
        schedule: str = "linear",
        T: float = 1.0,
        beta_min: float = 0.01,  # Linear schedule param
        beta_max: float = 0.95,  # Linear schedule param
        cosine_s: float = 0.008,  # Cosine schedule param
        eps_var: float = 1e-7,  # Small epsilon for variance stability
    ):
        self.schedule_type = schedule
        self.T = T
        self.eps_var = eps_var
        print(
            f"Initializing Sub-VP diffusion process with {schedule} schedule, T={T}."
        )

        # Get alpha_bar and beta from the chosen schedule
        if schedule == "linear":
            self.beta, self.alpha_bar, _ = get_linear_schedule_functions(
                beta_min=beta_min, beta_max=beta_max, T=T
            )
            self.beta_min = beta_min
            self.beta_max = beta_max
        elif schedule == "cosine":
            self.beta, self.alpha_bar, _ = get_cosine_schedule_functions(
                T=T, s=cosine_s
            )
            self.cosine_s = cosine_s
        else:
            raise ValueError(
                f"Unknown schedule: {schedule}. Choose 'linear' or 'cosine'."
            )

        # --- Sub-VP specific standard deviation --- #
        # sigma(t)^2 = 1 - alpha_bar(t)^2
        def _std_dev_subvp(t):
            alpha_bar_t = self.alpha_bar(t)
            # Clamp alpha_bar away from 1.0 to prevent variance=0 at t=0
            alpha_bar_t = torch.clamp(
                alpha_bar_t, min=0.0, max=1.0 - self.eps_var
            )
            variance = 1.0 - alpha_bar_t**2
            return torch.sqrt(torch.clamp(variance, min=self.eps_var))

        self.std_dev = _std_dev_subvp
        # -------------------------------------------- #

    def marginal_prob(
        self, x_0: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the mean and standard deviation of p(xt | x0) for the Sub-VP SDE.
        \mu(t) = sqrt(\bar{\alpha}(t)) * x_0
        \sigma(t) = sqrt(1 - \bar{\alpha}(t)^2)

        Returns:
            Tuple[mean, std_dev]
        """
        t_clamped = torch.clamp(t, 0.0, self.T).to(x_0.device)
        alpha_bar_t = self.alpha_bar(t_clamped).view(
            -1, *([1] * (x_0.dim() - 1))
        )
        std_dev_t = self.std_dev(t_clamped).view(
            -1, *([1] * (x_0.dim() - 1))
        )  # Use the sub-VP std_dev

        # Clamp alpha_bar_t slightly away from 1.0
        alpha_bar_t_sqrt = torch.sqrt(
            torch.clamp(alpha_bar_t, max=1.0 - self.eps_var)
        )
        mean = alpha_bar_t_sqrt * x_0
        std = std_dev_t
        return mean, std

    def sample_forward(
        self, x_0: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample x_t from x_0 at time t using the forward process analytical solution:
        x_t = sqrt(\bar{\alpha}(t)) * x_0 + sqrt(1 - \bar{\alpha}(t)^2) * \epsilon,
        where \epsilon ~ N(0, I)

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
        Calculate the score \nabla log p_t(xt) assuming the score_model predicts x_0.
        Score = -(x_t - \mu(t)) / \sigma(t)^2
              = -(x_t - sqrt(\bar{\alpha}(t)) * x0_hat) / (1 - \bar{\alpha}(t)^2)
        Requires the model output *not* to be scaled by sigma(t).
        """
        t_clamped = torch.clamp(t, 0.0, self.T).to(x_t.device)
        alpha_bar_t = self.alpha_bar(t_clamped).view(
            -1, *([1] * (x_t.dim() - 1))
        )
        alpha_bar_t = torch.clamp(
            alpha_bar_t, min=0.0, max=1.0 - self.eps_var
        )  # Ensure variance is not zero
        sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
        # variance_t = 1.0 - alpha_bar_t**2
        # (equivalent to self.std_dev(t_clamped)**2)
        variance_t = (
            self.std_dev(t_clamped).view(-1, *([1] * (x_t.dim() - 1))) ** 2
        )
        variance_t = torch.clamp(variance_t, min=self.eps_var)

        # Get x_0 prediction from model
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
                f"Warning: NaN in predicted x_0 at t={t.mean().item():.4f}. "
                f"Replacing with zeros."
            )
            x_0_predicted = torch.nan_to_num(x_0_predicted, nan=0.0)

        # Calculate score using the derived formula for Sub-VP
        score = -(x_t - sqrt_alpha_bar_t * x_0_predicted) / variance_t
        return score

    def loss_fn(
        self,
        score_model: nn.Module,
        x_0: torch.Tensor,
        y=None,
        eps: float = 1e-5,
    ) -> torch.Tensor:
        """
        Compute the x_0 prediction loss (same as VP).
        Assumes the score_model is trained to predict the original clean image x_0.
        Loss = E_t [ || model_output(x_t, t) - x_0 ||^2 ]
        Requires model output *not* to be scaled by sigma(t).
        """
        # Sample time uniformly in [eps, T]
        t = torch.rand(x_0.shape[0], device=x_0.device) * (self.T - eps) + eps
        x_t, noise = self.sample_forward(x_0, t)  # Get noisy sample x_t

        # Get x_0 prediction from model
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
                f"Warning: NaN in predicted x_0 at t={t.mean().item():.4f} during loss. "
                f"Replacing with zeros."
            )
            x_0_predicted = torch.nan_to_num(x_0_predicted, nan=0.0)

        # Calculate MSE loss between predicted x_0 and actual x_0
        loss = torch.mean(
            torch.sum((x_0_predicted - x_0) ** 2, dim=list(range(1, x_0.dim())))
        )
        return loss

    # --- SDE Coefficients (Forward and Reverse) --- #

    def sde_drift_forward(self) -> Callable:
        """
        Returns the drift function f(x, t) for the forward Sub-VP SDE:
        f(x, t) = -1/2 * \beta(t) * x
        (Same as VP)
        """
        beta_fn = self.beta

        def _drift(x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            t_clamped = torch.clamp(t, 0.0, self.T).to(x_t.device)
            beta_t = beta_fn(t_clamped).view(-1, *([1] * (x_t.dim() - 1)))
            beta_t = torch.clamp(beta_t, 1e-7, 0.999)
            return -0.5 * beta_t * x_t

        return _drift

    def diffusion_squared(self) -> Callable:
        """
        Returns the squared diffusion function g(t)^2 for the Sub-VP SDE:
        g(t)^2 = \beta(t) * (1 - \bar{\alpha}(t)^2)
        """
        beta_fn = self.beta
        alpha_bar_fn = self.alpha_bar

        def _diffusion_sq(t: torch.Tensor) -> torch.Tensor:
            t_clamped = torch.clamp(t, 0.0, self.T).to(device)
            beta_t = beta_fn(t_clamped)
            alpha_bar_t = alpha_bar_fn(t_clamped)
            alpha_bar_t = torch.clamp(
                alpha_bar_t, min=0.0, max=1.0 - self.eps_var
            )
            diffusion_sq_val = beta_t * (1.0 - alpha_bar_t**2)
            return torch.clamp(
                diffusion_sq_val, min=self.eps_var
            )  # Ensure positive

        return _diffusion_sq

    def sde_drift_reverse(self, score_model: nn.Module) -> Callable:
        """
        Returns the drift function f\bar{x}(x, t) for the reverse Sub-VP SDE:
        f\bar{x}(x, t) = [-1/2 * \beta(t) * x - g(t)^2 * \nabla log p_t(x)]
                = [-1/2 * \beta(t) * x - g(t)^2 * score_fn(x, t)]
        """
        beta_fn = self.beta
        diffusion_sq_fn = self.diffusion_squared()  # Get g(t)^2 function

        def _drift(
            x_t: torch.Tensor, t: torch.Tensor, class_labels=None
        ) -> torch.Tensor:
            t_clamped = torch.clamp(t, 0.0, self.T).to(x_t.device)
            beta_t = beta_fn(t_clamped).view(-1, *([1] * (x_t.dim() - 1)))
            beta_t = torch.clamp(beta_t, 1e-7, 0.999)
            g2_t_val = diffusion_sq_fn(t_clamped).view(
                -1, *([1] * (x_t.dim() - 1))
            )
            g2_t_val = torch.clamp(g2_t_val, min=self.eps_var)

            # Calculate score using the Sub-VP score_fn method
            score = self.score_fn(score_model, x_t, t_clamped, class_labels)
            score = torch.clamp(score, -1000, 1000)  # Clamp score magnitude

            drift_term1 = -0.5 * beta_t * x_t
            drift_term2 = -g2_t_val * score

            result = drift_term1 + drift_term2
            # Clamping the final drift might need adjustment based on typical magnitudes
            return torch.clamp(result, -20.0, 20.0)

        return _drift

    def sde_diffusion_reverse(self) -> Callable:
        """
        Returns the diffusion function \bar{g}(t) for the reverse Sub-VP SDE:
        \bar{g}(t) = g(t) = sqrt[ \beta(t) * (1 - \bar{\alpha}(t)^2) ]
        """
        diffusion_sq_fn = self.diffusion_squared()  # Get g(t)^2 function

        def _diffusion(t: torch.Tensor) -> torch.Tensor:
            g2_t_val = diffusion_sq_fn(t)  # Already clamped positive
            return torch.sqrt(g2_t_val)

        return _diffusion
