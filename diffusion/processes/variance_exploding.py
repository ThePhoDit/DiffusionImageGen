from functools import partial
from typing import Callable

import torch
from torch.utils.data import DataLoader

from torch import Tensor

from diffusion.samplers.euler_maruyama import euler_maruyama_integrator
from diffusion.models.score_model import ScoreNet

from torch.optim import Adam
from tqdm.notebook import trange
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
# cpu
n_threads = torch.get_num_threads()

# Given parameter
sigma = 25.0


# Define the drift coefficient: for standard Brownian motion the drift is zero.
def bm_drift_coefficient(x_t, t):
    return torch.zeros_like(x_t)


# Define the instantaneous diffusion coefficient.
# We want the variance accumulated over time t to be:
#     Var[X_t] = \int_{0}^{t} (\sigma^s)^2 ds
#              = (\sigma^{2t} - 1) / (2 * \ln(\sigma))
# so we set the instantaneous diffusion coefficient as:
def bm_diffusion_coefficient(t, sigma):
    return sigma**t


# The mean function: with zero drift the mean remains the initial value.
def bm_mu_t(x_0, t):
    return x_0


# Provided function for overall standard deviation at time t.
def bm_sigma_t(t, sigma):
    log_sigma = torch.log(torch.tensor(sigma, dtype=torch.float32))
    return torch.sqrt(0.5 * (sigma ** (2 * t) - 1.0) / log_sigma)


# Prepare the functions to be passed to the diffusion process.
# We partially apply the functions that depend on sigma.
drift_coefficient = bm_drift_coefficient
diffusion_coefficient = lambda t: bm_diffusion_coefficient(t, sigma)
mu_t = bm_mu_t
sigma_t = lambda t: bm_sigma_t(t, sigma)


class DiffussionProcess:

    def __init__(
        self,
        drift_coefficient: Callable[[float, float], float] = lambda x_t, t: 0.0,
        diffusion_coefficient: Callable[[float], float] = lambda t: 1.0,
    ):
        self.drift_coefficient = drift_coefficient
        self.diffusion_coefficient = diffusion_coefficient


class GaussianDiffussionProcess(DiffussionProcess):
    """
    A class representing a Gaussian diffusion process, which models the evolution of a stochastic process
    with a constant drift and diffusion coefficient. This process can be used to simulate Brownian motion
    or other diffusion phenomena in a variety of applications.

    Attributes:
        drift_coefficient (Callable): A function that defines the drift term of the process.
        diffusion_coefficient (Callable): A function that defines the diffusion term of the process.
        mu_t (Callable): A function that computes the expected value of the process at time t given the initial state x_0.
        sigma_t (Callable): A function that computes the standard deviation of the process at time t.

    Args:
        drift_coefficient (Callable): A function of the form `f(x_t, t)` that returns the drift term at state `x_t` and time `t`.
        diffusion_coefficient (Callable): A function of the form `g(t)` that returns the diffusion coefficient at time `t`.
        mu_t (Callable): A function of the form `h(x_0, t)` that computes the expected value of the process at time `t`.
        sigma_t (Callable): A function of the form `k(t)` that computes the standard deviation of the process at time `t`.
    """

    kind = "Gaussian"

    def __init__(
        self,
        drift_coefficient: Callable[[float, float], float] = lambda x_t, t: 0.0,
        diffusion_coefficient: Callable[[float], float] = lambda t: 1.0,
        mu_t: Callable[[float, float], float] = lambda x_0, t: x_0,
        sigma_t: Callable[[float], float] = lambda t: torch.sqrt(t),
    ):
        super().__init__(drift_coefficient, diffusion_coefficient)
        self.mu_t = mu_t
        self.sigma_t = sigma_t

    def loss_function(self, score_model, x_0, y=None, eps=1e-5):
        """
        Computes the loss for the diffusion process, optionally using class labels.
        """
        t = torch.rand(x_0.shape[0], device=x_0.device) * (1.0 - eps) + eps
        noise = torch.randn_like(x_0)
        sigma = self.sigma_t(t).view(x_0.shape[0], *([1] * (x_0.dim() - 1)))
        x_t = x_0 + sigma * noise

        # Pass class labels to the model if it supports conditioning
        if (
            getattr(score_model.module, "use_class_condition", False)
            and y is not None
        ):
            score = score_model(x_t, t, y)
        else:
            score = score_model(x_t, t)

        mse_per_example = torch.sum(
            (sigma * score + noise) ** 2, dim=list(range(1, x_0.dim()))
        )
        loss = torch.mean(mse_per_example)
        return loss


# Create an instance of the Gaussian diffusion process
diffusion_process = GaussianDiffussionProcess(
    drift_coefficient=drift_coefficient,
    diffusion_coefficient=diffusion_coefficient,
    mu_t=mu_t,
    sigma_t=sigma_t,
)


def get_score_model(num_classes=None):
    """Get a score model, optionally configured for class conditioning."""
    score_model_instance = ScoreNet(
        marginal_prob_std=partial(bm_sigma_t, sigma=sigma),
        image_channels=3,
        channels=[64, 128, 256, 512],
        num_classes=num_classes,  # Pass num_classes to enable conditioning
    )
    score_model = torch.nn.DataParallel(score_model_instance)
    return score_model.to(device)


def load_model(name: str = "check_point.pth", num_classes=None):
    """Load a trained score model, specifying if it was class-conditioned."""
    model = get_score_model(
        num_classes=num_classes
    )  # Pass num_classes when loading
    model.load_state_dict(torch.load(name, map_location=torch.device(device)))
    model.eval()
    return model


def train(
    data_train,
    batch_size=32,
    n_epochs=10,
    learning_rate=1.0e-3,
    save_model_to: str = "check_point.pth",
    use_class_condition: bool = False,  # Add flag to control conditioning
    num_classes: int = 10,  # Default to 10 for MNIST/CIFAR
):
    """Train the score model, optionally with class conditioning."""
    # Get the model, enabling class conditioning if requested
    score_model = get_score_model(
        num_classes=num_classes if use_class_condition else None
    )

    data_loader = DataLoader(
        data_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_threads,
    )
    optimizer = Adam(score_model.parameters(), lr=learning_rate)
    tqdm_epoch = trange(n_epochs, desc="Training epochs")

    for epoch in tqdm_epoch:
        avg_loss = 0.0
        num_items = 0
        batch_progress = tqdm(
            data_loader, desc=f"Epoch {epoch+1}/{n_epochs}", leave=False
        )

        for batch_idx, (x, y) in enumerate(batch_progress):
            x = x.to(device)
            # Pass labels to loss function if using conditioning
            y_cond = y.to(device) if use_class_condition else None

            loss = diffusion_process.loss_function(score_model, x, y_cond)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_loss += loss.item() * x.shape[0]
            num_items += x.shape[0]
            batch_progress.set_postfix(
                {
                    "batch_loss": f"{loss.item():.4f}",
                    "avg_loss": (
                        f"{avg_loss/num_items:.4f}" if num_items > 0 else "N/A"
                    ),
                }
            )

        epoch_loss = avg_loss / max(num_items, 1)
        tqdm_epoch.set_postfix({"epoch_loss": f"{epoch_loss:.4f}"})

        if (epoch % 50 == 0) and epoch != 0:
            file_parts = save_model_to.split(".")
            chkpt_name = f"{'.'.join(file_parts[:-1])}_{epoch}.{file_parts[-1]}"
            torch.save(score_model.state_dict(), chkpt_name)
            print(f"\nSaved checkpoint: {chkpt_name}")

    torch.save(score_model.state_dict(), save_model_to)
    print(f"\nTraining completed. Final model saved to {save_model_to}")
    return score_model


def backward_drift_coefficient(score_model):
    """Compute backward drift, potentially conditioned on class labels."""

    def _backward_drift_coefficient(
        x_t, t, diffusion_coefficient, class_labels=None
    ):
        g_t = diffusion_coefficient(t)
        g_t = g_t.view(-1, 1, 1, 1)  # Reshape for broadcasting
        beta_t = g_t**2

        # Pass class labels to the model forward pass if conditioning is enabled
        if (
            getattr(score_model.module, "use_class_condition", False)
            and class_labels is not None
        ):
            score = score_model(x_t, t, class_labels)
        else:
            score = score_model(x_t, t)

        drift = -(beta_t * score)
        return drift

    return _backward_drift_coefficient


def generate_images(
    score_model,
    n_images: int = 3,
    final_time: float = 1.0,
    target_class: int = None,  # The desired class index
    image_size: tuple = (28, 28),
    n_channels: int = 3,
    n_steps: int = 500,
) -> Tensor:
    """
    Generate images using the score model, optionally conditioned on a target class.
    Returns only the final generated images.

    Args:
        score_model: The score model used to generate images.
        n_images: Number of images to generate.
        final_time: The final time for the diffusion process.
        target_class: The desired class index for the generated images.
        image_size: The size of the generated images.
        n_channels: The number of channels in the generated images.
        n_steps: Number of steps for the Euler-Maruyama sampler.
    """
    T = final_time
    image_T = torch.randn(
        n_images, n_channels, image_size[0], image_size[1], device=device
    )

    # Prepare class labels tensor if conditioning is requested and model supports it
    class_labels = None
    model_is_conditional = getattr(
        score_model.module, "use_class_condition", False
    )

    if target_class is not None:
        if model_is_conditional:
            print(
                f"Generating images for class {target_class} using conditioning."
            )
            class_labels = torch.full(
                (n_images,), target_class, dtype=torch.long, device=device
            )
        else:
            print(
                f"Warning: Model was not trained with class conditioning. Generating unconditioned images."
            )
            pass  # Continue with unconditional generation

    with torch.no_grad():
        # Wrapper for drift coefficient to pass class labels
        def drift_fn_wrapper(x, t):
            drift_calculator = backward_drift_coefficient(score_model)
            return drift_calculator(x, t, diffusion_coefficient, class_labels)

        # Run the integrator
        times, synthetic_images_t = euler_maruyama_integrator(
            image_T,
            t_0=T,
            t_end=1.0e-3,
            n_steps=n_steps,
            drift_coefficient=drift_fn_wrapper,
            diffusion_coefficient=diffusion_coefficient,
        )

        # --- Correctly select only the final images ---
        # Get the state at the last time step by indexing the last dimension
        final_images = synthetic_images_t[..., -1]
        # final_images should now have shape [N, C, H, W]

    # Ensure output is in [0, 1] range (important for visualization)
    # Check if normalization was applied during training - if so, reverse it here.
    # Assuming output range should be [0, 1] for plotting.
    final_images = torch.clamp(final_images, 0.0, 1.0)

    return final_images  # Return only the final images tensor [N, C, H, W]
