# -*- coding: utf-8 -*-
"""
Simulate Gaussian processes.

@author: <alberto.suarez@uam.es>
"""
# Load packages

from __future__ import annotations

from typing import Callable, Union

import numpy as np
import torch
from torch import Tensor


def euler_maruyama_integrator(
    x_0: Tensor,
    t_0: float,
    t_end: float,
    n_steps: int,
    drift_coefficient: Callable[float, float],
    diffusion_coefficient: Callable[float],
    seed: Union[int, None] = None,
) -> Tensor:
    """Euler-Maruyama integrator (approximate)

        Args:
        x_0: The initial images of dimensions
            (batch_size, n_channels, image_height, image_width)
        t_0: float,
        t_end: endpoint of the integration interval
        n_steps: number of integration steps
        drift_coefficient: Function of :math`(x(t), t)` that defines the drift term
        diffusion_coefficient: Function of :math`(t)` that defines the diffusion term
        seed: Seed for the random number generator

    Returns:
        x_t: Trajectories that result from the integration of the SDE.
                The shape is (*np.shape(x_0), (n_steps + 1))

    Notes:
        The implementation is fully vectorized except for a loop over time.

    Examples:
        >>> import numpy as np
        >>> drift_coefficient = lambda x_t, t: - x_t
        >>> diffusion_coefficient = lambda t: torch.ones_like(t)
        >>> x_0 = torch.tensor(np.reshape(np.arange(120), (2, 3, 5, 4)))
        >>> t_0, t_end = 0.0, 3.0
        >>> n_steps = 6
        >>> times, x_t = euler_maruyama_integrator(
        ...     x_0, t_0, t_end, n_steps, drift_coefficient, diffusion_coefficient,
        ... )
        >>> print(times)
        tensor([0.0000, 0.5000, 1.0000, 1.5000, 2.0000, 2.5000, 3.0000])
        >>> print(np.shape(x_t))
        torch.Size([2, 3, 5, 4, 7])
    """
    device = x_0.device

    # Create a tensor of time points from t_0 to t_end
    times = torch.linspace(t_0, t_end, n_steps + 1, device=device)
    dt = times[1] - times[0]

    # Initialize the tensor to store the trajectories
    x_t = torch.tensor(
        np.empty((*np.shape(x_0), len(times))),
        dtype=torch.float32,
        device=device,
    )
    x_t[..., 0] = x_0  # Set the initial condition

    z = torch.randn_like(x_t)  # Generate random noise for the diffusion term

    # Loop over each time step (except the last one)
    for n, t in enumerate(times[:-1]):
        t = (
            torch.ones(x_0.shape[0], device=device) * t
        )  # Create a tensor of the current time
        # Update the state using the Euler-Maruyama method
        x_t[..., n + 1] = (
            x_t[..., n]  # Current state
            + drift_coefficient(x_t[..., n], t) * dt  # Drift term
            + diffusion_coefficient(t).view(
                -1, 1, 1, 1
            )  # Diffusion term reshaped for broadcasting
            * torch.sqrt(
                torch.abs(dt)
            )  # Scale by the square root of the time step
            * z[..., n]  # Add random noise
        )

    return times, x_t


class DiffussionProcess:
    """Base class for defining diffusion process coefficients."""

    def __init__(
        self,
        drift_coefficient: Callable[float, float] = lambda x_t, t: 0.0,
        diffusion_coefficient: Callable[float] = lambda t: 1.0,
    ):
        self.drift_coefficient = drift_coefficient
        self.diffusion_coefficient = diffusion_coefficient

    def loss_function(self, score_model, x_0, eps=1e-5):
        """
        Computes the loss for the diffusion process using a time-dependent score model.

        This function calculates the loss based on the difference between the predicted scores
        from the score model and the true scores derived from the original data. The loss is
        computed as the mean squared error of the sum of the scaled predicted score and Gaussian
        noise, which quantifies how well the score model approximates the gradients of the log
        probability density function of the data.

        Args:
            score_model (Callable): A time-dependent score model that takes in the current state
                                    `x_t` and time `t`, returning the estimated score (gradient
                                    of the log probability density).
            x_0 (Tensor): The original data batch, expected shape (batch_size, C, H, W) or similar.
            eps (float, optional): A small constant to avoid division by zero or log of zero
                                   when sampling times. Default is 1e-5.

        Returns:
            Tensor: The computed loss value, which can be used for backpropagation to update
                    the parameters of the score model.

        Example:
            >>> score_model = ...  # Assume a pre-defined score model
            >>> x_0 = torch.randn(32, 3, 64, 64)  # Example batch of images
            >>> loss = self.loss_function(score_model, x_0)
            >>> print(loss.item())  # Output the loss value

        Notes:
            The loss function is crucial for training the score model, as it guides the optimization
            process to improve the model's ability to estimate the score accurately. The choice of
            loss function may vary depending on the specific application and desired properties of
            the diffusion process. The implementation assumes a simple forward diffusion process
            where noise is added to the original data based on a time-dependent scaling factor.
        """

        # 1. Sample times t in [eps, 1]
        t = torch.rand(x_0.shape[0], device=x_0.device) * (1.0 - eps) + eps

        # 2. Sample Gaussian noise
        noise = torch.randn_like(x_0)

        # 3. Compute sigma(t). This might be a direct function or a learned schedule.
        #    Suppose sigma(t) returns shape (batch_size,).
        sigma = self.diffusion_coefficient(t)  # e.g. shape (batch_size,)

        # 4. Reshape sigma to broadcast with x_0
        #    e.g. if x_0 is (batch_size, C, H, W), make sigma (batch_size, 1, 1, 1).
        sigma = sigma.view(x_0.shape[0], *([1] * (x_0.dim() - 1)))

        # 5. Create the noisy sample x_t
        #    If your forward diffusion is x_t = x_0 + sigma(t)*noise (simple case):
        x_t = x_0 + sigma * noise

        # 6. Get the model's predicted score s(x_t, t)
        score = score_model(x_t, t)

        # 7. According to class notes, the loss is:
        #       || sigma(t) * score + noise ||^2
        #    so we just implement that directly:
        #    We sum over all non-batch dimensions, then average over the batch.
        mse_per_example = torch.sum(
            (sigma * score + noise) ** 2, dim=list(range(1, x_0.dim()))
        )
        loss = torch.mean(mse_per_example)

        return loss


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

    Example 1:
        >>> mu, sigma = 1.5, 2.0
        >>> bm = GaussianDiffussionProcess(
        ...     drift_coefficient=lambda x_t, t: mu,
        ...     diffusion_coefficient=lambda t: sigma,
        ...     mu_t=lambda x_0, t: x_0 + mu*t,
        ...     sigma_t=lambda t: np.sqrt(2.0 * t),
        ... )
        >>> print(bm.drift_coefficient(x_t=3.0, t=10.0))  # Output: 1.5
        >>> print(bm.diffusion_coefficient(t=10.0))  # Output: 2.0
        >>> print(bm.mu_t(x_0=3.0, t=10.0), bm.sigma_t(t=10.0))  # Output: 18.0 4.47213595499958

    Notes:
        This class inherits from the `DiffussionProcess` base class and implements the necessary methods
        to define a Gaussian diffusion process. The drift and diffusion coefficients can be constant or
        time-dependent functions, allowing for flexibility in modeling various stochastic processes.
    """

    kind = "Gaussian"

    def __init__(
        self,
        drift_coefficient: Callable[float, float] = lambda x_t, t: 0.0,
        diffusion_coefficient: Callable[float] = lambda t: 1.0,
        mu_t: Callable[float, float] = lambda x_0, t: x_0,
        sigma_t: Callable[float] = lambda t: np.sqrt(t),
    ):
        self.drift_coefficient = drift_coefficient
        self.diffusion_coefficient = diffusion_coefficient
        self.mu_t = mu_t
        self.sigma_t = sigma_t

    def loss_function(self, score_model, x_0, eps=1e-5):
        """
        Computes the loss for the diffusion process using a time-dependent score model.

        This function calculates the loss based on the difference between the predicted scores
        from the score model and the true scores derived from the original data. The loss is
        computed as the mean squared error of the sum of the scaled predicted score and Gaussian
        noise, which quantifies how well the score model approximates the gradients of the log
        probability density function of the data.

        Args:
            score_model (Callable): A time-dependent score model that takes in the current state
                                    `x_t` and time `t`, returning the estimated score (gradient
                                    of the log probability density).
            x_0 (Tensor): The original data batch, expected shape (batch_size, C, H, W) or similar.
            eps (float, optional): A small constant to avoid division by zero or log of zero
                                   when sampling times. Default is 1e-5.

        Returns:
            Tensor: The computed loss value, which can be used for backpropagation to update
                    the parameters of the score model.

        Example:
            >>> score_model = ...  # Assume a pre-defined score model
            >>> x_0 = torch.randn(32, 3, 64, 64)  # Example batch of images
            >>> loss = self.loss_function(score_model, x_0)
            >>> print(loss.item())  # Output the loss value

        Notes:
            The loss function is crucial for training the score model, as it guides the optimization
            process to improve the model's ability to estimate the score accurately. The choice of
            loss function may vary depending on the specific application and desired properties of
            the diffusion process. The implementation assumes a simple forward diffusion process
            where noise is added to the original data based on a time-dependent scaling factor.
        """

        # 1. Sample times t in [eps, 1]
        t = torch.rand(x_0.shape[0], device=x_0.device) * (1.0 - eps) + eps

        # 2. Sample Gaussian noise
        noise = torch.randn_like(x_0)

        # 3. Compute sigma(t). This might be a direct function or a learned schedule.
        #    Suppose sigma(t) returns shape (batch_size,).
        sigma = self.sigma_t(t)  # e.g. shape (batch_size,)

        # 4. Reshape sigma to broadcast with x_0
        #    e.g. if x_0 is (batch_size, C, H, W), make sigma (batch_size, 1, 1, 1).
        sigma = sigma.view(x_0.shape[0], *([1] * (x_0.dim() - 1)))

        # 5. Create the noisy sample x_t
        #    If your forward diffusion is x_t = x_0 + sigma(t)*noise (simple case):
        x_t = x_0 + sigma * noise

        # 6. Get the model's predicted score s(x_t, t)
        score = score_model(x_t, t)

        # 7. According to class notes, the loss is:
        #       || sigma(t) * score + noise ||^2
        #    so we just implement that directly:
        #    We sum over all non-batch dimensions, then average over the batch.
        mse_per_example = torch.sum(
            (sigma * score + noise) ** 2, dim=list(range(1, x_0.dim()))
        )
        loss = torch.mean(mse_per_example)

        return loss


if __name__ == "__main__":
    import doctest

    doctest.testmod()
