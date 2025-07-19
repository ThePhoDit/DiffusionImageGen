# -*- coding: utf-8 -*-
"""
Plotting and animation utilities for visualizing image data and diffusion processes.

Provides functions for creating grids of images, plotting the evolution of images
over diffusion steps, and generating animations from sequences of images.
"""

from numpy.typing import ArrayLike

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

from matplotlib.colors import Colormap

import torch
from torchvision.utils import make_grid
from torchvision.transforms import functional


def plot_image_grid(
    images: torch.Tensor,
    figsize: tuple,
    n_rows: int,
    n_cols: int,
    padding: int = 2,
    pad_value: float = 1.0,
    cmap: Colormap = None,
    normalize: bool = False,
    axis_on_off: str = "off",
):
    """Plots a grid of images using matplotlib.

    Args:
        images: A tensor of images (batch) with shape [N, C, H, W].
        figsize: Figure size tuple (width, height) for matplotlib.
        n_rows: Number of rows in the grid (Note: `make_grid` uses `nrow` which corresponds to n_cols).
        n_cols: Number of columns in the grid (`nrow` argument for `make_grid`).
        padding: Amount of padding between images in the grid.
        pad_value: Value used for padding.
        cmap: Colormap to use for grayscale images (e.g., 'gray'). If None, defaults
              to 'gray' for single-channel images.
        normalize: Whether to normalize the image pixel values using `make_grid`'s logic.
        axis_on_off: Whether to display axes ('on' or 'off').

    Returns:
        A tuple (fig, ax) containing the matplotlib Figure and Axes objects.
    """
    grid = make_grid(
        images,
        nrow=n_cols,
        padding=padding,
        normalize=normalize,
        pad_value=pad_value,
    )

    # Convert to PIL Image and display
    fig, ax = plt.subplots(figsize=figsize)
    if images.shape[1] == 1:  # Grayscale image
        ax.imshow(functional.to_pil_image(grid), cmap=cmap or "gray")
    else:  # Color image
        ax.imshow(functional.to_pil_image(grid))
    ax.axis("off")
    return fig, ax


def plot_image_evolution(
    images: torch.Tensor,
    n_images: int,
    n_intermediate_steps: ArrayLike,
    figsize: tuple,
    cmap: Colormap = None,
):
    """Plots the evolution of multiple images at specified intermediate steps.

    Assumes `images` tensor has shape [N, C, H, W, T], where T is the number of steps.

    Args:
        images: Tensor containing sequences of images over time.
                Shape: [n_images, channels, height, width, n_steps].
        n_images: The number of individual image sequences to plot (rows in the grid).
        n_intermediate_steps: An array-like object containing the indices of the time
                               steps (last dimension) to display for each image sequence.
        figsize: Figure size tuple (width, height) for matplotlib.
        cmap: Colormap to use for grayscale images. Defaults to 'gray' if None.

    Returns:
        A tuple (fig, axs) containing the matplotlib Figure and Axes objects array.
    """
    images = images.cpu().numpy()
    fig, axs = plt.subplots(
        n_images,
        len(n_intermediate_steps),
        figsize=figsize,
        squeeze=False,  # Ensure axs is always 2D, even if n_images or len(steps) is 1
    )

    # Direct indexing is safer than iterators for potentially non-flat arrays
    for img_idx in range(n_images):
        for step_list_idx, step_array_idx in enumerate(n_intermediate_steps):
            # Use chained list indexing for list of lists
            ax = axs[img_idx][step_list_idx]
            if images.shape[1] == 1:  # Grayscale image
                ax.imshow(
                    images[img_idx, 0, :, :, step_array_idx],
                    cmap=cmap or "gray",
                )
            else:  # Color image
                ax.imshow(
                    images[img_idx, :, :, :, step_array_idx].transpose(1, 2, 0)
                )
            ax.set_axis_off()

    return fig, axs


def animation_images(
    images_t: torch.Tensor,
    interval: int,
    figsize: tuple,
):
    """Creates a matplotlib animation from a sequence of images.

    Assumes `images_t` tensor has shape compatible with plotting frames over time.
    Handles both grayscale [H, W, T] and color [C, H, W, T] or [H, W, C, T].
    The time/frame dimension should be the last one.

    Args:
        images_t: Tensor containing the sequence of images over time.
                  Example shapes: [H, W, T], [C, H, W, T], [H, W, C, T].
        interval: Delay between frames in milliseconds.
        figsize: Figure size tuple (width, height) for matplotlib.

    Returns:
        A tuple (fig, ax, anim) containing the matplotlib Figure, Axes,
        and FuncAnimation objects.
    """
    # Convert to numpy and determine shape/frames
    images_np = images_t.cpu().numpy()
    num_dims = images_np.ndim
    n_frames = images_np.shape[-1]

    # Determine if grayscale or color and initial frame shape
    if num_dims == 3:  # Grayscale: H, W, T
        is_grayscale = True
        initial_frame = images_np[:, :, 0]
    elif num_dims == 4:  # Color: C, H, W, T or H, W, C, T
        is_grayscale = False
        # Check if channels are first or third dimension (excluding time)
        if images_np.shape[0] in [1, 3]:  # C, H, W, T -> transpose to H, W, C
            initial_frame = images_np[:, :, :, 0].transpose(1, 2, 0)
        elif images_np.shape[2] in [1, 3]:  # H, W, C, T
            initial_frame = images_np[:, :, :, 0]
        else:
            raise ValueError(
                f"Unsupported color image shape: {images_np.shape}"
            )
    else:
        raise ValueError(f"Unsupported image shape: {images_np.shape}")

    # Create a figure and axes.
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")  # Turn off axis for cleaner animation
    img_display = ax.imshow(
        initial_frame, cmap=("gray" if is_grayscale else None)
    )

    def update(t):
        """Update function for the animation frame."""
        if is_grayscale:
            frame_data = images_np[:, :, t]
        else:
            # Handle C, H, W, T vs H, W, C, T based on original shape
            if images_np.shape[0] in [1, 3]:  # C, H, W, T
                frame_data = images_np[:, :, :, t].transpose(1, 2, 0)
            else:  # H, W, C, T
                frame_data = images_np[:, :, :, t]

        img_display.set_array(frame_data)
        return [img_display]

    # Create and return animation object
    anim = animation.FuncAnimation(
        fig,
        update,
        frames=n_frames,
        interval=interval,
        blit=True,  # Use blit=True if possible
    )
    plt.close(fig)  # Prevent duplicate static plot display in some environments

    return fig, ax, anim
