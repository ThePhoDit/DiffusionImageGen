# -*- coding: utf-8 -*-
"""
Created on Sat Mar  8 21:12:58 2025

Code adapted by alberto.suarez@uam.es from
https://yang-song.net/blog/2021/score/


"""

import torch
import torch.nn as nn
import numpy as np
import warnings


class GaussianRandomFourierFeatures(nn.Module):
    """Gaussian random Fourier features for encoding time steps."""

    def __init__(self, embed_dim, scale=30.0):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.rff_weights = nn.Parameter(
            torch.randn(embed_dim // 2) * scale,
            requires_grad=False,
        )

    def forward(self, x):
        x_proj = x[:, None] * self.rff_weights[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Dense(nn.Module):
    """A fully connected layer that reshapes outputs to feature maps."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.dense(x)[..., None, None]


class ScoreNet(nn.Module):
    """A time-dependent score-based model built upon U-Net architecture with optional class conditioning."""

    def __init__(
        self,
        marginal_prob_std,
        channels=[32, 64, 128, 256],
        embed_dim=256,
        image_channels=3,
        num_classes=None,  # Set to number of classes for conditioning (e.g., 10 for MNIST)
        class_embed_dim=128,  # Dimension for class embeddings
        disable_final_scaling=False,  # Store the flag
    ):
        """Initialize a time-dependent score-based network.

        Args:
          marginal_prob_std: A function that takes time t and gives the standard
            deviation of the perturbation kernel p_{0t}(x(t) | x(0)).
          channels: The number of channels for feature maps of each resolution.
          embed_dim: The dimensionality of Gaussian random Fourier feature embeddings.
          image_channels: Number of input/output channels (3 for RGB, 1 for grayscale).
          num_classes: If provided, enables class conditioning with this many classes.
          class_embed_dim: Dimension of the learnable class embeddings.
          disable_final_scaling: If True, the final output is not divided by marginal_prob_std(t).
                                Useful for noise prediction tasks where the raw network output is desired.
        """
        super().__init__()
        self.marginal_prob_std = marginal_prob_std
        self.num_classes = num_classes
        self.use_class_condition = num_classes is not None
        self.disable_final_scaling = disable_final_scaling  # Store the flag

        # Time embedding
        self.embed = nn.Sequential(
            GaussianRandomFourierFeatures(embed_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim),
        )

        # Class embedding (if enabled)
        if self.use_class_condition:
            self.class_embedding = nn.Embedding(num_classes, class_embed_dim)
            # Project class embedding to match time embedding dimension
            self.class_projection = nn.Linear(class_embed_dim, embed_dim)
        else:
            self.class_embedding = None
            self.class_projection = None

        # Encoding layers where the resolution decreases
        self.conv1 = nn.Conv2d(
            image_channels, channels[0], 3, stride=1, padding=1, bias=False
        )
        self.dense1 = Dense(embed_dim, channels[0])
        self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])
        self.conv2 = nn.Conv2d(
            channels[0], channels[1], 3, stride=2, padding=1, bias=False
        )
        self.dense2 = Dense(embed_dim, channels[1])
        self.gnorm2 = nn.GroupNorm(32, num_channels=channels[1])
        self.conv3 = nn.Conv2d(
            channels[1], channels[2], 3, stride=2, padding=1, bias=False
        )
        self.dense3 = Dense(embed_dim, channels[2])
        self.gnorm3 = nn.GroupNorm(32, num_channels=channels[2])
        self.conv4 = nn.Conv2d(
            channels[2], channels[3], 3, stride=2, padding=1, bias=False
        )
        self.dense4 = Dense(embed_dim, channels[3])
        self.gnorm4 = nn.GroupNorm(32, num_channels=channels[3])

        # Decoding layers where the resolution increases
        self.tconv4 = nn.ConvTranspose2d(
            channels[3],
            channels[2],
            3,
            stride=2,
            padding=1,
            output_padding=1,
            bias=False,
        )
        self.dense5 = Dense(embed_dim, channels[2])
        self.tgnorm4 = nn.GroupNorm(32, num_channels=channels[2])
        self.tconv3 = nn.ConvTranspose2d(
            channels[2],
            channels[1],
            3,
            stride=2,
            padding=1,
            output_padding=1,
            bias=False,
        )
        self.dense6 = Dense(embed_dim, channels[1])
        self.tgnorm3 = nn.GroupNorm(32, num_channels=channels[1])
        self.tconv2 = nn.ConvTranspose2d(
            channels[1],
            channels[0],
            3,
            stride=2,
            padding=1,
            output_padding=1,
            bias=False,
        )
        self.dense7 = Dense(embed_dim, channels[0])
        self.tgnorm2 = nn.GroupNorm(32, num_channels=channels[0])
        self.tconv1 = nn.ConvTranspose2d(
            channels[0], image_channels, 3, stride=1, padding=1, bias=False
        )

        # The swish activation function
        self.act = lambda x: x * torch.sigmoid(x)

    def forward(self, x, t, class_labels=None):
        """
        Forward pass of the score network.

        Args:
            x: Input tensor [batch, channels, height, width]
            t: Time tensor [batch]
            class_labels: Optional class labels [batch] for conditioning.
        """
        # Obtain the time embedding
        time_embed = self.act(self.embed(t))

        # Combine with class embedding if applicable
        if self.use_class_condition:
            if class_labels is None:
                # Use zero embedding for unconditional case or classifier-free guidance dropout
                warnings.warn(
                    "Class conditioning enabled, but no class labels provided. Using zero embedding.",
                    UserWarning,
                )
                class_embed = torch.zeros_like(time_embed)
            else:
                class_embed = self.class_embedding(
                    class_labels
                )  # [batch, class_embed_dim]
                class_embed = self.class_projection(
                    class_embed
                )  # [batch, embed_dim]

            embed = time_embed + class_embed  # Combine embeddings
        else:
            embed = time_embed  # Use only time embedding

        # Encoding path
        h1 = self.conv1(x)
        h1 += self.dense1(embed)
        h1 = self.gnorm1(h1)
        h1 = self.act(h1)

        h2 = self.conv2(h1)
        h2 += self.dense2(embed)
        h2 = self.gnorm2(h2)
        h2 = self.act(h2)

        h3 = self.conv3(h2)
        h3 += self.dense3(embed)
        h3 = self.gnorm3(h3)
        h3 = self.act(h3)

        h4 = self.conv4(h3)
        h4 += self.dense4(embed)
        h4 = self.gnorm4(h4)
        h4 = self.act(h4)

        # Decoding path
        h = self.tconv4(h4)
        h += self.dense5(embed)
        h = self.tgnorm4(h)
        h = self.act(h)

        # Add skip connections
        if (
            h.size() != h3.size()
        ):  # Adjust size if necessary (can happen with odd dimensions)
            h = torch.nn.functional.interpolate(
                h, size=h3.size()[2:], mode="bilinear", align_corners=False
            )
        h = h + h3

        h = self.tconv3(h)
        h += self.dense6(embed)
        h = self.tgnorm3(h)
        h = self.act(h)

        if h.size() != h2.size():
            h = torch.nn.functional.interpolate(
                h, size=h2.size()[2:], mode="bilinear", align_corners=False
            )
        h = h + h2

        h = self.tconv2(h)
        h += self.dense7(embed)
        h = self.tgnorm2(h)
        h = self.act(h)

        if h.size() != h1.size():
            h = torch.nn.functional.interpolate(
                h, size=h1.size()[2:], mode="bilinear", align_corners=False
            )
        h = h + h1

        # Final layer
        h = self.tconv1(h)

        # Optionally normalize output by standard deviation
        if not self.disable_final_scaling:
            # Add epsilon to std dev for numerical stability
            std = self.marginal_prob_std(t)[:, None, None, None] + 1e-5
            h = h / std

        return h
