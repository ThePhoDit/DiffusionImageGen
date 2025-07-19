import unittest
import torch
from unittest.mock import patch, MagicMock
import numpy as np

# Mock matplotlib before importing the utility functions
import sys

mock_matplotlib = MagicMock()
sys.modules["matplotlib"] = mock_matplotlib
sys.modules["matplotlib.pyplot"] = mock_matplotlib.pyplot
sys.modules["matplotlib.animation"] = mock_matplotlib.animation
sys.modules["matplotlib.colors"] = mock_matplotlib.colors

from diffusion.utils.diffusion_utilities import (
    plot_image_grid,
    plot_image_evolution,
    animation_images,
)


class TestDiffusionUtilities(unittest.TestCase):

    @patch("diffusion.utils.diffusion_utilities.plt")
    @patch("diffusion.utils.diffusion_utilities.make_grid")
    @patch("diffusion.utils.diffusion_utilities.functional")
    def test_plot_image_grid(self, mock_functional, mock_make_grid, mock_plt):
        """Test plot_image_grid execution and return types."""
        # Arrange
        batch_size = 4
        channels = 3
        height, width = 16, 16
        images = torch.randn(batch_size, channels, height, width)
        figsize = (8, 8)
        n_rows = 2
        n_cols = 2

        # Mock return values
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        mock_grid_tensor = torch.randn(
            channels, height * n_rows, width * n_cols
        )
        mock_make_grid.return_value = mock_grid_tensor
        mock_pil_image = MagicMock()
        mock_functional.to_pil_image.return_value = mock_pil_image

        # Act
        fig, ax = plot_image_grid(images, figsize, n_rows, n_cols)

        # Assert
        mock_plt.subplots.assert_called_once_with(figsize=figsize)
        mock_make_grid.assert_called_once()
        mock_functional.to_pil_image.assert_called_once_with(mock_grid_tensor)
        mock_ax.imshow.assert_called_once_with(mock_pil_image)
        mock_ax.axis.assert_called_once_with("off")
        self.assertEqual(fig, mock_fig)
        self.assertEqual(ax, mock_ax)

    @patch("diffusion.utils.diffusion_utilities.plt")
    def test_plot_image_evolution(self, mock_plt):
        """Test plot_image_evolution execution and return types."""
        # Arrange
        n_images = 2
        channels = 1
        height, width = 8, 8
        n_steps = 10
        images = torch.randn(n_images, channels, height, width, n_steps)
        n_intermediate_steps = [0, n_steps // 2, n_steps - 1]
        figsize = (10, 5)

        # Mock return values
        mock_fig = MagicMock()
        # Create a list of lists of mocks for axs
        mock_axes_list = [
            [MagicMock() for _ in range(len(n_intermediate_steps))]
            for _ in range(n_images)
        ]
        # Make plt.subplots return the mocks as a list of lists
        mock_plt.subplots.return_value = (mock_fig, mock_axes_list)

        # Act
        fig, axs = plot_image_evolution(
            images, n_images, n_intermediate_steps, figsize, cmap="gray"
        )

        # Assert
        mock_plt.subplots.assert_called_once_with(
            n_images, len(n_intermediate_steps), figsize=figsize, squeeze=False
        )
        call_count_imshow = 0
        call_count_axisoff = 0
        for r in range(n_images):
            for c in range(len(n_intermediate_steps)):
                ax_mock = mock_axes_list[r][c]
                call_count_imshow += ax_mock.imshow.call_count
                call_count_axisoff += ax_mock.set_axis_off.call_count

        self.assertEqual(
            call_count_imshow, n_images * len(n_intermediate_steps)
        )
        self.assertEqual(
            call_count_axisoff, n_images * len(n_intermediate_steps)
        )
        self.assertEqual(fig, mock_fig)
        # Compare the structure if needed, or just ensure the type is right
        self.assertEqual(axs, mock_axes_list)

    @patch("diffusion.utils.diffusion_utilities.plt")
    @patch("diffusion.utils.diffusion_utilities.animation")
    def test_animation_images(self, mock_animation, mock_plt):
        """Test animation_images execution and return types."""
        # Arrange
        channels = 3
        height, width = 8, 8
        n_frames = 5
        # Shape: H, W, C, T
        images_t = torch.randn(height, width, channels, n_frames)
        interval = 100
        figsize = (5, 5)

        # Mock return values
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        mock_img_display = MagicMock()
        mock_ax.imshow.return_value = mock_img_display
        mock_anim_obj = MagicMock()
        mock_animation.FuncAnimation.return_value = mock_anim_obj

        # Act
        fig, ax, anim = animation_images(images_t, interval, figsize)

        # Assert
        mock_plt.subplots.assert_called_once_with(figsize=figsize)
        mock_ax.imshow.assert_called_once()
        mock_animation.FuncAnimation.assert_called_once()
        mock_plt.close.assert_called_once_with(mock_fig)
        self.assertEqual(fig, mock_fig)
        self.assertEqual(ax, mock_ax)
        self.assertEqual(anim, mock_anim_obj)


if __name__ == "__main__":
    import os

    current_dir = os.path.dirname(os.path.abspath(__file__))
    tests_dir = os.path.dirname(current_dir)
    utils_test_dir = os.path.join(tests_dir, "utils")

    if not os.path.exists(tests_dir):
        os.makedirs(tests_dir)
    if not os.path.exists(os.path.join(tests_dir, "__init__.py")):
        open(os.path.join(tests_dir, "__init__.py"), "a").close()

    if not os.path.exists(utils_test_dir):
        os.makedirs(utils_test_dir)
    if not os.path.exists(os.path.join(utils_test_dir, "__init__.py")):
        open(os.path.join(utils_test_dir, "__init__.py"), "a").close()

    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestDiffusionUtilities))
    runner = unittest.TextTestRunner()
    runner.run(suite)
