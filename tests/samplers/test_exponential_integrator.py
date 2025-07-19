import unittest
import torch
from unittest.mock import MagicMock

from diffusion.samplers.exponential_integrator import (
    exponential_integrator_sampler,
)


class TestExponentialIntegratorSampler(unittest.TestCase):

    def setUp(self):
        """Set up mock VP-like diffusion process, score model, and tensors."""
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.batch_size = 2
        self.channels = 1
        self.height = 8
        self.width = 8
        self.x_T = torch.randn(
            self.batch_size,
            self.channels,
            self.height,
            self.width,
            device=self.device,
        )
        self.t_0 = 1.0
        self.t_end = 1e-3
        self.n_steps = 5

        # Mock ScoreNet model
        self.mock_score_model = MagicMock()
        self.mock_score_model.return_value = torch.randn_like(self.x_T) * 0.1
        self.mock_score_model.module = self.mock_score_model
        self.mock_score_model.use_class_condition = False

        # Mock VP-like DiffusionProcess
        self.mock_diffusion_process = MagicMock()
        self.mock_diffusion_process.kind = "VP"  # Identify as VP

        # Define behavior for methods needed by exponential_integrator_sampler
        # 1. beta:
        self.mock_diffusion_process.beta = MagicMock(
            return_value=torch.ones(self.batch_size, device=self.device) * 0.01
        )

        # 2. alpha_bar:
        # Simulate decreasing alpha_bar
        def mock_alpha_bar_fn(t):
            # Simple linear decrease from ~1 to ~0 for testing
            t_clamped = torch.clamp(t, 0.0, 1.0)
            return 1.0 - t_clamped * 0.99

        self.mock_diffusion_process.alpha_bar = MagicMock(
            side_effect=mock_alpha_bar_fn
        )

        # 3. score_fn:
        def mock_score_logic(*args, **kwargs):
            # args[0] is model, args[1] is x, args[2] is t
            return self.mock_score_model.return_value

        self.mock_diffusion_process.score_fn = MagicMock(
            side_effect=mock_score_logic
        )

    def test_ei_sampler_execution_shape(self):
        """Test that exponential integrator runs and returns correct shapes."""
        times, x_t_traj = exponential_integrator_sampler(
            diffusion_process=self.mock_diffusion_process,
            score_model=self.mock_score_model,
            x_T=self.x_T,
            t_0=self.t_0,
            t_end=self.t_end,
            n_steps=self.n_steps,
            use_notebook_tqdm=False,
        )

        expected_time_shape = (self.n_steps + 1,)
        expected_traj_shape = (
            self.batch_size,
            self.channels,
            self.height,
            self.width,
            self.n_steps + 1,
        )

        self.assertEqual(times.shape, expected_time_shape)
        self.assertEqual(x_t_traj.shape, expected_traj_shape)

        # Check if the process/model methods were called
        self.assertTrue(
            self.mock_diffusion_process.beta.call_count >= self.n_steps
        )
        self.assertTrue(
            self.mock_diffusion_process.alpha_bar.call_count >= 2 * self.n_steps
        )  # Called for t_curr and t_next
        self.assertTrue(
            self.mock_diffusion_process.score_fn.call_count >= self.n_steps
        )

    def test_ei_sampler_conditional(self):
        """Test exponential integrator runs with class conditioning."""
        num_classes = 5
        class_labels = torch.randint(
            0, num_classes, (self.batch_size,), device=self.device
        )
        self.mock_score_model.use_class_condition = True

        times, x_t_traj = exponential_integrator_sampler(
            diffusion_process=self.mock_diffusion_process,
            score_model=self.mock_score_model,
            x_T=self.x_T,
            t_0=self.t_0,
            t_end=self.t_end,
            n_steps=self.n_steps,
            class_labels=class_labels,  # Pass labels
            use_notebook_tqdm=False,
        )

        expected_time_shape = (self.n_steps + 1,)
        expected_traj_shape = (
            self.batch_size,
            self.channels,
            self.height,
            self.width,
            self.n_steps + 1,
        )
        self.assertEqual(times.shape, expected_time_shape)
        self.assertEqual(x_t_traj.shape, expected_traj_shape)

        # Check if the process score_fn was called with labels
        found_call_with_labels = False
        for args, kwargs in self.mock_diffusion_process.score_fn.call_args_list:
            # Check kwargs first
            if "class_labels" in kwargs and torch.equal(
                kwargs["class_labels"], class_labels
            ):
                found_call_with_labels = True
                break
            # Check positional args (model=0, x=1, t=2, labels=3)
            elif len(args) > 3 and torch.equal(args[3], class_labels):
                found_call_with_labels = True
                break
        self.assertTrue(
            found_call_with_labels,
            "process score_fn was not called with correct class_labels",
        )

    def test_ei_sampler_raises_type_error(self):
        """Test that EI sampler raises TypeError if process doesn't have VP methods."""
        mock_ve_process = MagicMock()
        mock_ve_process.kind = "VE"
        # Missing beta, alpha_bar methods

        with self.assertRaises(TypeError):
            exponential_integrator_sampler(
                diffusion_process=mock_ve_process,
                score_model=self.mock_score_model,
                x_T=self.x_T,
                t_0=self.t_0,
                t_end=self.t_end,
                n_steps=self.n_steps,
                use_notebook_tqdm=False,
            )


if __name__ == "__main__":
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestExponentialIntegratorSampler))
    runner = unittest.TextTestRunner()
    runner.run(suite)
