import unittest
import torch
from unittest.mock import MagicMock

from diffusion.samplers.predictor_corrector import pc_sampler


class TestPCSampler(unittest.TestCase):

    def setUp(self):
        """Set up mock diffusion process, score model, and tensors."""
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
        self.n_steps = 5  # Keep low for testing

        # Mock ScoreNet model
        self.mock_score_model = MagicMock()
        self.mock_score_model.return_value = torch.randn_like(self.x_T)
        self.mock_score_model.module = self.mock_score_model
        self.mock_score_model.use_class_condition = False

        # Mock DiffusionProcess
        self.mock_diffusion_process = MagicMock()
        # Define behavior for methods needed by pc_sampler

        # 1. sde_drift_reverse:
        # Needs to return a FUNCTION that calls the process's score_fn internally
        def mock_reverse_drift_calculator(x, t, class_labels=None):
            # Call the process's mocked score_fn (which calls the base model mock)
            score = self.mock_diffusion_process.score_fn(
                self.mock_score_model, x, t, class_labels=class_labels
            )
            # Calculate a dummy drift based on score (e.g., -score, simplified)
            # The exact formula isn't critical, just that score_fn is called
            dummy_drift = -score * 0.1
            return dummy_drift

        self.mock_diffusion_process.sde_drift_reverse.return_value = (
            mock_reverse_drift_calculator
        )

        # 2. sde_diffusion_reverse:
        mock_reverse_diffusion_fn = MagicMock(
            return_value=torch.ones(self.batch_size, device=self.device)
        )
        self.mock_diffusion_process.sde_diffusion_reverse.return_value = (
            mock_reverse_diffusion_fn
        )

        # 3. score_fn:
        # Ensure the mock records calls correctly AND CALLS the underlying model mock
        def mock_score_logic(*args, **kwargs):
            # The process's score_fn is called like: process.score_fn(model, x, t, labels)
            # We need to call the underlying model mock with its expected args: model(x, t, labels)
            model_arg = args[0]
            x_arg = args[1]
            t_arg = args[2]
            labels_arg = kwargs.get("class_labels", None)
            if labels_arg is None and len(args) > 3:
                labels_arg = args[3]  # Get from positional if passed that way

            # Actually call the base mock model
            return self.mock_score_model(x_arg, t_arg, class_labels=labels_arg)

        self.mock_diffusion_process.score_fn = MagicMock(
            side_effect=mock_score_logic
        )

        # 4. marginal_prob (only need std_dev for corrector step size):
        mock_mean = torch.zeros_like(self.x_T)
        mock_std = (
            torch.ones(self.batch_size, 1, 1, 1, device=self.device) * 0.5
        )
        self.mock_diffusion_process.marginal_prob.return_value = (
            mock_mean,
            mock_std,
        )

    def test_pc_sampler_execution_shape(self):
        """Test that pc_sampler runs and returns correct shapes."""
        times, x_t_traj = pc_sampler(
            diffusion_process=self.mock_diffusion_process,
            score_model=self.mock_score_model,
            x_T=self.x_T,
            t_0=self.t_0,
            t_end=self.t_end,
            n_steps=self.n_steps,
            snr=0.1,
            num_corrector_steps=1,
            use_notebook_tqdm=False,  # Avoid tqdm import issues in tests
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
        self.mock_diffusion_process.sde_drift_reverse.assert_called_with(
            self.mock_score_model
        )
        self.mock_diffusion_process.sde_diffusion_reverse.assert_called()
        self.assertTrue(
            self.mock_diffusion_process.score_fn.call_count >= self.n_steps
        )
        self.assertTrue(
            self.mock_diffusion_process.marginal_prob.call_count >= self.n_steps
        )

    def test_pc_sampler_conditional(self):
        """Test pc_sampler runs with class conditioning."""
        num_classes = 10
        class_labels = torch.randint(
            0, num_classes, (self.batch_size,), device=self.device
        )

        # Set up mocks to expect class labels
        self.mock_score_model.use_class_condition = True

        times, x_t_traj = pc_sampler(
            diffusion_process=self.mock_diffusion_process,
            score_model=self.mock_score_model,
            x_T=self.x_T,
            t_0=self.t_0,
            t_end=self.t_end,
            n_steps=self.n_steps,
            snr=0.1,
            num_corrector_steps=1,
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

        # Check if the underlying score model was called with labels (via score_fn or drift_fn) - KEEP THIS CHECK
        found_call_with_labels = False
        for args, kwargs in self.mock_score_model.call_args_list:
            if "class_labels" in kwargs and torch.equal(
                kwargs["class_labels"], class_labels
            ):
                found_call_with_labels = True
                break
            elif (
                len(args) > 2
                and isinstance(args[2], torch.Tensor)
                and torch.equal(args[2], class_labels)
            ):
                found_call_with_labels = True
                break
        self.assertTrue(
            found_call_with_labels,
            "mock_score_model was not called with correct class_labels",
        )


if __name__ == "__main__":
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestPCSampler))
    runner = unittest.TextTestRunner()
    runner.run(suite)
