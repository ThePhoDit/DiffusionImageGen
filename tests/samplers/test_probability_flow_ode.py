import unittest
import torch
from unittest.mock import MagicMock
from unittest.mock import patch

from diffusion.samplers.probability_flow_ode import probability_flow_ode_sampler


class ConcreteMockDiffusionProcess:
    """A concrete mock for diffusion process that always returns tensors."""

    def __init__(self, device, batch_size=2, channels=1, height=8, width=8):
        self.device = device
        self.batch_size = batch_size
        self.channels = channels
        self.height = height
        self.width = width
        self.sample_shape = (batch_size, channels, height, width)
        self.call_counts = {
            "sde_drift_forward": 0,
            "diffusion_squared": 0,
            "score_fn": 0,
        }

        # Pre-compute tensor responses
        self.drift_tensor = torch.zeros(self.sample_shape, device=device)
        self.g2_tensor = torch.ones(batch_size, device=device) * 0.1
        self.score_tensor = torch.randn(self.sample_shape, device=device) * 0.1

    def sde_drift_forward(self):
        """Return a concrete callable that returns a tensor."""
        self.call_counts["sde_drift_forward"] += 1

        def drift_fn(x, t):
            # Return pre-computed drift tensor
            return self.drift_tensor

        return drift_fn

    def diffusion_squared(self):
        """Return a concrete callable that returns a tensor."""
        self.call_counts["diffusion_squared"] += 1

        def g2_fn(t):
            # Return pre-computed diffusion squared tensor
            return self.g2_tensor

        return g2_fn

    def score_fn(self, model, x, t, class_labels=None):
        """Return a score tensor directly."""
        self.call_counts["score_fn"] += 1
        self.last_score_call_args = (model, x, t, class_labels)
        # Return pre-computed score tensor
        return self.score_tensor


class TestProbabilityFlowODESampler(unittest.TestCase):

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
        self.n_steps = 6  # Use even number for RK4 steps check

        # Mock ScoreNet model that returns a tensor
        self.mock_score_model = MagicMock()
        self.mock_score_model.return_value = torch.randn_like(self.x_T) * 0.1
        self.mock_score_model.module = self.mock_score_model
        self.mock_score_model.use_class_condition = False

        # Concrete mock diffusion process that always returns tensors
        self.mock_diffusion_process = ConcreteMockDiffusionProcess(
            device=self.device,
            batch_size=self.batch_size,
            channels=self.channels,
            height=self.height,
            width=self.width,
        )

    def _run_sampler(self, use_rk4, early_stop_time=None):
        return probability_flow_ode_sampler(
            diffusion_process=self.mock_diffusion_process,
            score_model=self.mock_score_model,
            x_T=self.x_T,
            t_0=self.t_0,
            t_end=self.t_end,
            n_steps=self.n_steps,
            class_labels=None,
            early_stop_time=early_stop_time,
            use_rk4=use_rk4,
            use_notebook_tqdm=False,
        )

    def test_ode_sampler_euler_execution_shape(self):
        """Test ODE sampler (Euler) runs and returns correct shapes."""
        times, x_t_traj = self._run_sampler(use_rk4=False)

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
        # Check mocks called
        self.assertGreaterEqual(
            self.mock_diffusion_process.call_counts["sde_drift_forward"], 1
        )
        # diffusion_squared gets called once per step in Euler
        self.assertEqual(
            self.mock_diffusion_process.call_counts["diffusion_squared"],
            self.n_steps,
        )
        # Score fn called n_steps times for Euler
        self.assertEqual(
            self.mock_diffusion_process.call_counts["score_fn"], self.n_steps
        )

    def test_ode_sampler_rk4_execution_shape(self):
        """Test ODE sampler (RK4) runs and returns correct shapes."""
        times, x_t_traj = self._run_sampler(use_rk4=True)

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
        # Check mocks called
        self.assertGreaterEqual(
            self.mock_diffusion_process.call_counts["sde_drift_forward"], 1
        )
        # diffusion_squared gets called 4 times per step in RK4
        self.assertEqual(
            self.mock_diffusion_process.call_counts["diffusion_squared"],
            4 * self.n_steps,
        )
        # Score fn called 4 * n_steps times for RK4
        self.assertEqual(
            self.mock_diffusion_process.call_counts["score_fn"],
            4 * self.n_steps,
        )

    def test_ode_sampler_early_stop(self):
        """Test ODE sampler early stopping."""
        early_stop_time = self.t_0 / 2.0  # Stop halfway
        times, x_t_traj = self._run_sampler(
            use_rk4=True, early_stop_time=early_stop_time
        )

        # Check that the last time step is >= early_stop_time
        self.assertTrue(times[-1] >= early_stop_time)
        # Check that the number of steps is less than n_steps + 1
        self.assertTrue(times.shape[0] < self.n_steps + 1)
        self.assertEqual(times.shape[0], x_t_traj.shape[-1])

    def test_ode_sampler_conditional(self):
        """Test ODE sampler runs with class conditioning."""
        num_classes = 10
        class_labels = torch.randint(
            0, num_classes, (self.batch_size,), device=self.device
        )
        self.mock_score_model.use_class_condition = True

        # Reset call counts for this test
        self.mock_diffusion_process.call_counts = {
            "sde_drift_forward": 0,
            "diffusion_squared": 0,
            "score_fn": 0,
        }

        times, x_t_traj = probability_flow_ode_sampler(
            diffusion_process=self.mock_diffusion_process,
            score_model=self.mock_score_model,
            x_T=self.x_T,
            t_0=self.t_0,
            t_end=self.t_end,
            n_steps=self.n_steps,
            class_labels=class_labels,  # Pass labels
            early_stop_time=None,
            use_rk4=True,
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
        # The concrete mock stores the last call args
        _, _, _, last_labels = self.mock_diffusion_process.last_score_call_args
        self.assertIsNotNone(last_labels)
        self.assertTrue(
            torch.equal(last_labels, class_labels),
            "score_fn was not called with correct class_labels",
        )


if __name__ == "__main__":
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestProbabilityFlowODESampler))
    runner = unittest.TextTestRunner()
    runner.run(suite)
