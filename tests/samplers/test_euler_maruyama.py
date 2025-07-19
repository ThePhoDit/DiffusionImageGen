import unittest
import torch

from diffusion.samplers.euler_maruyama import euler_maruyama_integrator


class TestEulerMaruyama(unittest.TestCase):

    def setUp(self):
        """Set up common parameters and tensors for tests."""
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.batch_size = 2
        self.channels = 1
        self.height = 4
        self.width = 4
        self.x_0 = torch.ones(
            self.batch_size,
            self.channels,
            self.height,
            self.width,
            device=self.device,
        )
        self.t_0 = 0.0
        self.t_end = 1.0
        self.n_steps = 10

    def test_integrator_shape(self):
        """Test the output shape of the Euler-Maruyama integrator."""

        # Simple drift/diffusion functions for testing execution
        def drift_fn(x, t):
            return -0.1 * x

        def diffusion_fn(t):
            # Ensure output shape is (batch_size,) even if t is scalar in call
            t_size = t.shape[0] if t.dim() > 0 else 1
            return torch.ones(t_size, device=t.device) * 0.5

        times, x_t_traj = euler_maruyama_integrator(
            x_0=self.x_0,
            t_0=self.t_0,
            t_end=self.t_end,
            n_steps=self.n_steps,
            drift_coefficient=drift_fn,
            diffusion_coefficient=diffusion_fn,
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

    def test_integrator_values_simple(self):
        """Test integrator values for a very simple case (zero drift/diffusion)."""

        def zero_drift(x, t):
            return torch.zeros_like(x)

        def zero_diffusion(t):
            t_size = t.shape[0] if t.dim() > 0 else 1
            return torch.zeros(t_size, device=t.device)

        times, x_t_traj = euler_maruyama_integrator(
            x_0=self.x_0,
            t_0=self.t_0,
            t_end=self.t_end,
            n_steps=self.n_steps,
            drift_coefficient=zero_drift,
            diffusion_coefficient=zero_diffusion,
        )

        # With zero drift and diffusion, trajectory should remain constant at x_0
        for i in range(self.n_steps + 1):
            torch.testing.assert_close(x_t_traj[..., i], self.x_0)

    def test_integrator_time_points(self):
        """Check if the generated time points are correct."""
        times, _ = euler_maruyama_integrator(
            x_0=self.x_0,
            t_0=self.t_0,
            t_end=self.t_end,
            n_steps=self.n_steps,
            drift_coefficient=lambda x, t: x,  # Dummy functions
            diffusion_coefficient=lambda t: torch.ones(
                t.shape[0] if t.dim() > 0 else 1, device=t.device
            ),
        )
        expected_times = torch.linspace(
            self.t_0, self.t_end, self.n_steps + 1, device=self.device
        )
        torch.testing.assert_close(times, expected_times)


if __name__ == "__main__":
    import os

    current_dir = os.path.dirname(os.path.abspath(__file__))
    tests_dir = os.path.dirname(current_dir)
    samplers_test_dir = os.path.join(tests_dir, "samplers")

    if not os.path.exists(tests_dir):
        os.makedirs(tests_dir)
    if not os.path.exists(os.path.join(tests_dir, "__init__.py")):
        open(os.path.join(tests_dir, "__init__.py"), "a").close()

    if not os.path.exists(samplers_test_dir):
        os.makedirs(samplers_test_dir)
    if not os.path.exists(os.path.join(samplers_test_dir, "__init__.py")):
        open(os.path.join(samplers_test_dir, "__init__.py"), "a").close()

    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestEulerMaruyama))
    runner = unittest.TextTestRunner()
    runner.run(suite)
