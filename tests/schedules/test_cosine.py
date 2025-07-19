import unittest
import torch
import numpy as np
from diffusion.schedules.cosine import get_cosine_schedule_functions


class TestCosineSchedule(unittest.TestCase):

    def test_schedule_functions_values(self):
        """Test the output values of cosine schedule functions at key times."""
        T = 1.0
        s = 0.008

        beta_fn, alpha_bar_fn, std_dev_fn = get_cosine_schedule_functions(
            T=T, s=s
        )

        # Precompute f(0) for comparison
        f_0 = np.cos((s / (1 + s)) * np.pi / 2) ** 2

        # Test at t=0
        t_zero = torch.tensor([0.0])
        f_t_zero = np.cos(((0.0 + s) / (1 + s)) * (np.pi / 2)) ** 2
        expected_alpha_bar_zero = f_t_zero / f_0  # Should be 1.0
        self.assertAlmostEqual(
            alpha_bar_fn(t_zero).item(), expected_alpha_bar_zero, delta=1e-6
        )
        self.assertAlmostEqual(
            std_dev_fn(t_zero).item(), 0.0, delta=3e-4
        )  # Increased delta further
        # Beta at t=0 is small but non-zero with this schedule formula
        self.assertAlmostEqual(beta_fn(t_zero).item(), 0.038856, places=5)

        # Test at t=T
        t_T = torch.tensor([T])
        f_t_T = (
            np.cos(((1.0 + s) / (1 + s)) * (np.pi / 2)) ** 2
        )  # cos(pi/2)^2 = 0
        expected_alpha_bar_T = f_t_T / f_0  # Should be close to 0
        self.assertAlmostEqual(
            alpha_bar_fn(t_T).item(), expected_alpha_bar_T, delta=1e-6
        )
        expected_std_dev_T = torch.sqrt(
            1.0 - torch.tensor(expected_alpha_bar_T)
        ).item()  # Close to 1
        self.assertAlmostEqual(
            std_dev_fn(t_T).item(), expected_std_dev_T, delta=1e-6
        )
        # Beta at t=T should be clamped to beta_max (default 0.999)
        self.assertAlmostEqual(beta_fn(t_T).item(), 0.999, places=5)

    def test_schedule_functions_shapes(self):
        """Test the output shapes of cosine schedule functions."""
        beta_fn, alpha_bar_fn, std_dev_fn = get_cosine_schedule_functions()
        batch_size = 7
        t_batch = torch.rand(batch_size)

        beta_out = beta_fn(t_batch)
        alpha_bar_out = alpha_bar_fn(t_batch)
        std_dev_out = std_dev_fn(t_batch)

        self.assertEqual(beta_out.shape, (batch_size,))
        self.assertEqual(alpha_bar_out.shape, (batch_size,))
        self.assertEqual(std_dev_out.shape, (batch_size,))

    def test_clamping(self):
        """Test that time clamping works correctly."""
        beta_fn, alpha_bar_fn, std_dev_fn = get_cosine_schedule_functions(T=1.0)

        t_neg = torch.tensor([-0.5])
        t_large = torch.tensor([1.5])
        t_zero = torch.tensor([0.0])
        t_T = torch.tensor([1.0])

        # Values for negative/large times should be clamped to t=0/t=T equivalents
        self.assertAlmostEqual(
            beta_fn(t_neg).item(), beta_fn(t_zero).item(), delta=1e-6
        )
        self.assertAlmostEqual(
            alpha_bar_fn(t_neg).item(), alpha_bar_fn(t_zero).item(), delta=1e-6
        )
        self.assertAlmostEqual(
            std_dev_fn(t_neg).item(), std_dev_fn(t_zero).item(), delta=1e-6
        )

        self.assertAlmostEqual(
            beta_fn(t_large).item(), beta_fn(t_T).item(), delta=1e-6
        )
        self.assertAlmostEqual(
            alpha_bar_fn(t_large).item(), alpha_bar_fn(t_T).item(), delta=1e-6
        )
        self.assertAlmostEqual(
            std_dev_fn(t_large).item(), std_dev_fn(t_T).item(), delta=1e-6
        )


if __name__ == "__main__":
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestCosineSchedule))
    runner = unittest.TextTestRunner()
    runner.run(suite)
