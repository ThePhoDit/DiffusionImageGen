import unittest
import torch
from diffusion.schedules.linear import get_linear_schedule_functions


class TestLinearSchedule(unittest.TestCase):

    def test_schedule_functions_values(self):
        """Test the output values of schedule functions at key times."""
        beta_min = 0.0001
        beta_max = 0.02
        T = 1.0

        beta_fn, alpha_bar_fn, std_dev_fn = get_linear_schedule_functions(
            beta_min=beta_min, beta_max=beta_max, T=T
        )

        # Test at t=0
        t_zero = torch.tensor([0.0])
        self.assertAlmostEqual(beta_fn(t_zero).item(), beta_min, delta=1e-6)
        self.assertAlmostEqual(alpha_bar_fn(t_zero).item(), 1.0, delta=1e-6)
        self.assertAlmostEqual(std_dev_fn(t_zero).item(), 0.0, delta=2e-4)

        # Test at t=T
        t_T = torch.tensor([T])
        expected_integral_T = beta_min * T + 0.5 * (beta_max - beta_min) * T
        expected_alpha_bar_T = torch.exp(
            -torch.tensor(expected_integral_T)
        ).item()
        self.assertAlmostEqual(beta_fn(t_T).item(), beta_max, delta=1e-6)
        self.assertAlmostEqual(
            alpha_bar_fn(t_T).item(), expected_alpha_bar_T, delta=1e-6
        )
        expected_std_dev_T = torch.sqrt(
            1.0 - torch.tensor(expected_alpha_bar_T)
        ).item()
        self.assertAlmostEqual(
            std_dev_fn(t_T).item(), expected_std_dev_T, delta=1e-6
        )

        # Test at t=T/2
        t_half = torch.tensor([T / 2.0])
        expected_beta_half = beta_min + (beta_max - beta_min) * 0.5
        expected_integral_half = (
            beta_min * (T / 2.0)
            + 0.5 * (beta_max - beta_min) * ((T / 2.0) ** 2) / T
        )
        expected_alpha_bar_half = torch.exp(
            -torch.tensor(expected_integral_half)
        ).item()
        expected_std_dev_half = torch.sqrt(
            1.0 - torch.tensor(expected_alpha_bar_half)
        ).item()

        self.assertAlmostEqual(
            beta_fn(t_half).item(), expected_beta_half, delta=1e-6
        )
        self.assertAlmostEqual(
            alpha_bar_fn(t_half).item(), expected_alpha_bar_half, delta=1e-6
        )
        self.assertAlmostEqual(
            std_dev_fn(t_half).item(), expected_std_dev_half, delta=1e-6
        )

    def test_schedule_functions_shapes(self):
        """Test the output shapes of schedule functions."""
        beta_fn, alpha_bar_fn, std_dev_fn = get_linear_schedule_functions()
        batch_size = 5
        t_batch = torch.rand(batch_size)

        beta_out = beta_fn(t_batch)
        alpha_bar_out = alpha_bar_fn(t_batch)
        std_dev_out = std_dev_fn(t_batch)

        self.assertEqual(beta_out.shape, (batch_size,))
        self.assertEqual(alpha_bar_out.shape, (batch_size,))
        self.assertEqual(std_dev_out.shape, (batch_size,))

    def test_clamping(self):
        """Test that time clamping works correctly."""
        beta_fn, alpha_bar_fn, std_dev_fn = get_linear_schedule_functions(T=1.0)

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
    # Create dummy __init__.py if they don't exist
    import os

    # Ensure the test directory exists
    current_dir = os.path.dirname(os.path.abspath(__file__))
    tests_dir = os.path.dirname(
        current_dir
    )  # Assumes this file is in tests/schedules/
    schedules_test_dir = os.path.join(tests_dir, "schedules")

    if not os.path.exists(tests_dir):
        os.makedirs(tests_dir)
    if not os.path.exists(os.path.join(tests_dir, "__init__.py")):
        open(os.path.join(tests_dir, "__init__.py"), "a").close()

    if not os.path.exists(schedules_test_dir):
        os.makedirs(schedules_test_dir)
    if not os.path.exists(os.path.join(schedules_test_dir, "__init__.py")):
        open(os.path.join(schedules_test_dir, "__init__.py"), "a").close()

    # Run tests using unittest's discovery mechanism from the project root is better,
    # but this allows running the file directly.
    # For discovery from root: python -m unittest discover tests
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestLinearSchedule))
    runner = unittest.TextTestRunner()
    runner.run(suite)
