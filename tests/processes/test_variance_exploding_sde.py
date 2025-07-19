import unittest
import torch
from unittest.mock import MagicMock, patch

from diffusion.processes.variance_exploding_sde import VEDiffusionProcess


class TestVEDiffusionProcess(unittest.TestCase):

    def setUp(self):
        """Set up a VEDiffusionProcess instance and mock model."""
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.sigma_param = 25.0
        self.T = 1.0
        self.process = VEDiffusionProcess(
            sigma_param=self.sigma_param, T=self.T
        )

        self.batch_size = 4
        self.channels = 1
        self.height = (
            8  # Needs to be divisible for potential downsampling in mock model
        )
        self.width = 8
        self.x_0 = torch.randn(
            self.batch_size,
            self.channels,
            self.height,
            self.width,
            device=self.device,
        )
        self.t = torch.rand(self.batch_size, device=self.device) * self.T

        # Mock ScoreNet model
        self.mock_score_model = MagicMock()
        # Simulate the model returning a tensor of the correct shape
        self.mock_score_model.return_value = torch.randn_like(self.x_0)
        # Simulate DataParallel structure if needed (can be simplified)
        self.mock_score_model.module = self.mock_score_model
        # Simulate conditioning attribute
        self.mock_score_model.use_class_condition = False

    def test_init(self):
        """Test initialization of VEDiffusionProcess."""
        self.assertEqual(self.process.kind, "VE")
        self.assertEqual(self.process.T, self.T)
        self.assertEqual(self.process.sigma_param, self.sigma_param)
        self.assertTrue(callable(self.process.sigma_t))
        self.assertTrue(callable(self.process.g_t))
        self.assertTrue(callable(self.process.g2_t))
        self.assertTrue(callable(self.process.mu_t))

    def test_marginal_prob(self):
        """Test marginal probability calculation."""
        mean, std = self.process.marginal_prob(self.x_0, self.t)
        self.assertEqual(mean.shape, self.x_0.shape)
        self.assertEqual(std.shape, (self.batch_size, 1, 1, 1))
        # For VE SDE, mean should be x_0
        torch.testing.assert_close(mean, self.x_0)

        # Test std dev at t=0 (should be near 0)
        t_zero = torch.zeros_like(self.t)
        _, std_zero = self.process.marginal_prob(self.x_0, t_zero)
        torch.testing.assert_close(
            std_zero, torch.zeros_like(std_zero), atol=1e-4, rtol=1e-5
        )

    def test_sample_forward(self):
        """Test forward sampling."""
        x_t, noise = self.process.sample_forward(self.x_0, self.t)
        self.assertEqual(x_t.shape, self.x_0.shape)
        self.assertEqual(noise.shape, self.x_0.shape)

    def test_sde_coeffs_forward(self):
        """Test forward SDE coefficients."""
        drift_fn = self.process.sde_drift_forward()
        diffusion_fn = self.process.sde_diffusion()

        drift = drift_fn(self.x_0, self.t)
        diffusion = diffusion_fn(self.t)

        self.assertEqual(drift.shape, self.x_0.shape)
        # VE drift is zero
        torch.testing.assert_close(drift, torch.zeros_like(drift))
        self.assertEqual(diffusion.shape, (self.batch_size,))
        self.assertTrue(torch.all(diffusion >= 0))

    def test_score_fn(self):
        """Test the score function calculation using a mock model."""
        score = self.process.score_fn(self.mock_score_model, self.x_0, self.t)
        self.assertEqual(score.shape, self.x_0.shape)
        self.mock_score_model.assert_called_once()

    def test_score_fn_conditional(self):
        """Test score function with conditioning enabled."""
        self.mock_score_model.use_class_condition = True
        num_classes = 10
        class_labels = torch.randint(
            0, num_classes, (self.batch_size,), device=self.device
        )
        score = self.process.score_fn(
            self.mock_score_model, self.x_0, self.t, class_labels
        )
        self.assertEqual(score.shape, self.x_0.shape)
        # Check that the underlying mock model was called with labels
        found_call_with_labels = False
        # Check the last call primarily, but allow any call to match
        for args, kwargs in self.mock_score_model.call_args_list:
            if "class_labels" in kwargs and torch.equal(
                kwargs["class_labels"], class_labels
            ):
                found_call_with_labels = True
                break
            # Check positional args (x=0, t=1, labels=2)
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

    def test_loss_fn(self):
        """Test the loss function calculation using a mock model."""
        loss = self.process.loss_fn(self.mock_score_model, self.x_0)
        self.assertTrue(torch.is_tensor(loss))
        self.assertEqual(loss.numel(), 1)
        self.mock_score_model.assert_called_once()

    def test_loss_fn_conditional(self):
        """Test loss function with conditioning enabled."""
        self.mock_score_model.use_class_condition = True
        num_classes = 10
        class_labels = torch.randint(
            0, num_classes, (self.batch_size,), device=self.device
        )
        loss = self.process.loss_fn(
            self.mock_score_model, self.x_0, class_labels
        )
        self.assertTrue(torch.is_tensor(loss))
        self.assertEqual(loss.numel(), 1)
        # Check that the underlying mock model was called with labels
        found_call_with_labels = False
        for args, kwargs in self.mock_score_model.call_args_list:
            if "class_labels" in kwargs and torch.equal(
                kwargs["class_labels"], class_labels
            ):
                found_call_with_labels = True
                break
            # Check positional args (assuming x, t, labels)
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

    def test_sde_coeffs_reverse(self):
        """Test reverse SDE coefficients."""
        drift_fn_reverse = self.process.sde_drift_reverse(self.mock_score_model)
        diffusion_fn_reverse = self.process.sde_diffusion_reverse()

        drift = drift_fn_reverse(self.x_0, self.t)
        diffusion = diffusion_fn_reverse(self.t)

        self.assertEqual(drift.shape, self.x_0.shape)
        self.mock_score_model.assert_called_once()
        self.assertEqual(diffusion.shape, (self.batch_size,))
        self.assertTrue(torch.all(diffusion >= 0))

        # Check reverse diffusion matches forward diffusion func
        forward_diffusion_fn = self.process.sde_diffusion()
        torch.testing.assert_close(diffusion, forward_diffusion_fn(self.t))


if __name__ == "__main__":
    import os

    current_dir = os.path.dirname(os.path.abspath(__file__))
    tests_dir = os.path.dirname(current_dir)
    processes_test_dir = os.path.join(tests_dir, "processes")

    if not os.path.exists(tests_dir):
        os.makedirs(tests_dir)
    if not os.path.exists(os.path.join(tests_dir, "__init__.py")):
        open(os.path.join(tests_dir, "__init__.py"), "a").close()

    if not os.path.exists(processes_test_dir):
        os.makedirs(processes_test_dir)
    if not os.path.exists(os.path.join(processes_test_dir, "__init__.py")):
        open(os.path.join(processes_test_dir, "__init__.py"), "a").close()

    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestVEDiffusionProcess))
    runner = unittest.TextTestRunner()
    runner.run(suite)
