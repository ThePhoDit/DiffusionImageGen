import unittest
import torch
from unittest.mock import MagicMock

from diffusion.processes.subvariance_preserving import (
    SubVariancePreservingDiffusionProcess,
)


class TestSubVPDiffusionProcess(unittest.TestCase):

    def setUp(self):
        """Set up a SubVPDiffusionProcess instance and mock model."""
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.T = 1.0
        # Use linear schedule for testing simplicity
        self.process = SubVariancePreservingDiffusionProcess(
            schedule="linear",
            T=self.T,
            beta_min=0.001,
            beta_max=0.02,
            eps_var=1e-7,
        )

        self.batch_size = 4
        self.channels = 3
        self.height = 16
        self.width = 16
        self.x_0 = torch.randn(
            self.batch_size,
            self.channels,
            self.height,
            self.width,
            device=self.device,
        )
        self.t = torch.rand(self.batch_size, device=self.device) * self.T

        # Mock ScoreNet model - Assume it predicts x0 for SubVP loss/score functions
        self.mock_score_model = MagicMock()
        self.mock_score_model.return_value = torch.randn_like(
            self.x_0
        )  # Model returns x0_hat
        self.mock_score_model.module = self.mock_score_model
        self.mock_score_model.use_class_condition = False

    def test_init(self):
        """Test initialization of SubVPDiffusionProcess."""
        self.assertEqual(self.process.kind, "Sub-VP")
        self.assertEqual(self.process.T, self.T)
        self.assertEqual(self.process.schedule_type, "linear")
        self.assertTrue(callable(self.process.beta))
        self.assertTrue(callable(self.process.alpha_bar))
        self.assertTrue(
            callable(self.process.std_dev)
        )  # Specific std_dev for SubVP

    def test_marginal_prob(self):
        """Test SubVP marginal probability calculation."""
        mean, std = self.process.marginal_prob(self.x_0, self.t)
        self.assertEqual(mean.shape, self.x_0.shape)
        self.assertEqual(std.shape, (self.batch_size, 1, 1, 1))

        # Check mean scaling at t=0 (should be ~x_0)
        t_zero = torch.zeros_like(self.t)
        mean_zero, std_zero = self.process.marginal_prob(self.x_0, t_zero)
        torch.testing.assert_close(mean_zero, self.x_0, atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(
            std_zero, torch.zeros_like(std_zero), atol=5e-4, rtol=1e-5
        )

        # Check std dev formula: sigma(t)^2 = 1 - alpha_bar(t)^2
        alpha_bar_vals = self.process.alpha_bar(self.t).view(-1, 1, 1, 1)
        expected_std_sq = 1.0 - alpha_bar_vals**2
        # Need to handle potential clamping in self.std_dev
        std_clamped = torch.sqrt(
            torch.clamp(expected_std_sq, min=self.process.eps_var)
        )
        torch.testing.assert_close(std, std_clamped, atol=1e-6, rtol=1e-5)

    def test_sample_forward(self):
        """Test SubVP forward sampling."""
        x_t, noise = self.process.sample_forward(self.x_0, self.t)
        self.assertEqual(x_t.shape, self.x_0.shape)
        self.assertEqual(noise.shape, self.x_0.shape)

    def test_sde_coeffs_forward(self):
        """Test SubVP forward SDE coefficients."""
        drift_fn = self.process.sde_drift_forward()
        diffusion_sq_fn = self.process.diffusion_squared()

        drift = drift_fn(self.x_0, self.t)
        diffusion_sq = diffusion_sq_fn(self.t)

        self.assertEqual(drift.shape, self.x_0.shape)
        self.assertEqual(diffusion_sq.shape, (self.batch_size,))
        self.assertTrue(torch.all(diffusion_sq >= 0))

        # Check g^2 = beta * (1 - alpha_bar^2)
        beta_vals = self.process.beta(self.t)
        alpha_bar_vals = self.process.alpha_bar(self.t)
        expected_diffusion_sq = beta_vals * (1.0 - alpha_bar_vals**2)
        expected_diffusion_sq = torch.clamp(
            expected_diffusion_sq, min=self.process.eps_var
        )
        torch.testing.assert_close(
            diffusion_sq, expected_diffusion_sq, atol=1e-6, rtol=1e-5
        )

    def test_score_fn(self):
        """Test the SubVP score function calculation (assumes model predicts x0)."""
        score = self.process.score_fn(self.mock_score_model, self.x_0, self.t)
        self.assertEqual(score.shape, self.x_0.shape)
        self.mock_score_model.assert_called_once()

    def test_score_fn_conditional(self):
        """Test SubVP score function with conditioning."""
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
        for args, kwargs in self.mock_score_model.call_args_list:
            if "class_labels" in kwargs and torch.equal(
                kwargs["class_labels"], class_labels
            ):
                found_call_with_labels = True
                break
            # Check positional args (x=0, t=1, labels=2)
            elif len(args) > 2 and torch.equal(args[2], class_labels):
                found_call_with_labels = True
                break
        self.assertTrue(
            found_call_with_labels,
            "mock_score_model was not called with correct class_labels",
        )

    def test_loss_fn(self):
        """Test the SubVP loss function (assumes model predicts x0, same as VP)."""
        loss = self.process.loss_fn(self.mock_score_model, self.x_0)
        self.assertTrue(torch.is_tensor(loss))
        self.assertEqual(loss.numel(), 1)
        self.mock_score_model.assert_called_once()

    def test_loss_fn_conditional(self):
        """Test SubVP loss function with conditioning."""
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
            # Check positional args (x=0, t=1, labels=2)
            elif len(args) > 2 and torch.equal(args[2], class_labels):
                found_call_with_labels = True
                break
        self.assertTrue(
            found_call_with_labels,
            "mock_score_model was not called with correct class_labels",
        )

    def test_sde_coeffs_reverse(self):
        """Test SubVP reverse SDE coefficients."""
        drift_fn_reverse = self.process.sde_drift_reverse(self.mock_score_model)
        diffusion_fn_reverse = self.process.sde_diffusion_reverse()

        drift = drift_fn_reverse(self.x_0, self.t)
        diffusion = diffusion_fn_reverse(self.t)

        self.assertEqual(drift.shape, self.x_0.shape)
        self.mock_score_model.assert_called_once()
        self.assertEqual(diffusion.shape, (self.batch_size,))
        self.assertTrue(torch.all(diffusion >= 0))

        # Check reverse diffusion = sqrt(g(t)^2)
        diffusion_sq = self.process.diffusion_squared()(self.t)
        torch.testing.assert_close(
            diffusion**2, diffusion_sq, atol=1e-6, rtol=1e-5
        )


if __name__ == "__main__":
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestSubVPDiffusionProcess))
    runner = unittest.TextTestRunner()
    runner.run(suite)
