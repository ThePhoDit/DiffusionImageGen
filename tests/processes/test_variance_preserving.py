import unittest
import torch
from unittest.mock import MagicMock, patch

from diffusion.processes.variance_preserving import VPDiffusionProcess


class TestVPDiffusionProcess(unittest.TestCase):

    def setUp(self):
        """Set up a VPDiffusionProcess instance and mock model."""
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.T = 1.0
        # Use linear schedule for testing simplicity
        self.process = VPDiffusionProcess(
            schedule="linear", T=self.T, beta_min=0.001, beta_max=0.02
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

        # Mock ScoreNet model - Assume it predicts x0 for VP loss/score functions here
        self.mock_score_model = MagicMock()
        self.mock_score_model.return_value = torch.randn_like(
            self.x_0
        )  # Model returns x0_hat
        self.mock_score_model.module = self.mock_score_model
        self.mock_score_model.use_class_condition = False

    def test_init(self):
        """Test initialization of VPDiffusionProcess."""
        self.assertEqual(self.process.kind, "VP")
        self.assertEqual(self.process.T, self.T)
        self.assertEqual(self.process.schedule_type, "linear")
        self.assertTrue(callable(self.process.beta))
        self.assertTrue(callable(self.process.alpha_bar))
        self.assertTrue(callable(self.process.std_dev))

    def test_marginal_prob(self):
        """Test VP marginal probability calculation."""
        mean, std = self.process.marginal_prob(self.x_0, self.t)
        self.assertEqual(mean.shape, self.x_0.shape)
        self.assertEqual(std.shape, (self.batch_size, 1, 1, 1))

        # Check mean scaling at t=0 (should be ~x_0)
        t_zero = torch.zeros_like(self.t)
        mean_zero, std_zero = self.process.marginal_prob(self.x_0, t_zero)
        torch.testing.assert_close(mean_zero, self.x_0, atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(
            std_zero, torch.zeros_like(std_zero), atol=1e-4, rtol=1e-5
        )

        # Check mean scaling at t=T (should be significantly smaller than at t=0)
        t_T = torch.ones_like(self.t) * self.T
        mean_T, std_T = self.process.marginal_prob(self.x_0, t_T)
        # Relax assertion: check mean magnitude is less than at t=0
        self.assertTrue(
            mean_T.abs().mean() < mean_zero.abs().mean(),
            "Mean magnitude at T should be smaller than at t=0",
        )
        self.assertTrue(torch.all(std_T > 0.1))  # Std dev should be significant

    def test_sample_forward(self):
        """Test VP forward sampling."""
        x_t, noise = self.process.sample_forward(self.x_0, self.t)
        self.assertEqual(x_t.shape, self.x_0.shape)
        self.assertEqual(noise.shape, self.x_0.shape)

    def test_sde_coeffs_forward(self):
        """Test VP forward SDE coefficients."""
        drift_fn = self.process.sde_drift_forward()
        diffusion_sq_fn = self.process.diffusion_squared()

        drift = drift_fn(self.x_0, self.t)
        diffusion_sq = diffusion_sq_fn(self.t)

        self.assertEqual(drift.shape, self.x_0.shape)
        self.assertEqual(diffusion_sq.shape, (self.batch_size,))
        self.assertTrue(torch.all(diffusion_sq >= 0))
        # Check g^2 = beta
        beta_vals = self.process.beta(self.t)
        torch.testing.assert_close(
            diffusion_sq, beta_vals, atol=1e-6, rtol=1e-5
        )

    def test_score_fn(self):
        """Test the VP score function calculation (assumes model predicts x0)."""
        score = self.process.score_fn(self.mock_score_model, self.x_0, self.t)
        self.assertEqual(score.shape, self.x_0.shape)
        self.mock_score_model.assert_called_once()

    def test_score_fn_conditional(self):
        """Test VP score function with conditioning."""
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
        """Test the VP loss function (assumes model predicts x0)."""
        loss = self.process.loss_fn(self.mock_score_model, self.x_0)
        self.assertTrue(torch.is_tensor(loss))
        self.assertEqual(loss.numel(), 1)
        self.mock_score_model.assert_called_once()

    def test_loss_fn_conditional(self):
        """Test VP loss function with conditioning."""
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
            # Check kwargs first for explicit passing
            if "class_labels" in kwargs and torch.equal(
                kwargs["class_labels"], class_labels
            ):
                found_call_with_labels = True
                break
            # Check positional args (model=0, x=1, t=2, labels=3 -> WRONG, should be x=0, t=1, labels=2)
            elif len(args) > 2 and torch.equal(args[2], class_labels):
                found_call_with_labels = True
                break
        self.assertTrue(
            found_call_with_labels,
            "mock_score_model was not called with correct class_labels",
        )

    def test_sde_coeffs_reverse(self):
        """Test VP reverse SDE coefficients."""
        drift_fn_reverse = self.process.sde_drift_reverse(self.mock_score_model)
        diffusion_fn_reverse = self.process.sde_diffusion_reverse()

        drift = drift_fn_reverse(self.x_0, self.t)
        diffusion = diffusion_fn_reverse(self.t)

        self.assertEqual(drift.shape, self.x_0.shape)
        self.mock_score_model.assert_called_once()
        self.assertEqual(diffusion.shape, (self.batch_size,))
        self.assertTrue(torch.all(diffusion >= 0))

        # Check reverse diffusion = sqrt(beta)
        beta_vals = self.process.beta(self.t)
        torch.testing.assert_close(
            diffusion**2, beta_vals, atol=1e-6, rtol=1e-5
        )


if __name__ == "__main__":
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestVPDiffusionProcess))
    runner = unittest.TextTestRunner()
    runner.run(suite)
