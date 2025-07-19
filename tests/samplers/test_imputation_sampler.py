import unittest
import torch
from unittest.mock import MagicMock

from diffusion.samplers.imputation_sampler import repaint_sampler


class TestImputationSampler(unittest.TestCase):

    def setUp(self):
        """Set up mock diffusion process, score model, and tensors for imputation."""
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.batch_size = 2
        self.channels = 1
        self.height = 8
        self.width = 8
        # Known data (e.g., original image normalized)
        self.x_masked = (
            torch.rand(
                self.batch_size,
                self.channels,
                self.height,
                self.width,
                device=self.device,
            )
            * 2
            - 1
        )  # Simulate range [-1, 1]
        # Mask (e.g., center square unknown)
        self.mask = torch.ones_like(self.x_masked)
        self.mask[:, :, 2:-2, 2:-2] = 0  # 1 = known, 0 = unknown

        self.t_0 = 1.0
        self.t_end = 1e-3
        self.n_steps = 6  # Keep low for testing
        self.jump_length = 2
        self.jump_n_sample = 1

        # Mock ScoreNet model
        self.mock_score_model = MagicMock()
        # Need a consistent return value shape for score/drift calculations
        self.mock_score_model.return_value = torch.randn_like(self.x_masked)
        self.mock_score_model.module = self.mock_score_model
        self.mock_score_model.use_class_condition = False

        # --- Mock VP Process (for testing VP path in repaint) ---
        self.mock_vp_process = MagicMock()
        self.mock_vp_process.kind = "VP"
        # 1. sde_drift_reverse
        mock_vp_reverse_drift_fn = MagicMock(
            return_value=torch.zeros_like(self.x_masked)
        )
        self.mock_vp_process.sde_drift_reverse.return_value = (
            mock_vp_reverse_drift_fn
        )
        # 2. sde_diffusion_reverse
        mock_vp_reverse_diffusion_fn = MagicMock(
            return_value=torch.ones(self.batch_size, device=self.device) * 0.1
        )
        self.mock_vp_process.sde_diffusion_reverse.return_value = (
            mock_vp_reverse_diffusion_fn
        )
        # 3. sample_forward (for known data noise)
        mock_vp_xt = (
            self.x_masked * 0.9 + torch.randn_like(self.x_masked) * 0.1
        )  # Simulate noisy known data
        self.mock_vp_process.sample_forward.return_value = (
            mock_vp_xt,
            torch.randn_like(self.x_masked),
        )

        # 4. alpha_bar (for forward noise step)
        def mock_vp_alpha_bar_fn(t):
            t_clamped = torch.clamp(t, 0.0, 1.0)
            return 1.0 - t_clamped * 0.99

        self.mock_vp_process.alpha_bar = MagicMock(
            side_effect=mock_vp_alpha_bar_fn
        )

        # --- Mock VE Process (for testing VE path in repaint) ---
        self.mock_ve_process = MagicMock()
        self.mock_ve_process.kind = "VE"
        # 1. sde_drift_reverse
        mock_ve_reverse_drift_fn = MagicMock(
            return_value=torch.zeros_like(self.x_masked)
        )
        self.mock_ve_process.sde_drift_reverse.return_value = (
            mock_ve_reverse_drift_fn
        )
        # 2. sde_diffusion_reverse
        mock_ve_reverse_diffusion_fn = MagicMock(
            return_value=torch.ones(self.batch_size, device=self.device) * 0.5
        )
        self.mock_ve_process.sde_diffusion_reverse.return_value = (
            mock_ve_reverse_diffusion_fn
        )
        # 3. sample_forward
        mock_ve_xt = (
            self.x_masked + torch.randn_like(self.x_masked) * 0.5
        )  # Simulate noisy known data
        self.mock_ve_process.sample_forward.return_value = (
            mock_ve_xt,
            torch.randn_like(self.x_masked),
        )

        # 4. sigma_t (for forward noise step)
        def mock_ve_sigma_t_fn(t):
            t_clamped = torch.clamp(t, 0.0, 1.0)
            # Simplified sigma_t for testing
            return torch.sqrt(
                t_clamped * 5.0 + 1e-7
            )  # Ensure variance increases with t

        self.mock_ve_process.sigma_t = MagicMock(side_effect=mock_ve_sigma_t_fn)

    def test_repaint_vp_execution_shape(self):
        """Test repaint_sampler with mock VP process runs and returns shapes."""
        times, x_t_traj = repaint_sampler(
            diffusion_process=self.mock_vp_process,
            score_model=self.mock_score_model,
            x_masked=self.x_masked,
            mask=self.mask,
            t_0=self.t_0,
            t_end=self.t_end,
            n_steps=self.n_steps,
            jump_length=self.jump_length,
            jump_n_sample=self.jump_n_sample,
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
        # Check required methods were called
        self.mock_vp_process.sde_drift_reverse.assert_called_with(
            self.mock_score_model
        )
        self.mock_vp_process.sde_diffusion_reverse.assert_called()
        # sample_forward and alpha_bar should be called during resampling steps
        num_jumps = (
            self.n_steps - 1
        ) // self.jump_length  # Jumps happen after step 0
        expected_forward_calls = num_jumps * self.jump_n_sample
        self.assertEqual(
            self.mock_vp_process.sample_forward.call_count,
            expected_forward_calls,
        )
        self.assertTrue(
            self.mock_vp_process.alpha_bar.call_count >= expected_forward_calls
        )

    def test_repaint_ve_execution_shape(self):
        """Test repaint_sampler with mock VE process runs and returns shapes."""
        times, x_t_traj = repaint_sampler(
            diffusion_process=self.mock_ve_process,
            score_model=self.mock_score_model,
            x_masked=self.x_masked,
            mask=self.mask,
            t_0=self.t_0,
            t_end=self.t_end,
            n_steps=self.n_steps,
            jump_length=self.jump_length,
            jump_n_sample=self.jump_n_sample,
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
        # Check required methods were called
        self.mock_ve_process.sde_drift_reverse.assert_called_with(
            self.mock_score_model
        )
        self.mock_ve_process.sde_diffusion_reverse.assert_called()
        num_jumps = (self.n_steps - 1) // self.jump_length
        expected_forward_calls = num_jumps * self.jump_n_sample
        self.assertEqual(
            self.mock_ve_process.sample_forward.call_count,
            expected_forward_calls,
        )
        self.assertTrue(
            self.mock_ve_process.sigma_t.call_count >= expected_forward_calls
        )


if __name__ == "__main__":
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestImputationSampler))
    runner = unittest.TextTestRunner()
    runner.run(suite)
