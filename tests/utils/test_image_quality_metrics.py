import unittest
import torch
import numpy as np
from unittest.mock import patch, MagicMock

# Need to potentially mock torchvision models and transforms if not installed
# Or ensure they are installed in the test environment
try:
    import torchvision.transforms as transforms
    import torchvision.models as models
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    # Create dummy mocks if torchvision is not available
    # This allows testing the structure without the dependency,
    # but real functionality tests require torchvision.
    print(
        "Warning: torchvision not found. Using mocks for InceptionV3 and transforms."
    )
    models = MagicMock()
    models.inception_v3.return_value = MagicMock()
    transforms = MagicMock()
    transforms.Resize = MagicMock()
    transforms.Normalize = MagicMock()
    DataLoader = MagicMock()
    TensorDataset = MagicMock()

from diffusion.utils.image_quality_metrics import (
    get_inception_v3,
    preprocess_for_inception,
    calculate_activations,  # Assuming this might be tested indirectly
    calculate_average_loss,
    calculate_fid_from_activations,
    calculate_fid,
    calculate_inception_score,
)

# Mock the diffusion process needed for average loss
mock_diffusion_process = MagicMock()
mock_diffusion_process.loss_fn.return_value = torch.tensor(
    0.5
)  # Mock loss value

# Mock the score model needed for average loss
mock_score_model = MagicMock()


class TestImageQualityMetrics(unittest.TestCase):

    def setUp(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.batch_size = 4
        self.channels = 3
        self.height = 32
        self.width = 32

    @patch("diffusion.utils.image_quality_metrics.models.inception_v3")
    def test_get_inception_v3(self, mock_inception_load):
        """Test getting the InceptionV3 model."""
        mock_model_instance = MagicMock()
        mock_model_instance.to.return_value = mock_model_instance  # Chain .to()
        mock_model_instance.eval.return_value = (
            mock_model_instance  # Chain .eval()
        )
        mock_inception_load.return_value = mock_model_instance

        model = get_inception_v3(self.device)

        mock_inception_load.assert_called_once_with(
            pretrained=True, transform_input=False
        )
        # Check that the final layer is replaced
        self.assertEqual(mock_model_instance.fc.__class__.__name__, "Identity")
        mock_model_instance.to.assert_called_once_with(self.device)
        mock_model_instance.eval.assert_called_once()
        self.assertEqual(model, mock_model_instance)

    def test_preprocess_for_inception(self):
        """Test the preprocessing steps for InceptionV3 input."""
        # Test with range [-1, 1]
        images_neg1_1 = (
            torch.rand(self.batch_size, self.channels, self.height, self.width)
            * 2
            - 1
        )
        processed_neg1_1 = preprocess_for_inception(images_neg1_1)
        # Should resize and have 3 channels
        self.assertEqual(processed_neg1_1.shape, (self.batch_size, 3, 299, 299))
        # Output range depends on Normalize, difficult to test exactly without it mocked better
        # Check that the input tensor values were likely scaled from [-1, 1] to [0, 1] before normalize
        # This assumes Normalize doesn't drastically change the mean from ~0.5
        self.assertTrue(
            processed_neg1_1.mean().item() < 0.6
            and processed_neg1_1.mean().item() > -0.6
        )

        # Test with range [0, 1]
        images_0_1 = torch.rand(
            self.batch_size, self.channels, self.height, self.width
        )
        processed_0_1 = preprocess_for_inception(images_0_1)
        self.assertEqual(processed_0_1.shape, (self.batch_size, 3, 299, 299))
        self.assertTrue(
            processed_0_1.mean().item() < 0.6
            and processed_0_1.mean().item() > -0.6
        )

        # Test with grayscale
        images_gray = torch.rand(self.batch_size, 1, self.height, self.width)
        processed_gray = preprocess_for_inception(images_gray)
        self.assertEqual(processed_gray.shape, (self.batch_size, 3, 299, 299))

    def test_calculate_fid_from_activations(self):
        """Test FID calculation from pre-computed activations."""
        # Simple case: identical activations should yield FID near 0
        act1 = np.random.rand(100, 2048).astype(np.float32)
        fid0 = calculate_fid_from_activations(act1, act1)
        self.assertAlmostEqual(fid0, 0.0, delta=1e-4)

        # Slightly different activations
        act2 = act1 + 0.1  # Shift mean slightly
        fid1 = calculate_fid_from_activations(act1, act2)
        self.assertTrue(fid1 > 0)

        # Test complex handling (mock sqrtm to return complex)
        with patch(
            "diffusion.utils.image_quality_metrics.linalg.sqrtm"
        ) as mock_sqrtm:
            complex_result = (np.random.rand(2048, 2048) + 1j * 1e-4).astype(
                np.complex64
            )
            mock_sqrtm.return_value = (
                complex_result,
                None,
            )  # Simulate sqrtm output
            # Use activations with different means ensure ssdiff > 0
            act1_diff_mean = np.random.rand(100, 2048).astype(np.float32)
            act2_diff_mean = act1_diff_mean + 0.2
            fid_complex = calculate_fid_from_activations(
                act1_diff_mean, act2_diff_mean
            )
            # Should warn but proceed by taking real part
            # Ensure it calls sqrtm
            mock_sqrtm.assert_called()

    @patch("diffusion.utils.image_quality_metrics.calculate_activations")
    @patch(
        "diffusion.utils.image_quality_metrics.calculate_fid_from_activations"
    )
    def test_calculate_fid(self, mock_fid_calc, mock_activations):
        """Test the main FID function, mocking activation calculation."""
        # Mock activation calculation results
        real_act = np.random.rand(100, 2048).astype(np.float32)
        gen_act = np.random.rand(90, 2048).astype(np.float32)  # Different sizes
        mock_activations.side_effect = [real_act, gen_act]
        mock_fid_calc.return_value = 15.5

        mock_dataloader = MagicMock()  # Mock a dataloader
        gen_tensor = torch.rand(90, self.channels, self.height, self.width)
        mock_inception = MagicMock()

        fid_score = calculate_fid(
            mock_dataloader,
            gen_tensor,
            mock_inception,
            self.device,
            max_samples=100,
        )

        # Should use the smaller number of samples (90)
        self.assertEqual(mock_activations.call_count, 2)
        mock_fid_calc.assert_called_once()
        # Check that activations passed to fid_calc were truncated
        args_fid, _ = mock_fid_calc.call_args
        self.assertEqual(args_fid[0].shape[0], 90)
        self.assertEqual(args_fid[1].shape[0], 90)
        self.assertAlmostEqual(fid_score, 15.5)

    @patch("diffusion.utils.image_quality_metrics.preprocess_for_inception")
    @patch(
        "diffusion.utils.image_quality_metrics.models.inception_v3"
    )  # Mock the original model load inside IS
    def test_calculate_inception_score(
        self, mock_inception_load_orig, mock_preprocess
    ):
        """Test Inception Score calculation (mocking model predictions)."""
        num_samples = 50
        num_classes = 10
        gen_tensor = torch.rand(
            num_samples, self.channels, self.height, self.width
        )
        mock_preprocess.return_value = torch.rand(
            self.batch_size, 3, 299, 299
        )  # Preprocess output

        # Mock the output of the *original* InceptionV3 (logits)
        mock_inception_orig_instance = MagicMock()
        mock_inception_orig_instance.to.return_value = (
            mock_inception_orig_instance
        )
        mock_inception_orig_instance.eval.return_value = (
            mock_inception_orig_instance
        )
        # Return mock logits when the original model is called inside calculate_is
        mock_logits = torch.randn(
            self.batch_size, num_classes, device=self.device
        )
        mock_inception_orig_instance.return_value = mock_logits
        mock_inception_load_orig.return_value = mock_inception_orig_instance

        # The inception model passed to the function is only used for device/eval, not prediction in this impl
        mock_inception_arg = MagicMock()

        is_mean, is_std = calculate_inception_score(
            gen_tensor,
            mock_inception_arg,
            self.device,
            batch_size=self.batch_size,
            splits=2,
        )

        # Check types and basic validity
        self.assertTrue(
            isinstance(is_mean, (float, np.floating)),
            f"is_mean type is {type(is_mean)}",
        )
        self.assertTrue(
            isinstance(is_std, (float, np.floating)),
            f"is_std type is {type(is_std)}",
        )
        self.assertGreaterEqual(is_mean, 0)
        self.assertGreaterEqual(is_std, 0)
        mock_inception_load_orig.assert_called()
        self.assertTrue(mock_preprocess.call_count > 0)

    # Test for calculate_average_loss
    def test_calculate_average_loss(self):
        """Test average loss calculation using mock process and model."""
        num_batches = 3
        mock_dataset = TensorDataset(
            torch.rand(
                self.batch_size * num_batches,
                self.channels,
                self.height,
                self.width,
            ),
            torch.randint(0, 10, (self.batch_size * num_batches,)),
        )  # Add dummy labels
        mock_dataloader = DataLoader(mock_dataset, batch_size=self.batch_size)

        avg_loss = calculate_average_loss(
            mock_diffusion_process,
            mock_score_model,
            mock_dataloader,
            self.device,
            num_batches=num_batches,  # Limit batches for test
        )

        self.assertIsInstance(avg_loss, float)
        # Check if the process loss_fn was called num_batches times
        self.assertEqual(mock_diffusion_process.loss_fn.call_count, num_batches)
        self.assertAlmostEqual(avg_loss, 0.5)  # Based on the mock return value


if __name__ == "__main__":
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestImageQualityMetrics))
    runner = unittest.TextTestRunner()
    runner.run(suite)
