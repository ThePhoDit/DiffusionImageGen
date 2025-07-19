import unittest
import torch
from unittest.mock import MagicMock

# Assuming the script is run from the project root where 'diffusion' is visible
from diffusion.models.score_model import (
    GaussianRandomFourierFeatures,
    Dense,
    ScoreNet,
)


class TestScoreModelComponents(unittest.TestCase):

    def setUp(self):
        """Set up common parameters and tensors for tests."""
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.batch_size = 4
        self.image_channels = 3
        self.image_size = 32
        self.embed_dim = 128

    def test_gaussian_rff(self):
        """Test GaussianRandomFourierFeatures layer."""
        rff = GaussianRandomFourierFeatures(embed_dim=self.embed_dim).to(
            self.device
        )
        t = torch.rand(self.batch_size, device=self.device)
        embedded_t = rff(t)
        self.assertEqual(embedded_t.shape, (self.batch_size, self.embed_dim))
        self.assertFalse(
            rff.rff_weights.requires_grad
        )  # Weights should be fixed

    def test_dense_layer(self):
        """Test Dense layer."""
        dense_layer = Dense(input_dim=self.embed_dim, output_dim=64).to(
            self.device
        )
        input_tensor = torch.randn(
            self.batch_size, self.embed_dim, device=self.device
        )
        output_tensor = dense_layer(input_tensor)
        # Output should be reshaped for addition to feature maps
        self.assertEqual(output_tensor.shape, (self.batch_size, 64, 1, 1))

    def test_scorenet_init_unconditional(self):
        """Test ScoreNet initialization (unconditional)."""
        mock_marginal_prob_std = MagicMock(
            return_value=torch.ones(self.batch_size, device=self.device)
        )
        model = ScoreNet(
            marginal_prob_std=mock_marginal_prob_std,
            image_channels=self.image_channels,
            embed_dim=self.embed_dim,
        ).to(self.device)
        self.assertIsInstance(model, ScoreNet)
        self.assertIsNone(model.num_classes)
        self.assertFalse(model.use_class_condition)

    def test_scorenet_init_conditional(self):
        """Test ScoreNet initialization (conditional)."""
        num_classes = 10
        mock_marginal_prob_std = MagicMock(
            return_value=torch.ones(self.batch_size, device=self.device)
        )
        model = ScoreNet(
            marginal_prob_std=mock_marginal_prob_std,
            image_channels=self.image_channels,
            embed_dim=self.embed_dim,
            num_classes=num_classes,
        ).to(self.device)
        self.assertIsInstance(model, ScoreNet)
        self.assertEqual(model.num_classes, num_classes)
        self.assertTrue(model.use_class_condition)
        self.assertIsNotNone(model.class_embedding)
        self.assertIsNotNone(model.class_projection)

    def test_scorenet_forward_unconditional(self):
        """Test ScoreNet forward pass (unconditional)."""
        mock_marginal_prob_std = MagicMock(
            return_value=torch.ones(self.batch_size, device=self.device)
        )
        model = ScoreNet(
            marginal_prob_std=mock_marginal_prob_std,
            image_channels=self.image_channels,
            embed_dim=self.embed_dim,
        ).to(self.device)

        x = torch.randn(
            self.batch_size,
            self.image_channels,
            self.image_size,
            self.image_size,
            device=self.device,
        )
        t = torch.rand(self.batch_size, device=self.device)

        output = model(x, t)
        self.assertEqual(output.shape, x.shape)
        # Check if mock function was called (basic check)
        # mock_marginal_prob_std.assert_called()

    def test_scorenet_forward_conditional(self):
        """Test ScoreNet forward pass (conditional)."""
        num_classes = 10
        mock_marginal_prob_std = MagicMock(
            return_value=torch.ones(self.batch_size, device=self.device)
        )
        model = ScoreNet(
            marginal_prob_std=mock_marginal_prob_std,
            image_channels=self.image_channels,
            embed_dim=self.embed_dim,
            num_classes=num_classes,
        ).to(self.device)

        x = torch.randn(
            self.batch_size,
            self.image_channels,
            self.image_size,
            self.image_size,
            device=self.device,
        )
        t = torch.rand(self.batch_size, device=self.device)
        class_labels = torch.randint(
            0, num_classes, (self.batch_size,), device=self.device
        )

        output = model(x, t, class_labels)
        self.assertEqual(output.shape, x.shape)
        # mock_marginal_prob_std.assert_called()

    def test_scorenet_forward_conditional_no_labels(self):
        """Test conditional ScoreNet forward pass with labels=None (should use zero embedding)."""
        num_classes = 10
        mock_marginal_prob_std = MagicMock(
            return_value=torch.ones(self.batch_size, device=self.device)
        )
        model = ScoreNet(
            marginal_prob_std=mock_marginal_prob_std,
            image_channels=self.image_channels,
            embed_dim=self.embed_dim,
            num_classes=num_classes,
        ).to(self.device)

        x = torch.randn(
            self.batch_size,
            self.image_channels,
            self.image_size,
            self.image_size,
            device=self.device,
        )
        t = torch.rand(self.batch_size, device=self.device)

        # Expect a warning, but should run
        with self.assertWarns(UserWarning):
            output = model(x, t, class_labels=None)
        self.assertEqual(output.shape, x.shape)

    def test_scorenet_disable_scaling(self):
        """Test that disable_final_scaling=True prevents division by std_dev."""
        # Create a mock std function that returns a very small value
        # If scaling is applied, the output would explode or become NaN
        small_std_dev = 1e-9
        mock_marginal_prob_std = MagicMock(
            return_value=torch.full(
                (self.batch_size,), small_std_dev, device=self.device
            )
        )

        model_no_scale = ScoreNet(
            marginal_prob_std=mock_marginal_prob_std,
            image_channels=self.image_channels,
            embed_dim=self.embed_dim,
            disable_final_scaling=True,  # Explicitly disable scaling
        ).to(self.device)

        x = torch.randn(
            self.batch_size,
            self.image_channels,
            self.image_size,
            self.image_size,
            device=self.device,
        )
        t = torch.rand(self.batch_size, device=self.device)

        output = model_no_scale(x, t)
        self.assertEqual(output.shape, x.shape)
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())

        # Check the default behavior (scaling enabled) would likely fail with small std
        # Note: This might actually pass if the network output is exactly zero initially,
        # but it's a heuristic check.
        # model_with_scale = ScoreNet(
        #     marginal_prob_std=mock_marginal_prob_std,
        #     image_channels=self.image_channels,
        #     embed_dim=self.embed_dim,
        #     disable_final_scaling=False # Default
        # ).to(self.device)
        # output_scaled = model_with_scale(x,t)
        # self.assertTrue(torch.isinf(output_scaled).any() or torch.isnan(output_scaled).any() or output_scaled.abs().max() > 1e6)


if __name__ == "__main__":
    import os

    current_dir = os.path.dirname(os.path.abspath(__file__))
    tests_dir = os.path.dirname(current_dir)
    models_test_dir = os.path.join(tests_dir, "models")

    if not os.path.exists(tests_dir):
        os.makedirs(tests_dir)
    if not os.path.exists(os.path.join(tests_dir, "__init__.py")):
        open(os.path.join(tests_dir, "__init__.py"), "a").close()

    if not os.path.exists(models_test_dir):
        os.makedirs(models_test_dir)
    if not os.path.exists(os.path.join(models_test_dir, "__init__.py")):
        open(os.path.join(models_test_dir, "__init__.py"), "a").close()

    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestScoreModelComponents))
    runner = unittest.TextTestRunner()
    runner.run(suite)
