# -*- coding: utf-8 -*-
"""
Functions for calculating image quality metrics for generative models:
- Average Loss (related to BPD)
- Fréchet Inception Distance (FID)
- Inception Score (IS)
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from scipy import linalg  # For FID
import warnings

# --- Inception Model Helper ---


def get_inception_v3(device):
    """Load pre-trained Inception-v3 model."""
    # Load pre-trained Inception v3
    inception_model = models.inception_v3(
        pretrained=True, transform_input=False
    )
    # Modify the model to output features from the final average pooling layer
    inception_model.fc = nn.Identity()  # Remove final classification layer
    inception_model = inception_model.to(device)
    inception_model.eval()
    return inception_model


def preprocess_for_inception(images_tensor):
    """Preprocess images for Inception-v3: resize, normalize [-1, 1] -> [0, 1] -> ImageNet norm."""
    # 1. Ensure input is [0, 1] range if it's [-1, 1]
    if images_tensor.min() < -0.1:  # Heuristic check for [-1, 1]
        images_tensor = (images_tensor + 1.0) / 2.0
    images_tensor = torch.clamp(images_tensor, 0.0, 1.0)

    # 2. Resize to Inception's expected size (299x299)
    resizer = transforms.Resize((299, 299), antialias=True)
    images_tensor = resizer(images_tensor)

    # 3. Repeat channels if grayscale
    if images_tensor.shape[1] == 1:
        images_tensor = images_tensor.repeat(1, 3, 1, 1)

    # 4. Apply ImageNet normalization
    normalizer = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    images_tensor = normalizer(images_tensor)

    return images_tensor


def calculate_activations(
    dataloader_or_tensor, model, device, max_samples=None
):
    """Calculate Inception activations for a dataset or tensor."""
    model.eval()
    activations = []

    total_processed = 0

    if isinstance(dataloader_or_tensor, DataLoader):
        print("Calculating activations from DataLoader...")
        iterator = iter(dataloader_or_tensor)
        with torch.no_grad():
            while True:
                try:
                    batch_data = next(iterator)
                    # Handle datasets returning (image, label) or just image
                    if isinstance(batch_data, (list, tuple)):
                        batch = batch_data[0].to(device)
                    else:
                        batch = batch_data.to(device)

                    if (
                        max_samples is not None
                        and total_processed >= max_samples
                    ):
                        break

                    current_batch_size = batch.shape[0]
                    if max_samples is not None:
                        remaining = max_samples - total_processed
                        if current_batch_size > remaining:
                            batch = batch[:remaining]
                            current_batch_size = remaining

                    batch = preprocess_for_inception(batch)
                    pred = model(batch)
                    activations.append(pred.cpu().numpy())
                    total_processed += current_batch_size
                    if (
                        max_samples is not None
                        and total_processed >= max_samples
                    ):
                        break  # Exit loop after processing enough
                except StopIteration:
                    break  # End of dataloader
    elif torch.is_tensor(dataloader_or_tensor):
        print("Calculating activations from Tensor...")
        tensor_data = dataloader_or_tensor
        if max_samples is not None and tensor_data.shape[0] > max_samples:
            tensor_data = tensor_data[:max_samples]

        total_processed = tensor_data.shape[0]
        # Process tensor in batches if needed (e.g., batch_size=50)
        batch_size = 50
        num_batches = int(np.ceil(total_processed / batch_size))
        with torch.no_grad():
            for i in range(num_batches):
                start = i * batch_size
                end = min((i + 1) * batch_size, total_processed)
                batch = tensor_data[start:end].to(device)
                batch = preprocess_for_inception(batch)
                pred = model(batch)
                activations.append(pred.cpu().numpy())
    else:
        raise TypeError("Input must be a PyTorch DataLoader or Tensor")

    activations = np.concatenate(activations, axis=0)
    if max_samples is not None:
        activations = activations[:max_samples]  # Ensure exact count

    print(f"Calculated activations shape: {activations.shape}")
    return activations


# --- BPD Surrogate: Average Loss ---


def calculate_average_loss(
    diffusion_process,
    score_model,
    dataloader,
    device,
    use_class_condition=False,
    num_batches=None,
):
    """
    Calculates the average loss (surrogate for NLL/BPD) on a dataset.
    Lower is generally better.
    """
    score_model.eval()
    total_loss = 0.0
    total_samples = 0
    batch_count = 0

    print("Calculating average loss...")
    with torch.no_grad():
        for batch_data in dataloader:
            # Handle datasets returning (image, label) or just image
            if isinstance(batch_data, (list, tuple)):
                x, y = batch_data
                y = y.to(device) if use_class_condition else None
            else:
                x = batch_data
                y = None

            x = x.to(device)
            loss = diffusion_process.loss_fn(score_model, x, y)

            if torch.isnan(loss) or torch.isinf(loss):
                warnings.warn(
                    f"NaN/Inf loss detected in batch {batch_count}. Skipping batch."
                )
                continue

            total_loss += loss.item() * x.shape[0]
            total_samples += x.shape[0]
            batch_count += 1
            if num_batches is not None and batch_count >= num_batches:
                break

    if total_samples == 0:
        return float("inf")

    avg_loss = total_loss / total_samples
    print(f"Average Loss: {avg_loss:.4f}")
    return avg_loss


# --- FID: Fréchet Inception Distance ---
# Requires scipy: pip install scipy


def calculate_fid_from_activations(act1, act2):
    """Calculates FID between two sets of activations."""
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)

    # Calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2) ** 2.0)

    # Calculate sqrt of product between cov
    # Add small identity matrix for numerical stability
    eps = 1e-6
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        warnings.warn(
            "FID calculation produced singular product; adding epsilon to diagonal of cov estimates"
        )
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(
                f"Imaginary component {m} too high in matrix sqrt calculation"
            )
        covmean = covmean.real

    # Calculate score
    tr_covmean = np.trace(covmean)
    fid = ssdiff + np.trace(sigma1) + np.trace(sigma2) - 2.0 * tr_covmean

    return fid


def calculate_fid(
    real_dataloader,
    generated_images_tensor,
    inception_model,
    device,
    max_samples=None,
):
    """
    Calculates FID between real images (from dataloader) and generated images (tensor).
    Lower is better.
    """
    # Calculate activations
    real_activations = calculate_activations(
        real_dataloader, inception_model, device, max_samples=max_samples
    )
    gen_activations = calculate_activations(
        generated_images_tensor,
        inception_model,
        device,
        max_samples=max_samples,
    )

    # Ensure same number of samples are compared if max_samples was used
    num_samples = min(real_activations.shape[0], gen_activations.shape[0])
    if num_samples == 0:
        warnings.warn("Not enough samples to calculate FID.")
        return float("inf")

    real_activations = real_activations[:num_samples]
    gen_activations = gen_activations[:num_samples]
    print(f"Calculating FID using {num_samples} samples.")

    fid_value = calculate_fid_from_activations(
        real_activations, gen_activations
    )
    print(f"FID: {fid_value:.4f}")
    return fid_value


# --- IS: Inception Score ---


def calculate_inception_score(
    generated_images_tensor,
    inception_model,
    device,
    batch_size=50,
    splits=10,
    eps=1e-16,
):
    """
    Calculates the Inception Score for a set of generated images.
    Higher is better.
    """
    N = generated_images_tensor.shape[0]
    if N == 0:
        warnings.warn("No generated images provided for Inception Score.")
        return 0.0, 0.0

    print(f"Calculating Inception Score using {N} samples...")
    inception_model.eval()
    preds = []
    num_batches = int(np.ceil(N / batch_size))

    # Get predictions
    with torch.no_grad():
        for i in range(num_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, N)
            batch = generated_images_tensor[start:end].to(device)
            batch = preprocess_for_inception(batch)
            # Need logits for Inception v3 to apply softmax
            logits = inception_model(batch)
            # If using the modified model returning features, need the original model for classification
            # Let's reload original Inception v3 for classification probabilities
            original_inception = (
                models.inception_v3(pretrained=True, transform_input=False)
                .to(device)
                .eval()
            )
            logits = original_inception(batch)
            # -----------------------------------------------------------------------------
            p = torch.nn.functional.softmax(logits, dim=1)
            preds.append(p.cpu().numpy())

    preds = np.concatenate(preds, axis=0)

    # Calculate scores
    scores = []
    for i in range(splits):
        part = preds[i * (N // splits) : (i + 1) * (N // splits), :]
        # Calculate p(y) - marginal distribution
        py = np.mean(part, axis=0)
        # Calculate KL divergence D_KL(p(y|x) || p(y)) for each x
        kl_divs = []
        for k in range(part.shape[0]):
            pyx = part[k, :]
            # D_KL = sum[ p(y|x) * log( p(y|x) / p(y) ) ]
            kl = np.sum(pyx * (np.log(pyx + eps) - np.log(py + eps)))
            kl_divs.append(kl)
        # IS = exp( mean[ D_KL(...) ] )
        scores.append(np.exp(np.mean(kl_divs)))

    is_mean = np.mean(scores)
    is_std = np.std(scores)
    print(f"Inception Score: {is_mean:.4f} +/- {is_std:.4f}")
    return is_mean, is_std


# --- Example Usage (Illustrative) ---
if __name__ == "__main__":
    # This block is for illustration and testing. You'll integrate these calls
    # into your main evaluation script after loading models and data properly.
    print("Image Quality Metrics Module - Example Usage")

    # --- !! Placeholder Setup !! ---
    # Replace these with your actual loaded objects and data based on your main script's logic
    print("\n--- Placeholder Setup --- ")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Instantiate Diffusion Process (Example for VE)
    try:
        from ..processes.variance_exploding_sde import VEDiffusionProcess

        diffusion_process = VEDiffusionProcess()
        print(f"Placeholder: Instantiated {diffusion_process.kind} process")
    except ImportError:
        print("Error: Could not import VEDiffusionProcess. Cannot proceed.")
        exit(1)

    # 2. Load Trained Score Model (Using generic_load_model pattern)
    # --- Parameters you need to set based on your model ---
    placeholder_model_path = "mnist_model/model_ve_uncond_all_sz28.pth"  # <-- Replace with your VE model path
    placeholder_image_channels = 3  # <-- Set to 1 (grayscale) or 3 (RGB)
    placeholder_image_size = 28  # <-- Set to your model's image size
    placeholder_is_conditional = (
        True  # <-- Set to True if your model is class-conditional
    )
    num_classes = 10 if placeholder_is_conditional else None
    # ------------------------------------------------------
    try:
        from ..processes.variance_exploding_sde import generic_load_model

        print(
            f"Placeholder: Attempting to load model from: {placeholder_model_path}"
        )
        score_model = generic_load_model(
            diffusion_process,
            placeholder_model_path,
            image_channels=placeholder_image_channels,
            num_classes=num_classes,
        )
        print("Placeholder: Loaded score model successfully.")
    except ImportError:
        print("Error: Could not import generic_load_model. Cannot load model.")
        score_model = None
    except FileNotFoundError:
        print(
            f"Error: Model file not found at {placeholder_model_path}. Cannot load model."
        )
        score_model = None
    except Exception as e:
        print(f"Error loading model: {e}")
        score_model = None

    # 3. Create DataLoader for Real Test Data
    # --- Parameters you need to set ---
    placeholder_batch_size = 64
    # ----------------------------------
    try:
        img_transforms = transforms.Compose(
            [
                transforms.Resize(
                    (placeholder_image_size, placeholder_image_size)
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5,) * placeholder_image_channels,
                    (0.5,) * placeholder_image_channels,
                ),
            ]
        )
        # Use train=False for the test set
        real_dataset = datasets.MNIST(
            "./data", train=False, download=True, transform=img_transforms
        )
        real_dataloader = DataLoader(
            real_dataset, batch_size=placeholder_batch_size
        )
        print(
            f"Placeholder: Created DataLoader for MNIST Test Set (Size: {len(real_dataset)})"
        )
    except Exception as e:
        print(f"Error creating real data loader: {e}")
        real_dataloader = None

    # 4. Generate Images (Using generic_generate_images pattern)
    # --- Parameters you need to set ---
    num_eval_samples = 1000  # Use 10k-50k for real FID/IS, smaller for testing
    placeholder_gen_steps = 500  # Use appropriate steps for your model
    placeholder_sampler = "euler"  # Choose sampler used for generation
    # ----------------------------------
    generated_images = None
    if score_model:
        try:
            from ..processes.variance_exploding_sde import (
                generate_images as generic_generate_images,
            )

            print(f"Placeholder: Generating {num_eval_samples} images...")
            generated_images = generic_generate_images(
                diffusion_process,
                score_model,
                n_images=num_eval_samples,
                image_size=(placeholder_image_size, placeholder_image_size),
                n_channels=placeholder_image_channels,
                n_steps=placeholder_gen_steps,
                sampler_type=placeholder_sampler,
                # Add other sampler args like pc_snr, ode_*, etc. if needed
            )
            print(
                f"Placeholder: Generated images tensor shape: {generated_images.shape}"
            )
        except ImportError:
            print(
                "Error: Could not import generic_generate_images. Cannot generate images."
            )
        except Exception as e:
            print(f"Error during image generation: {e}")
    else:
        print("Skipping image generation because model loading failed.")

    print("------------------------\n")
    # --- !! End Placeholder Setup !! ---

    # --- BPD Surrogate ---
    if diffusion_process and score_model and real_dataloader:
        print("\n--- Calculating Average Loss (BPD Surrogate) ---")
        avg_loss = calculate_average_loss(
            diffusion_process,
            score_model,
            real_dataloader,
            device,
            num_batches=10,
        )  # Limit batches for example
        print(f"---> Average Loss (BPD Surrogate): {avg_loss:.4f}")
    else:
        print("\nSkipping Average Loss calculation due to missing components.")

    # --- FID / IS ---
    if generated_images is not None and real_dataloader:
        print("\n--- Calculating FID & IS --- ")
        print("Loading Inception V3 for FID/IS...")
        try:
            inception_model_fid = get_inception_v3(device)
            print("Inception model loaded.")

            # --- FID ---
            print("\nCalculating FID...")
            # Using max_samples=num_eval_samples here for example consistency.
            fid_score = calculate_fid(
                real_dataloader,
                generated_images,
                inception_model_fid,
                device,
                max_samples=num_eval_samples,
            )
            print(f"---> FID Score: {fid_score}")

            # --- IS ---
            print("\nCalculating Inception Score...")
            is_mean, is_std = calculate_inception_score(
                generated_images, inception_model_fid, device, splits=2
            )  # Reduce splits for example
            print(f"---> Inception Score: {is_mean:.4f} +/- {is_std:.4f}")

        except ImportError:
            print(
                "\nFID/IS calculation requires scipy. Please install it: pip install scipy"
            )
        except Exception as e:
            print(f"\nError during FID/IS calculation: {e}")
    else:
        print(
            "\nSkipping FID/IS calculation due to missing generated images or real dataloader."
        )

    print("\n--- Example Usage Finished ---")
    pass
