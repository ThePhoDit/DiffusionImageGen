import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os
import argparse
from PIL import Image
import h5py
import numpy as np
import torch.nn.functional as F

# --- Import specific process classes and generic helpers ---
from diffusion.processes.variance_preserving import VPDiffusionProcess
from diffusion.processes.variance_exploding_sde import (
    VEDiffusionProcess,
    train as generic_train,  # Rename to avoid conflict
    generate_images as generic_generate_images,
    load_model as generic_load_model,
)
from diffusion.processes.subvariance_preserving import (
    SubVariancePreservingDiffusionProcess,
)

# --- Add sampler import ---
from diffusion.samplers.probability_flow_ode import probability_flow_ode_sampler
from diffusion.samplers.exponential_integrator import (
    exponential_integrator_sampler,
)  # Actually ETD1 now
from diffusion.samplers.imputation_sampler import (
    repaint_sampler,
)  # <-- Import repaint sampler

# --- Add metrics imports ---
from diffusion.utils.image_quality_metrics import (
    calculate_average_loss,
    calculate_fid,
    calculate_inception_score,
    get_inception_v3,
)

# --------------------------------------------------------

from diffusion.utils.diffusion_utilities import (
    plot_image_grid,
    plot_image_evolution,
)
from torchvision.utils import save_image  # <-- Need save_image


# --- Argument Parsing ---
def parse_args():
    # Use common defaults favoring CIFAR first, override later based on dataset
    parser = argparse.ArgumentParser(
        description="Train or generate images (MNIST/CIFAR-10/Colored MNIST variants) with selectable diffusion process."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        choices=["mnist", "cifar10", "colored_mnist", "colored_mnist_h5"],
        help="Dataset to use: 'mnist', 'cifar10', or 'colored_mnist_h5' (HDF5 dataset).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "generate", "impute", "evaluate"],
        help="Operation mode.",
    )
    parser.add_argument(
        "--process",
        type=str,
        default="vp",
        choices=["ve", "vp", "subvp"],
        help="Diffusion process type.",
    )
    parser.add_argument(
        "--schedule",
        type=str,
        default="linear",
        choices=["linear", "cosine"],
        help="VP/Sub-VP Schedule.",
    )
    parser.add_argument(
        "--beta_min", type=float, default=0.01, help="Linear schedule beta_min."
    )
    parser.add_argument(
        "--beta_max", type=float, default=0.95, help="Linear schedule beta_max."
    )
    parser.add_argument(
        "--cosine_s", type=float, default=0.008, help="Cosine schedule s."
    )
    parser.add_argument(
        "--target_class",
        type=int,
        default=None,
        help="Target class index (0-9).",
    )
    parser.add_argument(
        "--use_class_condition",
        action="store_true",
        help="Enable class conditioning.",
    )
    parser.add_argument(
        "--filter_dataset",
        action="store_true",
        help="Filter training dataset to target_class.",
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Training epochs (default 50)."
    )  # Default higher
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size."
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Learning rate (default 1e-4)."
    )  # Default lower
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to load/save model. Default determined by dataset/config.",
    )
    parser.add_argument(
        "--num_generate",
        type=int,
        default=16,
        help="Number of images to generate.",
    )
    parser.add_argument(
        "--gen_steps",
        type=int,
        default=1000,
        help="Sampler steps for generation.",
    )
    parser.add_argument(
        "--image_channels",
        type=int,
        default=None,
        help="Image channels (default: 1 for MNIST, 3 for CIFAR).",
    )  # Set default later
    parser.add_argument(
        "--image_size",
        type=int,
        default=None,
        help="Image size (default: 28 for MNIST, 32 for CIFAR).",
    )  # Set default later
    parser.add_argument(
        "--gen_eps",
        type=float,
        default=1e-3,
        help="Reverse SDE integration endpoint.",
    )
    parser.add_argument(
        "--sampler",
        type=str,
        default="euler",
        choices=["euler", "pc", "ode", "ei", "etd1"],
        help="Sampler type.",
    )
    parser.add_argument(
        "--pc_snr", type=float, default=0.1, help="PC sampler SNR."
    )
    parser.add_argument(
        "--pc_corrector_steps",
        type=int,
        default=1,
        help="PC sampler corrector steps.",
    )
    parser.add_argument(
        "--ode_early_stop_time",
        type=float,
        default=None,
        help="ODE sampler early stop time.",
    )
    parser.add_argument(
        "--no_ode_rk4",
        action="store_false",
        dest="ode_use_rk4",
        help="Use Euler instead of RK4 for ODE sampler.",
    )
    parser.add_argument(
        "--input_image",
        type=str,
        default=None,
        help="Input image for imputation.",
    )
    parser.add_argument(
        "--mask_image",
        type=str,
        default=None,
        help="Mask image for imputation.",
    )
    parser.add_argument(
        "--output_image",
        type=str,
        default=None,
        help="Output path for imputed image.",
    )  # Set default later
    parser.add_argument(
        "--impute_steps",
        type=int,
        default=1000,
        help="Imputation sampler steps.",
    )
    parser.add_argument(
        "--jump_length", type=int, default=10, help="Repaint jump length N."
    )
    parser.add_argument(
        "--jump_n_sample",
        type=int,
        default=10,
        help="Repaint jump sample size R.",
    )
    # --- Evaluation Args ---
    parser.add_argument(
        "--num_eval_samples",
        type=int,
        default=10000,
        help="Number of samples for evaluation.",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=64,
        help="Batch size for evaluation.",
    )
    parser.add_argument(
        "--notebook_tqdm", action="store_true", help="Use tqdm.notebook."
    )

    args = parser.parse_args()
    return args


# --- Custom HDF5 Dataset Class ---
class ColoredMNISTH5Dataset(torch.utils.data.Dataset):
    """Custom Dataset for loading Colored MNIST from HDF5 files."""
    def __init__(self, h5_path, image_size, normalization_transform=None, train=True):
        self.h5_path = h5_path
        self.image_size = image_size # Expected output size (H, W)
        self.normalization_transform = normalization_transform
        self.train = train
        self.file = None
        self.images = None
        self.labels = None
        # Determine the correct file based on train flag
        actual_path = os.path.join(self.h5_path, 'training.h5' if self.train else 'testing.h5')
        if not os.path.exists(actual_path):
            raise FileNotFoundError(f"HDF5 file not found at {actual_path}. Make sure 'training.h5' and 'testing.h5' are in {self.h5_path}")
        try:
            self.file = h5py.File(actual_path, 'r')
            # Use the correct keys identified
            self.images = self.file['images']
            self.labels = self.file['digits']
            # Optional: Add 'colors' if needed later
            self.length = len(self.images)
        except KeyError as e:
            print(f"Error accessing HDF5 keys in {actual_path}. Found keys: {list(self.file.keys())}")
            print("Expected keys: 'images', 'digits' (and optionally 'colors').")
            if self.file:
                self.file.close()
            raise e
        except Exception as e:
            print(f"Error opening or reading HDF5 file {actual_path}: {e}")
            if self.file:
                self.file.close()
            raise e

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.file is None:
            raise RuntimeError("HDF5 file is not open. Access dataset after context management or ensure __del__ is not called prematurely.")

        # Load image and label
        img_data = self.images[idx] # Read from HDF5 -> NumPy array
        label_data = self.labels[idx]
        label = int(label_data)

        img_transformed = None
        # Handle different data types from HDF5
        if img_data.dtype == np.uint8:
            # Path for uint8 data (assumed 0-255)
            if img_data.ndim == 3 and img_data.shape[2] == 3: # HWC uint8
                img_pil = Image.fromarray(img_data, 'RGB')
            elif img_data.ndim == 2: # HW uint8 (grayscale) -> convert to RGB
                 img_pil = Image.fromarray(img_data).convert('RGB')
                 print(f"Warning: Read uint8 grayscale image (shape {img_data.shape}) at index {idx}, converted to RGB.")
            else:
                 raise TypeError(f"Unsupported uint8 image shape from HDF5: {img_data.shape}")
            # Apply Resize -> ToTensor -> Normalize
            img_resized = transforms.Resize(self.image_size)(img_pil)
            img_tensor = transforms.ToTensor()(img_resized) # PIL [0,255] -> Tensor [0,1]
            if self.normalization_transform:
                img_transformed = self.normalization_transform(img_tensor)
            else:
                img_transformed = img_tensor # Keep as [0,1] if no normalization

        elif np.issubdtype(img_data.dtype, np.floating):
             # Path for float data (assumed 0.0-1.0)
            if img_data.ndim == 3 and img_data.shape[2] == 3: # HWC float
                 # Convert HWC NumPy [0,1] to CHW Tensor [0,1]
                 img_tensor_chw = torch.from_numpy(img_data.transpose((2, 0, 1))).float()
            elif img_data.ndim == 2: # HW float (grayscale?)
                  print(f"Warning: Read float grayscale image (shape {img_data.shape}) at index {idx}, converting to RGB tensor.")
                  img_tensor_chw = torch.from_numpy(img_data).float().unsqueeze(0).repeat(3, 1, 1)
            else:
                  raise TypeError(f"Unsupported float image shape from HDF5: {img_data.shape}")
            # Apply Resize (interpolate) -> Normalize
            # Interpolate requires shape (N, C, H, W)
            img_resized = F.interpolate(img_tensor_chw.unsqueeze(0), size=self.image_size, mode='bilinear', align_corners=False).squeeze(0)
            if self.normalization_transform:
                 img_transformed = self.normalization_transform(img_resized)
            else:
                 img_transformed = img_resized # Keep as [0,1] if no normalization

        else:
            raise TypeError(f"Unsupported HDF5 image dtype: {img_data.dtype}")

        # Ensure final output is float tensor
        img_transformed = img_transformed.float()

        return img_transformed, label

    def __del__(self):
        if self.file:
            self.file.close()
            self.file = None # Prevent double closing


# --- Unified Data Loading ---
def load_data(args):
    """Load and preprocess the selected dataset."""
    print(f"Loading {args.dataset.upper()} dataset...")
    image_size_tuple = (args.image_size, args.image_size)

    # Common transforms (Resize and ToTensor are almost always needed first)
    transform_list = [
        transforms.Resize(image_size_tuple),
        transforms.ToTensor(),  # Converts to [0, 1] range PIL Image/NumPy array
    ]

    # --- Dataset Specific Loading & Transformations ---
    if args.dataset == "mnist":
        # Basic MNIST: Ensure correct channels & normalization
        if args.image_channels == 3:
            print("Converting MNIST to 3 channels.")
            transform_list.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1)))
            transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        else:
            transform_list.append(transforms.Normalize((0.5,), (0.5,))) # Single channel

        data_transform = transforms.Compose(transform_list)
        train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=data_transform)
        test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=data_transform)
        num_classes = 10

    elif args.dataset == "cifar10":
        # CIFAR-10: Normalize 3 channels
        transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        data_transform = transforms.Compose(transform_list)
        train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=data_transform)
        test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=data_transform)
        num_classes = 10

    elif args.dataset == "colored_mnist_h5":
        # Specific HDF5 Colored MNIST
        print(f"Loading Colored MNIST from HDF5 files in ./data/ ...")
        h5_dir = "./data" # Directory containing the HDF5 files

        # Define ONLY the normalization transform here
        # Resize and ToTensor are handled within the Dataset class
        normalization_only_transform = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        train_h5_file = os.path.join(h5_dir, "training.h5")
        test_h5_file = os.path.join(h5_dir, "testing.h5")
        if not os.path.exists(train_h5_file):
             raise FileNotFoundError(f"HDF5 training file not found at {train_h5_file}. Please download from https://github.com/deepmind/colored_mnist and place it in {h5_dir}/")
        if not os.path.exists(test_h5_file):
             raise FileNotFoundError(f"HDF5 testing file not found at {test_h5_file}. Please download from https://github.com/deepmind/colored_mnist and place it in {h5_dir}/")

        train_dataset = ColoredMNISTH5Dataset(h5_path=h5_dir, image_size=image_size_tuple, normalization_transform=normalization_only_transform, train=True)
        test_dataset = ColoredMNISTH5Dataset(h5_path=h5_dir, image_size=image_size_tuple, normalization_transform=normalization_only_transform, train=False)
        num_classes = 10 # MNIST digits 0-9

    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    if args.filter_dataset and args.target_class is not None:
        print(
            f"Filtering {args.dataset.upper()} dataset for class {args.target_class}"
        )
        indices = [
            i
            for i, (_, label) in enumerate(train_dataset)
            if label == args.target_class
        ]
        if not indices:
            print(f"Warning: No samples found for class {args.target_class}!")
            return Subset(train_dataset, [])  # Return empty subset
        train_dataset = Subset(train_dataset, indices)
        print(
            f"Using {len(train_dataset)} samples for class {args.target_class}"
        )
    else:
        print(
            f"Using all {len(train_dataset)} {args.dataset.upper()} training samples."
        )

    # !!! Important: Need to reload the test dataset here if it's HDF5
    # because the previous test_dataset instance was created for HDF5
    # but the standard path below assumes torchvision datasets.
    # Load Test Dataset (conditionally)
    if args.dataset != "colored_mnist_h5":
        # Standard loading for MNIST/CIFAR
        print(f"Loading standard {args.dataset.upper()} test set...")
        # Need to reconstruct the appropriate transform for standard datasets
        standard_transform_list = [
            transforms.Resize(image_size_tuple),
            transforms.ToTensor(),
        ]
        if args.dataset == "mnist":
            dataset_class = datasets.MNIST
            if args.image_channels == 3:
                standard_transform_list.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1)))
                standard_transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
            else:
                standard_transform_list.append(transforms.Normalize((0.5,), (0.5,)))
        elif args.dataset == "cifar10":
            dataset_class = datasets.CIFAR10
            standard_transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        else:
            # This case should technically not be reached due to earlier check
            raise ValueError(f"Unhandled dataset type for test loading: {args.dataset}")

        standard_test_transform = transforms.Compose(standard_transform_list)
        test_dataset = dataset_class(root="./data", train=False, download=True, transform=standard_test_transform)

    print(
        f"Loaded {args.dataset.upper()} test set with {len(test_dataset)} samples."
    )

    return train_dataset, test_dataset


# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# --- Globals to be set in main() based on args.dataset ---
NUM_CLASSES = None
SAVE_DIR = None
MODEL_DIR = None
DATASET_NAME = None
# ----------------------------------------------------------

# --- Unified Wrapper Functions ---


def train_model(args, diffusion_process):
    """Wrapper to set up and call generic_train."""
    train_dataset, _ = load_data(args)  # Only need train dataset here

    print(
        f"Training model ({diffusion_process.kind} on {DATASET_NAME}) with class conditioning: {args.use_class_condition}"
    )
    print(f"Saving model to: {args.model_path}")
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)

    # Use the generic train function
    trained_model = generic_train(
        diffusion_process,
        train_dataset,
        batch_size=args.batch_size,
        n_epochs=args.epochs,
        learning_rate=args.lr,
        save_model_to=args.model_path,
        use_class_condition=args.use_class_condition,
        num_classes=NUM_CLASSES if args.use_class_condition else None,
        image_channels=args.image_channels,
        grad_clip_val=1.0,
        use_notebook_tqdm=args.notebook_tqdm,
    )
    return trained_model


def generate_images_wrapper(args, diffusion_process):
    """Wrapper to set up and call generic_generate_images."""
    print(f"Loading model from: {args.model_path}")
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        return

    load_as_conditional = args.use_class_condition
    print(
        f"Loading model as {'conditional' if load_as_conditional else 'unconditional'} based on --use_class_condition flag."
    )

    model = generic_load_model(
        diffusion_process,
        args.model_path,
        image_channels=args.image_channels,
        num_classes=NUM_CLASSES if load_as_conditional else None,
    )

    gen_class = args.target_class
    model_module = (
        model.module if isinstance(model, torch.nn.DataParallel) else model
    )
    model_is_truly_conditional = getattr(
        model_module, "use_class_condition", False
    )

    if gen_class is not None and not model_is_truly_conditional:
        print(
            f"Warning: Requesting class {gen_class} but loaded model is unconditional. Generating unconditioned images."
        )
        gen_class = None
    elif gen_class is None and model_is_truly_conditional:
        print(
            "Warning: Loaded model is conditional, but no target class specified. Generating unconditionally."
        )

    print(
        f"Generating {args.num_generate} {DATASET_NAME} images ({diffusion_process.kind})"
        + (
            f" for class {gen_class}"
            if gen_class is not None
            else " (unconditioned)"
        )
        + f" using {args.gen_steps} steps with {args.sampler} sampler..."
    )

    img_size_tuple = (args.image_size, args.image_size)

    final_images = generic_generate_images(
        diffusion_process,
        model,
        n_images=args.num_generate,
        image_size=img_size_tuple,
        n_channels=args.image_channels,
        target_class=gen_class,
        n_steps=args.gen_steps,
        eps=args.gen_eps,
        sampler_type=args.sampler,
        pc_snr=args.pc_snr,
        pc_num_corrector_steps=args.pc_corrector_steps,
        ode_early_stop_time=args.ode_early_stop_time,
        ode_use_rk4=args.ode_use_rk4,
        use_notebook_tqdm=args.notebook_tqdm,
    )

    print(f"Shape of final_images before plotting: {final_images.shape}")

    # Rescale images from [-1, 1] to [0, 1] for plotting
    final_images_display = (final_images + 1.0) / 2.0
    final_images_display = torch.clamp(final_images_display, 0.0, 1.0)

    n_cols = int(args.num_generate**0.5)
    n_rows = (args.num_generate + n_cols - 1) // n_cols
    fig, ax = plot_image_grid(
        final_images_display,
        figsize=(n_cols * 2, n_rows * 2),
        n_rows=n_rows,
        n_cols=n_cols,
        normalize=False,  # Assume images are [0, 1]
    )
    title = f"Generated {DATASET_NAME} ({diffusion_process.kind}"
    if diffusion_process.kind == "VP":
        title += f"_{diffusion_process.schedule_type}"
    title += ")" + (f" Class {gen_class}" if gen_class is not None else "")
    plt.suptitle(title)

    # Construct filename
    base_name = (
        os.path.basename(args.model_path).replace(".pth", "")
        if args.model_path
        else f"{args.dataset}_default"
    )
    gen_class_str = f"_class{gen_class}" if gen_class is not None else "_uncond"
    save_path = os.path.join(
        SAVE_DIR, f"generated_{base_name}{gen_class_str}.png"
    )  # Use global SAVE_DIR

    plt.savefig(save_path)
    print(f"Generated images saved to {save_path}")
    plt.close()


# --- Helper Function for Imputation Data (Largely dataset-agnostic) ---
def load_and_prepare_imputation_data(args):
    """Loads image and mask, resizes, normalizes, and creates binary mask tensor."""
    try:
        img = Image.open(args.input_image)
    except Exception as e:
        print(f"Error opening input image {args.input_image}: {e}")
        exit(1)

    mask = None
    if args.mask_image:
        try:
            mask = Image.open(args.mask_image)
        except Exception as e:
            print(f"Error opening mask image {args.mask_image}: {e}")
            exit(1)
    elif img.mode == "RGBA":
        print("Using alpha channel from input image as mask.")
        mask = img.getchannel("A")
    else:
        print(
            "Error: No mask image provided and input image does not have an alpha channel (RGBA)."
        )
        exit(1)

    # Basic Transforms (Resize, ToTensor)
    transform_list = [
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),  # Converts to [0, 1]
    ]
    img_transform = transforms.Compose(transform_list)
    mask_transform = transforms.Compose(
        [
            transforms.Resize(
                (args.image_size, args.image_size),
                interpolation=transforms.InterpolationMode.NEAREST,
            ),
            transforms.ToTensor(),
        ]
    )

    # Convert image based on requested channels
    if args.image_channels == 3:
        if img.mode == "RGBA":
            img = img.convert("RGB")  # Handle alpha transparency
        elif img.mode != "RGB":
            img = img.convert("RGB")
    elif args.image_channels == 1:
        if img.mode != "L":
            img = img.convert("L")
    else:
        raise ValueError(
            f"Imputation loading does not support {args.image_channels} channels."
        )

    img_tensor = img_transform(img)
    # Normalize image tensor to [-1, 1]
    normalize = transforms.Normalize(
        (0.5,) * args.image_channels, (0.5,) * args.image_channels
    )
    img_tensor = normalize(img_tensor)

    # Process mask
    mask = mask.convert("L")
    mask_tensor = mask_transform(mask)
    mask_tensor = (mask_tensor > 0.5).float()

    img_tensor = img_tensor.to(device).unsqueeze(0)
    mask_tensor = mask_tensor.to(device).unsqueeze(0)

    if args.image_channels > 1 and mask_tensor.shape[1] == 1:
        mask_tensor = mask_tensor.repeat(1, args.image_channels, 1, 1)

    print(f"Loaded image shape: {img_tensor.shape}")
    print(f"Created mask shape: {mask_tensor.shape}")

    return img_tensor, mask_tensor


# --- Main Imputation Function ---
def run_imputation(args, diffusion_process):
    """Wrapper to set up and call repaint_sampler."""
    print("Loading and preparing data for imputation...")
    img_tensor, mask_tensor = load_and_prepare_imputation_data(args)

    print(f"Loading model from: {args.model_path}")
    load_as_conditional = args.use_class_condition
    target_class_impute = args.target_class  # Store requested class
    imputation_class_labels = None  # Initialize labels as None

    if load_as_conditional:
        print(
            "Attempting to load a conditional model for imputation (--use_class_condition=True)."
        )
        # Load model specifying num_classes
        model = generic_load_model(
            diffusion_process,
            args.model_path,
            image_channels=args.image_channels,
            num_classes=NUM_CLASSES,  # Pass NUM_CLASSES
        )
        model.eval()
        # Check if the loaded model is *actually* conditional
        model_module = (
            model.module if isinstance(model, torch.nn.DataParallel) else model
        )
        model_is_truly_conditional = getattr(
            model_module, "use_class_condition", False
        )

        if not model_is_truly_conditional:
            print(
                "Warning: --use_class_condition specified, but loaded model appears UNconditional. Imputing unconditionally."
            )
            load_as_conditional = (
                False  # Override flag if model isn't conditional
            )
        elif target_class_impute is not None:
            print(
                f"Using class {target_class_impute} for conditional imputation."
            )
            batch_size = img_tensor.shape[0]  # Should be 1 for imputation
            imputation_class_labels = torch.full(
                (batch_size,),
                target_class_impute,
                dtype=torch.long,
                device=device,
            )
        else:
            print(
                "Warning: Loaded model is conditional, but no --target_class specified. Imputing unconditionally."
            )
            # imputation_class_labels remains None
    else:
        print("Loading an unconditional model for imputation.")
        # Load model without specifying num_classes
        model = generic_load_model(
            diffusion_process,
            args.model_path,
            image_channels=args.image_channels,
            num_classes=None,  # Ensure unconditional loading
        )
        model.eval()
        if target_class_impute is not None:
            print(
                "Warning: --target_class specified, but --use_class_condition is False. Imputing unconditionally."
            )

    print("Starting imputation sampling...")
    with torch.no_grad():
        times, samples_traj = repaint_sampler(
            diffusion_process=diffusion_process,
            score_model=model,
            x_masked=img_tensor,
            mask=mask_tensor,
            t_0=diffusion_process.T,
            t_end=args.gen_eps,
            n_steps=args.impute_steps,
            jump_length=args.jump_length,
            jump_n_sample=args.jump_n_sample,
            class_labels=imputation_class_labels,  # <-- Pass the determined labels
            use_notebook_tqdm=args.notebook_tqdm,
        )

    imputed_image = samples_traj[..., -1]

    # Determine default output path if not provided
    output_image_path = args.output_image
    if output_image_path is None:
        output_image_path = f"imputed_result_{args.dataset}.png"
    print(f"Saving imputed image to: {output_image_path}")

    imputed_image_display = (imputed_image.squeeze(0) + 1.0) / 2.0
    imputed_image_display = torch.clamp(imputed_image_display, 0.0, 1.0)

    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    save_image(imputed_image_display, output_image_path)

    print("Imputation finished.")


# --- Evaluation Function ---
def run_evaluation(args, diffusion_process, num_classes_eval):
    """Wrapper to set up and call evaluation logic."""
    print(f"Starting evaluation for model: {args.model_path} on {DATASET_NAME}")
    class_info = (
        f" for class {args.target_class}"
        if args.target_class is not None
        else " (all classes)"
    )
    print(f"Target Class: {class_info}")
    print(
        f"Process: {args.process}, Sampler: {args.sampler}, Steps: {args.gen_steps}, Eval Samples: {args.num_eval_samples}"
    )

    print("Loading score model...")
    load_as_conditional = args.use_class_condition

    target_requested = args.target_class is not None
    use_generation_conditioning = False
    if target_requested:
        if not load_as_conditional:
            print(
                f"Warning: Target class {args.target_class} requested, but --use_class_condition flag is OFF. Model loaded as unconditional."
            )
        else:
            print(
                f"Evaluation will generate samples conditioned on class {args.target_class}."
            )
            use_generation_conditioning = True
    elif load_as_conditional:
        print(
            "Warning: --use_class_condition flag is ON, but no target_class specified for generation. Generating unconditionally."
        )

    # --- DEBUG PRINT ---
    print(
        f"[DEBUG run_evaluation] Passing to generic_load_model: num_classes={num_classes_eval if load_as_conditional else None}, image_channels={args.image_channels}"
    )
    # --- END DEBUG ---

    # Add another debug print right before the previous one
    print(
        f"[DEBUG run_evaluation ENTRY] Received num_classes_eval={num_classes_eval}, args.use_class_condition={args.use_class_condition}, load_as_conditional={load_as_conditional}"
    )

    # Determine the correct number of channels based on the dataset name derived from args
    # This ensures the loaded model structure matches the checkpoint.
    if args.dataset.lower() == "mnist":
        load_image_channels = 3
        print(
            f"[DEBUG run_evaluation] Setting load_image_channels=1 for MNIST model."
        )
    elif args.dataset.lower() == "cifar10":
        load_image_channels = 3
        print(
            f"[DEBUG run_evaluation] Setting load_image_channels=3 for CIFAR10 model."
        )
    else:
        # Fallback to command-line arg if dataset name doesn't match known ones
        print(
            f"Warning: Unknown dataset '{args.dataset}' for determining model channels. "
            f"Falling back to args.image_channels={args.image_channels}. Checkpoint mismatch may occur."
        )
        load_image_channels = args.image_channels

    score_model = generic_load_model(
        diffusion_process,
        args.model_path,
        image_channels=load_image_channels,  # Use determined channels
        num_classes=num_classes_eval if load_as_conditional else None,
    )
    score_model.eval()

    print(f"Loading real test dataset ({DATASET_NAME})...")
    try:
        # Re-use load_data to get the test set with correct transforms
        _, full_real_dataset = load_data(args)  # Discard train set here

        if args.target_class is not None:
            print(
                f"Filtering real test dataset for class {args.target_class}..."
            )
            # Need to access original labels if Subset was used in load_data
            # Easier to just reload the test set here without filtering *initially*
            # Define transforms again (matching load_data logic)
            transform_list = [
                transforms.Resize((args.image_size, args.image_size)),
                transforms.ToTensor(),
            ]
            if args.dataset == "mnist":
                dataset_class = datasets.MNIST
                if args.image_channels == 3:
                    transform_list.append(
                        transforms.Lambda(lambda x: x.repeat(3, 1, 1))
                    )
            elif args.dataset == "cifar10":
                dataset_class = datasets.CIFAR10
                if args.image_channels == 1:
                    transform_list.append(
                        transforms.Grayscale(num_output_channels=1)
                    )
            else:
                raise ValueError(...)  # Should not happen

            transform_list.append(
                transforms.Normalize(
                    (0.5,) * args.image_channels, (0.5,) * args.image_channels
                )
            )
            img_transforms = transforms.Compose(transform_list)
            temp_full_test_dataset = dataset_class(
                "./data", train=False, download=True, transform=img_transforms
            )

            # Now filter the freshly loaded full test set
            indices = [
                i
                for i, (_, label) in enumerate(temp_full_test_dataset)
                if label == args.target_class
            ]
            if not indices:
                print(
                    f"Error: No samples found for class {args.target_class} in the {DATASET_NAME} test set."
                )
                return
            real_dataset_eval = Subset(temp_full_test_dataset, indices)
            print(
                f"Using {len(real_dataset_eval)} real samples for class {args.target_class}."
            )
        else:
            real_dataset_eval = (
                full_real_dataset  # Use the test set returned by load_data
            )
            print(
                f"Using full {DATASET_NAME} Test Set. Size: {len(real_dataset_eval)}"
            )

        real_dataloader = DataLoader(
            real_dataset_eval, batch_size=args.eval_batch_size, shuffle=False
        )

    except Exception as e:
        print(f"Error creating real test data loader: {e}")
        return

    print(
        f"Generating {args.num_eval_samples} images for evaluation{class_info}..."
    )
    generated_images_list = []
    num_generated = 0
    gen_batch_size = args.eval_batch_size
    with torch.no_grad():
        while num_generated < args.num_eval_samples:
            n_to_gen = min(
                gen_batch_size, args.num_eval_samples - num_generated
            )
            if n_to_gen <= 0:
                break
            print(f"  Generating batch of {n_to_gen}...")
            batch_images = generic_generate_images(
                diffusion_process,
                score_model,
                n_images=n_to_gen,
                image_size=(args.image_size, args.image_size),
                n_channels=args.image_channels,
                target_class=(
                    args.target_class if use_generation_conditioning else None
                ),
                n_steps=args.gen_steps,
                sampler_type=args.sampler,
                eps=args.gen_eps,
                pc_snr=args.pc_snr,
                pc_num_corrector_steps=args.pc_corrector_steps,
                ode_early_stop_time=args.ode_early_stop_time,
                ode_use_rk4=args.ode_use_rk4,
                use_notebook_tqdm=args.notebook_tqdm,
            )
            generated_images_list.append(batch_images.cpu())
            num_generated += n_to_gen

    generated_images_tensor = torch.cat(generated_images_list, dim=0)
    print(
        f"Generated {generated_images_tensor.shape[0]} images. Shape: {generated_images_tensor.shape}"
    )

    # --- Calculate Metrics ---
    print("\n--- Calculating Metrics --- ")

    inception_model = None
    try:
        print("Loading InceptionV3 model...")
        inception_model = get_inception_v3(device)
        print("InceptionV3 loaded.")
    except Exception as e:
        print(
            f"Could not load InceptionV3 model: {e}. FID and IS cannot be calculated."
        )

    # Calculate Average Loss
    print("\nCalculating Average Loss...")
    avg_loss = calculate_average_loss(
        diffusion_process,
        score_model,
        real_dataloader,
        device,
        num_batches=None,
    )

    # Calculate FID
    fid_score = float("nan")
    if inception_model:
        print("\nCalculating FID...")
        try:
            fid_score = calculate_fid(
                real_dataloader,
                generated_images_tensor,
                inception_model,
                device,
                max_samples=args.num_eval_samples,
            )
        except Exception as e:
            print(f"Error calculating FID: {e}")

    # Calculate Inception Score
    is_mean, is_std = float("nan"), float("nan")
    if inception_model:
        print("\nCalculating Inception Score...")
        try:
            is_mean, is_std = calculate_inception_score(
                generated_images_tensor, inception_model, device
            )
        except Exception as e:
            print(f"Error calculating Inception Score: {e}")

    # --- Print Results ---
    print(f"\n--- Evaluation Results ({DATASET_NAME}) --- ")
    print(f"Model: {args.model_path}")
    print(
        f"Target Class Evaluated: {args.target_class if args.target_class is not None else 'All'}"
    )
    print(
        f"Process: {args.process}, Sampler: {args.sampler}, Steps: {args.gen_steps}"
    )
    print(f"Number of Samples Used: {args.num_eval_samples}")
    print("---------------------------")
    print(f"Average Loss (BPD surrogate): {avg_loss:.4f}")
    print(f"FID Score: {fid_score:.4f}")
    print(f"Inception Score: {is_mean:.4f} +/- {is_std:.4f}")
    print("---------------------------")

    # Return metrics as a dictionary
    return {
        "model_path": args.model_path,
        "dataset": args.dataset,
        "process": args.process,
        "sampler": args.sampler,
        "gen_steps": args.gen_steps,
        "num_eval_samples": args.num_eval_samples,
        "avg_loss": avg_loss,
        "fid_score": fid_score,
        "inception_score_mean": is_mean,
        "inception_score_std": is_std,
    }


def main():
    args = parse_args()
    print(f"Selected Dataset: {args.dataset.upper()}")
    print(f"Selected mode: {args.mode}")
    print(f"Selected diffusion process: {args.process}")

    # --- Set Dataset Specific Globals & Argument Defaults ---
    global NUM_CLASSES, SAVE_DIR, MODEL_DIR, DATASET_NAME
    if args.dataset == "mnist":
        DATASET_NAME = "MNIST"
        NUM_CLASSES = 10
        MODEL_DIR = "mnist_model"
        SAVE_DIR = "mnist_results"
        # Override args if they weren't explicitly set or still have CIFAR defaults
        if args.image_size is None:
            args.image_size = 28
        if args.image_channels is None:
            args.image_channels = 3
        if args.output_image is None:
            args.output_image = "imputed_result_mnist.png"

    elif args.dataset == "cifar10":
        DATASET_NAME = "CIFAR-10"
        NUM_CLASSES = 10
        MODEL_DIR = "cifar10_model"
        SAVE_DIR = "cifar10_results"
        # Override args if they weren't explicitly set or still have MNIST defaults
        if args.image_size is None:
            args.image_size = 32
        if args.image_channels is None:
            args.image_channels = 3
        # Keep epochs/lr defaults from argparse (more suitable for CIFAR)
        if args.output_image is None:
            args.output_image = "imputed_result_cifar10.png"
    elif args.dataset == "colored_mnist":
        DATASET_NAME = "Colored MNIST"
        NUM_CLASSES = 10
        MODEL_DIR = "colored_mnist_model"
        SAVE_DIR = "colored_mnist_results"
        if args.image_size is None:
            args.image_size = 28
        if args.image_channels is None:
            args.image_channels = 3
        if args.output_image is None:
            args.output_image = "imputed_result_colored_mnist.png"
    elif args.dataset == "colored_mnist_h5":
        DATASET_NAME = "Colored MNIST HDF5"
        NUM_CLASSES = 10
        MODEL_DIR = "colored_mnist_h5_model"
        SAVE_DIR = "colored_mnist_h5_results"
        if args.image_size is None:
            args.image_size = 28
        if args.image_channels is None:
            args.image_channels = 3
        if args.output_image is None:
            args.output_image = "imputed_result_colored_mnist_h5.png"
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # Determine final model path if not provided by user
    if args.model_path is None:
        process_str = args.process
        schedule_str = f"_{args.schedule}" if args.process == "vp" else ""
        cond_str = "_cond" if args.use_class_condition else "_uncond"
        class_str = (
            f"_class{args.target_class}"
            if args.target_class is not None and args.filter_dataset
            else "_all"
        )
        args.model_path = os.path.join(
            MODEL_DIR,
            f"model_{process_str}{schedule_str}{cond_str}{class_str}_sz{args.image_size}.pth",
        )

    # Ensure SAVE_DIR exists
    os.makedirs(SAVE_DIR, exist_ok=True)
    # --------------------------------------------------------

    # --- Instantiate the correct diffusion process ---
    if args.process == "vp":
        print(f"Using VP Schedule: {args.schedule}")
        # Determine dataset-specific cosine beta clamps
        if args.schedule == "cosine":
            if DATASET_NAME == "MNIST":
                c_beta_min = 1e-4
                c_beta_max = 20.0
                print(f"Applying MNIST-specific cosine beta clamp [{c_beta_min:.1e}, {c_beta_max:.1f}]")
            else: # CIFAR or other
                c_beta_min = 1e-7
                c_beta_max = 0.999
                print(f"Applying default cosine beta clamp [{c_beta_min:.1e}, {c_beta_max:.1f}]")
        else:
            c_beta_min = None # Not used for linear
            c_beta_max = None # Not used for linear

        diffusion_process = VPDiffusionProcess(
            schedule=args.schedule,
            beta_min=args.beta_min,      # Used for linear schedule
            beta_max=args.beta_max,      # Used for linear schedule
            cosine_s=args.cosine_s,      # Used for cosine schedule
            cosine_beta_min=c_beta_min,  # Pass determined cosine min clamp
            cosine_beta_max=c_beta_max   # Pass determined cosine max clamp
        )
    elif args.process == "ve":
        diffusion_process = VEDiffusionProcess()  # Add VE params if needed
    elif args.process == "subvp":  # <-- Add Sub-VP case
        print(f"Using Sub-VP Schedule: {args.schedule}")
        diffusion_process = SubVariancePreservingDiffusionProcess(
            schedule=args.schedule,
            beta_min=args.beta_min,
            beta_max=args.beta_max,
            cosine_s=args.cosine_s,
            # eps_var could be added as an arg later if needed
        )
    else:
        raise ValueError(f"Unknown process type: {args.process}")
    # --- End of diffusion process instantiation ---

    # --- Execute selected mode ---
    if args.mode == "train":
        train_model(args, diffusion_process)  # Use unified function
    elif args.mode == "generate":
        generate_images_wrapper(args, diffusion_process)  # Use unified function
    elif args.mode == "impute":
        print(f"--- Running Image Imputation ({DATASET_NAME}) --- ")
        if not args.model_path or not args.input_image:
            print(
                "Error: --model_path and --input_image are required for impute mode."
            )
            exit(1)
        if not os.path.exists(args.model_path):
            print(f"Error: Model file not found at {args.model_path}")
            exit(1)
        if not os.path.exists(args.input_image):
            print(f"Error: Input image file not found at {args.input_image}")
            exit(1)
        if args.mask_image and not os.path.exists(args.mask_image):
            print(f"Error: Mask image file not found at {args.mask_image}")
            exit(1)
        run_imputation(args, diffusion_process)
    elif args.mode == "evaluate":
        print(f"--- Running Model Evaluation ({DATASET_NAME}) ---")
        if not args.model_path:
            print("Error: --model_path is required for evaluate mode.")
            exit(1)
        if not os.path.exists(args.model_path):
            print(f"Error: Model file not found at {args.model_path}")
            exit(1)
        run_evaluation(args, diffusion_process, NUM_CLASSES)


# ----------------------------------------

if __name__ == "__main__":
    main()
