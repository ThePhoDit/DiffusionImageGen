# === Imports ===
import torch
import pandas as pd
import argparse
import os
import sys
import ipywidgets as widgets
from IPython.display import display, clear_output

# === Setup Project Path ===
# Assuming your notebook is in the 'notebooks' directory
project_root = ".."
if project_root not in sys.path:
    sys.path.append(project_root)

# === Import Project Modules ===
# Import necessary classes and the evaluation function from main.py
try:
    from main import (
        VPDiffusionProcess,
        VEDiffusionProcess,
        SubVariancePreservingDiffusionProcess,
        run_evaluation,
        load_data,  # Needed for download side-effect and test set loading in eval
    )
except ImportError as e:
    print(f"Error importing from main.py: {e}")
    print(
        "Please ensure main.py exists in the parent directory and the notebook kernel has access."
    )
    # Stop execution if imports fail
    raise

# === Define Widgets (Adapted from selections.py) ===

# --- Available Samplers based on Process ---
samplers_options = {
    "Variance Exploding": ["euler", "pc", "ode"],
    "Variance Preserving": ["euler", "pc", "ode", "ei"],
    "Sub-variance Preserving": ["euler", "pc", "ode", "ei"],
}

# --- Dataset Selection ---
dataset_selector = widgets.Dropdown(
    options=["mnist", "cifar10"],
    value="mnist",
    description="Dataset:",
    style={"description_width": "initial"},
)

# --- Model Selection ---
model_path_input = widgets.Text(
    value="../mnist_model/vp_cosine_50_uncond_epoch20.pth",  # Example path
    placeholder="Enter path to model checkpoint (.pth)",
    description="Model Path:",
    layout=widgets.Layout(width="90%"),
    style={"description_width": "initial"},
)

# --- Conditioning ---
use_class_condition = widgets.Checkbox(
    value=False, description="Use Class Condition", indent=False
)
target_class = widgets.IntSlider(
    value=0,
    min=0,
    max=9,
    step=1,
    description="Target Class:",
    continuous_update=False,
    disabled=True,
    style={"description_width": "initial"},
)


def toggle_target_class(change):
    target_class.disabled = not change["new"]


use_class_condition.observe(toggle_target_class, names="value")

conditioning_box = widgets.VBox([use_class_condition, target_class])

# --- Diffusion Process Selection ---
diffusion_process_selector = widgets.Dropdown(
    options=list(samplers_options.keys()),
    value="Variance Preserving",
    description="Diffusion Process:",
    style={"description_width": "initial"},
)

# --- Noise Schedule Selection (VP/Sub-VP Only) ---
noise_schedule = widgets.Dropdown(
    options=["linear", "cosine"],
    description="Noise Schedule:",
    value="cosine",
    style={"description_width": "initial"},
)

# Schedule Params (Minimal for eval - assume defaults or model name implies them)
# You could add sliders like in selections.py if needed, but often we rely on model convention
schedule_params_output = widgets.Output()  # Keep output area for consistency

# --- Sampler Selection ---
sampler_selector = widgets.Dropdown(
    options=samplers_options[diffusion_process_selector.value],
    description="Sampler:",
    value="ei",
    style={"description_width": "initial"},
)

# --- Sampler Parameter Widgets ---
common_gen_steps = widgets.IntSlider(
    value=1000,
    min=100,
    max=10000,
    step=100,
    description="Steps:",
    continuous_update=False,
    style={"description_width": "initial"},
)
common_gen_eps = widgets.FloatLogSlider(
    value=1e-3,
    base=10,
    min=-5,
    max=-1,
    step=0.1,
    description="Epsilon (t_end):",
    readout_format=".1e",
    style={"description_width": "initial"},
)

# PC Parameters
pc_snr = widgets.FloatSlider(
    value=0.1,
    min=0.01,
    max=1.0,
    step=0.01,
    description="PC SNR:",
    continuous_update=False,
    style={"description_width": "initial"},
)
pc_corrector_steps = widgets.IntSlider(
    value=1,
    min=1,
    max=10,
    step=1,
    description="PC Corr Steps:",
    continuous_update=False,
    style={"description_width": "initial"},
)
params_pc = widgets.VBox(
    [common_gen_steps, common_gen_eps, pc_snr, pc_corrector_steps]
)

# ODE Parameters
ode_early_stop_time = widgets.FloatSlider(
    value=0.0,
    min=0.0,
    max=0.1,
    step=0.001,
    description="ODE Early Stop (0=off):",
    readout_format=".3f",
    continuous_update=False,
    style={"description_width": "initial"},
)
ode_use_rk4 = widgets.Checkbox(value=True, description="Use RK4", indent=False)
params_ode = widgets.VBox(
    [common_gen_steps, common_gen_eps, ode_early_stop_time, ode_use_rk4]
)

# Euler/EI Parameters
params_euler_ei = widgets.VBox([common_gen_steps, common_gen_eps])

sampler_param_widgets = {
    "euler": params_euler_ei,
    "pc": params_pc,
    "ode": params_ode,
    "ei": params_euler_ei,
}
sampler_params_output = widgets.Output()  # Dynamic display area

# --- Evaluation Specific Parameters ---
num_eval_samples = widgets.IntSlider(
    value=100,
    min=100,
    max=50000,
    step=100,
    description="# Eval Samples:",
    style={"description_width": "initial"},
)
eval_batch_size = widgets.IntText(
    value=64,
    description="Eval Batch Size:",
    style={"description_width": "initial"},
)
eval_box = widgets.VBox([num_eval_samples, eval_batch_size])

# --- Button and Output Area ---
evaluate_button = widgets.Button(description="Run Evaluation")
evaluation_output = widgets.Output()  # To capture prints and display results

# === Widget Callbacks (Adapted from selections.py) ===


# Update sampler options and noise schedule display based on process
def update_sampler_options(change):
    process_type = change["new"]
    valid_samplers = samplers_options.get(process_type, [])
    is_ve = process_type == "Variance Exploding"

    # Update Sampler options
    current_sampler_value = sampler_selector.value
    sampler_selector.options = valid_samplers
    if current_sampler_value in valid_samplers:
        sampler_selector.value = current_sampler_value
    elif valid_samplers:
        sampler_selector.value = valid_samplers[0]
    else:
        sampler_selector.value = None

    # Update Noise Schedule Visibility/State
    noise_schedule.disabled = is_ve


diffusion_process_selector.observe(update_sampler_options, names="value")


# Update dynamic display of sampler parameters
def update_sampler_params_display(change):
    selected_sampler = change["new"]
    with sampler_params_output:
        clear_output(wait=True)
        param_widget_box = sampler_param_widgets.get(selected_sampler)
        if param_widget_box:
            display(param_widget_box)


sampler_selector.observe(update_sampler_params_display, names="value")


# Update dynamic display of schedule parameters (simplified for now)
def update_schedule_params_display(change):
    with schedule_params_output:
        clear_output(wait=True)
        if not noise_schedule.disabled:
            # Optionally display beta/cosine sliders here if added
            pass


noise_schedule.observe(update_schedule_params_display, names="value")


# === Evaluation Function triggered by Button ===
def run_evaluation_from_widgets(button):
    with evaluation_output:
        clear_output(wait=True)  # Clear previous results
        print("Starting evaluation from widget values...")

        # --- 1. Construct args Namespace from Widgets ---
        args = argparse.Namespace()
        args.mode = "evaluate"
        args.dataset = dataset_selector.value
        args.model_path = model_path_input.value

        process_map = {
            "Variance Exploding": "ve",
            "Variance Preserving": "vp",
            "Sub-variance Preserving": "subvp",
        }
        args.process = process_map.get(diffusion_process_selector.value, "vp")

        args.schedule = noise_schedule.value
        args.beta_min = 0.01  # Using fixed defaults, add widgets if needed
        args.beta_max = 0.95
        args.cosine_s = 0.008

        args.use_class_condition = use_class_condition.value
        args.target_class = (
            target_class.value if args.use_class_condition else None
        )

        args.num_eval_samples = num_eval_samples.value
        args.eval_batch_size = eval_batch_size.value
        args.sampler = sampler_selector.value
        args.gen_steps = common_gen_steps.value
        args.gen_eps = common_gen_eps.value

        # Sampler specific
        args.pc_snr = pc_snr.value
        args.pc_corrector_steps = pc_corrector_steps.value
        args.ode_early_stop_time = (
            ode_early_stop_time.value
            if ode_early_stop_time.value > 1e-6
            else None
        )
        args.ode_use_rk4 = ode_use_rk4.value

        # Other needed args (can take defaults)
        args.filter_dataset = False  # Usually false for eval
        args.notebook_tqdm = True  # Use notebook version
        args.image_size = None
        args.image_channels = 3

        print(f"Dataset: {args.dataset.upper()}")
        print(f"Model: {args.model_path}")
        print(f"Process: {args.process.upper()}")
        if args.process in ["vp", "subvp"]:
            print(f"Schedule: {args.schedule}")
        print(f"Sampler: {args.sampler}")
        print(f"Steps: {args.gen_steps}")
        print(f"Conditioning: {args.use_class_condition}")
        if args.use_class_condition and args.target_class is not None:
            print(f"Target Class: {args.target_class}")

        # --- 2. Set Dataset Specific Globals & Defaults (Copied Logic) ---
        # Declare globals to modify them
        global NUM_CLASSES, SAVE_DIR, MODEL_DIR, DATASET_NAME
        if args.dataset == "mnist":
            DATASET_NAME = "MNIST"
            NUM_CLASSES = 10
            MODEL_DIR = "mnist_model"
            SAVE_DIR = "mnist_results"
            if args.image_size is None:
                args.image_size = 28
            if args.image_channels is None:
                args.image_channels = 1
        elif args.dataset == "cifar10":
            DATASET_NAME = "CIFAR-10"
            NUM_CLASSES = 10
            MODEL_DIR = "cifar10_model"
            SAVE_DIR = "cifar10_results"
            if args.image_size is None:
                args.image_size = 32
            if args.image_channels is None:
                args.image_channels = 3
        else:
            print(f"Error: Unknown dataset {args.dataset}")
            return

        # Check model path existence
        if not args.model_path or not os.path.exists(args.model_path):
            print(
                f"Error: Model path not found or not specified: {args.model_path}"
            )
            return

        os.makedirs(SAVE_DIR, exist_ok=True)

        # --- 3. Instantiate Diffusion Process (Copied Logic) ---
        print(f"Instantiating {args.process.upper()} process...")
        try:
            if args.process == "vp":
                diffusion_process = VPDiffusionProcess(
                    schedule=args.schedule,
                    beta_min=args.beta_min,
                    beta_max=args.beta_max,
                    cosine_s=args.cosine_s,
                )
            elif args.process == "ve":
                diffusion_process = VEDiffusionProcess()
            elif args.process == "subvp":
                diffusion_process = SubVariancePreservingDiffusionProcess(
                    schedule=args.schedule,
                    beta_min=args.beta_min,
                    beta_max=args.beta_max,
                    cosine_s=args.cosine_s,
                )
            else:
                raise ValueError(f"Unknown process type: {args.process}")
        except Exception as e:
            print(f"Error instantiating diffusion process: {e}")
            return

        # --- 4. Run Evaluation ---
        print("Running evaluation function...")
        # Ensure dataset is downloaded
        try:
            load_data(args)
        except Exception:
            pass  # Ignore errors if already loaded

        try:
            # Pass the locally determined NUM_CLASSES
            results_dict = run_evaluation(args, diffusion_process, NUM_CLASSES)
            print("Evaluation finished.")
        except Exception as e:
            print(f"\n--- ERROR during evaluation ---")
            print(e)
            import traceback

            traceback.print_exc()  # Print detailed traceback
            results_dict = None

        # --- 5. Display Results ---
        if results_dict:
            df_data = {
                "Metric": ["Avg Loss", "FID", "Inception Score"],
                "Value": [
                    f"{results_dict['avg_loss']:.4f}",
                    f"{results_dict['fid_score']:.4f}",
                    f"{results_dict['inception_score_mean']:.4f} +/- {results_dict['inception_score_std']:.4f}",
                ],
            }
            results_df = pd.DataFrame(df_data)
            print("\n--- Evaluation Summary ---")
            display(results_df)
        else:
            print("\nEvaluation function did not return results or failed.")


# Link button click to function
evaluate_button.on_click(run_evaluation_from_widgets)

# === Display Widgets ===
# Initial setup of dynamic widgets
update_sampler_options({"new": diffusion_process_selector.value})
update_sampler_params_display({"new": sampler_selector.value})
update_schedule_params_display({"new": noise_schedule.value})
toggle_target_class({"new": use_class_condition.value})  # Initial state

# Arrange layout
controls = widgets.VBox(
    [
        widgets.HTML("<b>Evaluation Configuration:</b>"),
        dataset_selector,
        model_path_input,
        conditioning_box,
        widgets.HTML("<hr><b>Diffusion Process & Schedule:</b>"),
        diffusion_process_selector,
        noise_schedule,
        schedule_params_output,
        widgets.HTML("<hr><b>Sampler Selection:</b>"),
        sampler_selector,
        sampler_params_output,
        widgets.HTML("<hr><b>Evaluation Parameters:</b>"),
        eval_box,
        widgets.HTML("<hr>"),
        evaluate_button,
        evaluation_output,  # Output area for messages and results
    ]
)


def get_controls():
    return controls
