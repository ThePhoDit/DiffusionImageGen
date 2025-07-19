import ipywidgets as widgets
from IPython.display import display
import os  # To potentially list models later

# --- Available Samplers based on Process ---
# Based on our findings:
# - VE works best with Euler, PC. ODE is unstable. EI is invalid.
# - VP works with Euler, PC, ODE, EI (ETD1).
# - Sub-VP should behave similarly to VP regarding sampler compatibility.
# We'll keep ODE for VE but maybe add a note later if displayed.
# We'll keep EI disabled for VE.
samplers_options = {
    "Variance Exploding": ["euler", "pc", "ode"],
    "Variance Preserving": ["euler", "pc", "ode", "ei"],
    "Sub-variance Preserving": ["euler", "pc", "ode", "ei"],
}

# --- Model Selection ---
# Using Text input for flexibility with paths
# Could be changed to Dropdown if models are listed from a directory
model_path_input = widgets.Text(
    value="mnist_model/model_vp_linear_uncond_class_all_sz28.pth",  # Default example
    placeholder="Enter path to model checkpoint (.pth)",
    description="Model Path:",
    disabled=False,
    style={"description_width": "initial"},
    layout=widgets.Layout(width="90%"),  # Make it wider
)

# --- Dataset Selection ---
dataset_selector = widgets.Dropdown(
    options=["mnist", "cifar10"],
    value="mnist",  # Default value
    description="Dataset:",
    disabled=False,
    style={"description_width": "initial"},
)

# --- Data/Model Configuration ---
image_size = widgets.IntText(
    value=28, description="Image Size:", style={"description_width": "initial"}
)
use_class_condition = widgets.Checkbox(
    value=True, description="Use Class Condition", indent=False
)
target_class = widgets.IntSlider(
    value=0,
    min=0,
    max=9,
    step=1,
    description="Target Class:",
    continuous_update=False,
    style={"description_width": "initial"},
)


# Disable target_class slider if class conditioning is not used
def toggle_target_class(change):
    target_class.disabled = not change["new"]


use_class_condition.observe(toggle_target_class, names="value")
target_class.disabled = not use_class_condition.value  # Initial state

data_model_box = widgets.VBox([image_size, use_class_condition, target_class])

# --- Diffusion Process Selection ---
diffusion_process = widgets.Dropdown(
    options=list(samplers_options.keys()),
    description="Diffusion Process:",
    disabled=False,
    style={"description_width": "initial"},  # Wider label
)

# --- Sampler Selection ---
sampler = widgets.Dropdown(
    options=samplers_options[diffusion_process.value],
    description="Sampler:",
    disabled=False,
    style={"description_width": "initial"},
)

# --- Noise Schedule Selection (VP/Sub-VP Only) ---
noise_schedule = widgets.Dropdown(
    options=["linear", "cosine"],
    description="Noise Schedule:",
    # Initial state based on default process
    disabled=(diffusion_process.value == "Variance Exploding"),
    style={"description_width": "initial"},
)

# --- Noise Schedule Parameter Widgets (Relevant for VP/Sub-VP) ---
# Linear Schedule Params
linear_beta_min = widgets.FloatSlider(
    value=0.0001,
    min=1e-5,
    max=0.1,
    step=1e-4,
    description="Beta Min:",
    readout_format=".4f",
    continuous_update=False,
    style={"description_width": "initial"},
)
linear_beta_max = widgets.FloatSlider(
    value=20,
    min=0.1,
    max=40,
    step=0.01,
    description="Beta Max:",
    readout_format=".3f",
    continuous_update=False,
    style={"description_width": "initial"},
)
params_linear = widgets.VBox([linear_beta_min, linear_beta_max])

# Cosine Schedule Params
cosine_s = widgets.FloatSlider(
    value=0.008,
    min=0.001,
    max=0.1,
    step=0.001,
    description="Cosine s:",
    readout_format=".3f",
    continuous_update=False,
    style={"description_width": "initial"},
)
params_cosine = widgets.VBox([cosine_s])

# Dictionary mapping schedule name to its parameter widget VBox
schedule_param_widgets = {"linear": params_linear, "cosine": params_cosine}

# Output widget to display schedule parameters dynamically
schedule_params_output = widgets.Output()


# --- Sampler Parameter Widgets ---

# Common Parameters (used by all/most)
common_gen_steps = widgets.IntSlider(
    value=1000,
    min=100,
    max=5000,
    step=50,
    description="Steps:",
    continuous_update=False,
    style={"description_width": "initial"},
)
common_gen_eps = widgets.FloatSlider(
    value=1e-3,
    min=1e-5,
    max=1e-1,
    step=1e-4,
    description="Epsilon (t_end):",
    readout_format=".4f",
    continuous_update=False,
    style={"description_width": "initial"},
)

# Euler Parameters Box
params_euler = widgets.VBox([common_gen_steps, common_gen_eps])

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
    description="PC Corrector Steps:",
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
ode_use_rk4 = widgets.Checkbox(
    value=True, description="Use RK4 (Uncheck for Euler)", indent=False
)
params_ode = widgets.VBox(
    [common_gen_steps, common_gen_eps, ode_early_stop_time, ode_use_rk4]
)

# EI Parameters Box (same as Euler for now)
params_ei = widgets.VBox([common_gen_steps, common_gen_eps])

# Dictionary mapping sampler name to its parameter widget VBox
sampler_param_widgets = {
    "euler": params_euler,
    "pc": params_pc,
    "ode": params_ode,
    "ei": params_ei,
}

# Output widget to display sampler parameters dynamically
sampler_params_output = widgets.Output()

# --- Training Parameter Widgets ---
train_epochs = widgets.IntSlider(
    value=10,
    min=1,
    max=200,
    step=1,
    description="Epochs:",
    style={"description_width": "initial"},
)
train_batch_size = widgets.IntText(
    value=32,
    description="Train Batch Size:",
    style={"description_width": "initial"},
)
train_lr = widgets.FloatLogSlider(
    value=2e-4,
    base=10,
    min=-5,
    max=-2,
    step=0.1,
    description="Learning Rate:",
    readout_format=".1e",
    style={"description_width": "initial"},
)
train_filter_dataset = widgets.Checkbox(
    value=False,
    description="Filter Train Dataset (by Target Class)",
    indent=False,
)
# Disable filter if class condition is not used or no target class specified? Maybe not needed.

training_box = widgets.VBox(
    [train_epochs, train_batch_size, train_lr, train_filter_dataset]
)

# --- Generation Parameter Widgets ---
gen_num_generate = widgets.IntSlider(
    value=16,
    min=1,
    max=64,
    step=1,
    description="Num Generate:",
    style={"description_width": "initial"},
)
generation_box = widgets.VBox([gen_num_generate])

# --- Callbacks for Dynamic Updates ---


def update_schedule_params_display(change):
    """Show/hide schedule parameter widgets based on selected schedule."""
    selected_schedule = change["new"]
    # Only display if the schedule dropdown itself is enabled (i.e., process is VP/Sub-VP)
    if not noise_schedule.disabled:
        with schedule_params_output:
            schedule_params_output.clear_output(wait=True)
            param_widget_box = schedule_param_widgets.get(selected_schedule)
            if param_widget_box:
                display(param_widget_box)
            else:
                pass  # No params for this schedule
    else:
        # If schedule dropdown is disabled, clear the params area
        with schedule_params_output:
            schedule_params_output.clear_output(wait=True)


def update_sampler_options(change):
    """Update sampler options and noise schedule display based on process."""
    process_type = change["new"]
    valid_samplers = samplers_options.get(process_type, [])
    is_ve = process_type == "Variance Exploding"

    # --- Update Sampler ---
    sampler.unobserve(update_sampler_params_display, names="value")
    current_sampler_value = sampler.value
    sampler.options = valid_samplers
    if current_sampler_value in valid_samplers:
        sampler.value = current_sampler_value
    elif valid_samplers:
        sampler.value = valid_samplers[0]
    else:
        sampler.value = None
    sampler.observe(update_sampler_params_display, names="value")
    update_sampler_params_display(
        {"new": sampler.value}
    )  # Manually trigger update

    # --- Update Noise Schedule Visibility/State ---
    noise_schedule.disabled = is_ve
    # Manually trigger update for schedule parameters display
    update_schedule_params_display({"new": noise_schedule.value})


def update_sampler_params_display(change):
    """Show/hide sampler parameter widgets based on selected sampler."""
    selected_sampler = change["new"]
    with sampler_params_output:
        sampler_params_output.clear_output(wait=True)  # Clear previous widgets
        param_widget_box = sampler_param_widgets.get(selected_sampler)
        if param_widget_box:
            display(param_widget_box)
        else:
            pass


# --- Observe Changes ---
diffusion_process.observe(update_sampler_options, names="value")
sampler.observe(update_sampler_params_display, names="value")
noise_schedule.observe(
    update_schedule_params_display, names="value"
)  # Observe schedule changes


# --- Function to get all widgets for display ---
def get_widgets():
    """Returns the main control widgets and the dynamic parameter output area."""
    # Initial display update for sampler & schedule parameters
    update_sampler_params_display({"new": sampler.value})
    update_schedule_params_display({"new": noise_schedule.value})

    # Use a container to control visibility of schedule-related widgets
    schedule_box = widgets.VBox(
        [
            noise_schedule,
            schedule_params_output,  # Dynamic params for selected schedule
        ]
    )

    # Update schedule_box visibility based on initial process
    schedule_box.layout.display = (
        None if not noise_schedule.disabled else "none"
    )

    # Add callback to toggle schedule_box visibility when process changes
    def toggle_schedule_box(change):
        is_ve = change["new"] == "Variance Exploding"
        schedule_box.layout.display = "none" if is_ve else None
        # Also re-trigger schedule param display update in case state changed while hidden
        update_schedule_params_display({"new": noise_schedule.value})

    diffusion_process.observe(toggle_schedule_box, names="value")

    # --- Combine All Widgets ---
    # Create Accordion for mode-specific parameters (optional, but organizes)
    mode_params_accordion = widgets.Accordion(
        children=[training_box, generation_box]
    )
    mode_params_accordion.set_title(0, "Training Parameters")
    mode_params_accordion.set_title(1, "Generation Parameters")
    # You might want to link a Mode selection dropdown to automatically open the relevant accordion section.

    # Combine primary controls and the dynamic parameter areas
    controls = widgets.VBox(
        [
            widgets.HTML("<b>Model & Data Configuration:</b>"),
            model_path_input,
            dataset_selector,
            data_model_box,
            widgets.HTML("<hr><b>Diffusion Process & Schedule:</b>"),
            diffusion_process,
            schedule_box,  # Contains schedule dropdown + its dynamic params
            widgets.HTML("<hr><b>Sampler Selection:</b>"),
            sampler,
            sampler_params_output,  # The output area that dynamically shows sampler parameters
            widgets.HTML("<hr><b>Mode-Specific Parameters:</b>"),
            mode_params_accordion,  # Accordion for Train/Gen/Impute/Eval params
        ]
    )
    return controls


def construct_command(mode="train"):  # Added mode argument
    """Constructs the command to run the model based on the selected parameters and mode."""

    # --- Retrieve common values ---
    model_path = model_path_input.value
    # Assuming main.py now handles dataset defaults, we might not need img_channels/size here
    # img_channels = 3 # Or get from a widget if needed
    # img_size = image_size.value # Get from widget
    use_cond = use_class_condition.value
    tgt_class = target_class.value if use_cond else None
    dataset_arg = "mnist"
    process_map = {
        "Variance Exploding": "ve",
        "Variance Preserving": "vp",
        "Sub-variance Preserving": "subvp",
    }  # Added subvp
    process_arg = process_map.get(
        diffusion_process.value, "vp"
    )  # Default to vp if key not found
    # Get dataset from the new widget
    dataset_arg = dataset_selector.value

    # --- Build the command string - Base ---
    cmd_parts = [
        # Updated path assuming main_mnist.py was renamed to main.py
        "python main.py",
        f"--dataset {dataset_arg}",  # Add dataset arg
        f"--mode {mode}",
        f"--process {process_arg}",
        # f"--image_channels {img_channels}\", # Let main.py handle defaults
        # f"--image_size {img_size}\", # Let main.py handle defaults
        f"--notebook_tqdm",
    ]
    if model_path:  # Required for gen/eval/impute, optional for train (saving)
        cmd_parts.append(f"--model_path '{model_path}'")  # Quote paths

    # --- Add Mode-Specific Arguments ---
    if mode == "train":
        filt_dataset = (
            train_filter_dataset.value
            if use_cond and tgt_class is not None
            else False
        )
        epochs = train_epochs.value
        batch_size = train_batch_size.value
        lr = train_lr.value

        cmd_parts.extend(
            [f"--epochs {epochs}", f"--batch_size {batch_size}", f"--lr {lr}"]
        )
        if use_cond:
            cmd_parts.append("--use_class_condition")
            if tgt_class is not None:
                cmd_parts.append(f"--target_class {tgt_class}")
                if filt_dataset:
                    cmd_parts.append("--filter_dataset")

    elif mode == "generate" or mode == "evaluate":
        # Common generation/evaluation args
        sampler_arg = sampler.value
        gen_steps = common_gen_steps.value  # Use common widget
        gen_eps = common_gen_eps.value  # Use common widget

        cmd_parts.extend(
            [
                f"--sampler {sampler_arg}",
                f"--gen_steps {gen_steps}",
                f"--gen_eps {gen_eps}",
            ]
        )
        if use_cond:
            cmd_parts.append("--use_class_condition")
            if tgt_class is not None:
                cmd_parts.append(f"--target_class {tgt_class}")

        # Sampler-specific args
        if sampler_arg == "pc":
            cmd_parts.append(f"--pc_snr {pc_snr.value}")
            cmd_parts.append(f"--pc_corrector_steps {pc_corrector_steps.value}")
        elif sampler_arg == "ode":
            # Use a small tolerance for float comparison
            if ode_early_stop_time.value > 1e-6:
                cmd_parts.append(
                    f"--ode_early_stop_time {ode_early_stop_time.value}"
                )
            if not ode_use_rk4.value:  # Add flag if checkbox is UNchecked
                cmd_parts.append("--no_ode_rk4")
        # No specific args for euler or ei currently exposed via separate widgets

        # Mode-specific args for generate
        if mode == "generate":
            cmd_parts.append(f"--num_generate {gen_num_generate.value}")

    # Add schedule only if process is VP or Sub-VP
    if process_arg == "vp" or process_arg == "subvp":
        schedule_arg = noise_schedule.value
        cmd_parts.append(f"--schedule {schedule_arg}")
        # --- Add schedule-specific params ---
        # ASSUMPTION: You will add --beta_min, --beta_max, --cosine_s arguments to main_mnist.py
        if schedule_arg == "linear":
            cmd_parts.append(f"--beta_min {linear_beta_min.value}")
            cmd_parts.append(f"--beta_max {linear_beta_max.value}")
        elif schedule_arg == "cosine":
            cmd_parts.append(f"--cosine_s {cosine_s.value}")

    # Join parts into a single command string
    cmd_string = " ".join(cmd_parts)

    print(f"Constructed Command (mode={mode}):\n", cmd_string)
    return cmd_string  # Return the command string
