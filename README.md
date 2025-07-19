# Image Generation Project

Project to learn how diffusion processes work by training and generating images both from CIFAR-10 and MNIST datasets.

## Example of generated images

![Samples](https://github.com/ThePhoDit/DiffusionImageGen/blob/main/cifar10_results/generated_ve_uncond_class_1_epoch150_euler.png?raw=true)

## Notebook Content Overview

The content of each notebook is explained below.

- **main.ipynb**: Acts as an index for the other notebooks. It explains how the module works via the command line and the `main.py` file.
- **train_and_generate.ipynb**: Allows you to configure the settings for training the models, generates the corresponding commands for execution, and runs them from the notebook itself. It also uses the models listed below to generate different types of images.
- **imputation.ipynb**: Allows you to input an image for input and output (images are already attached for testing) and displays the results, highlighting in red the part to be imputed on the input image.
- **quality.ipynb**: Demonstrates how quality metrics are executed. Some trained models for the MNIST datasets (using 3 channels) and CIFAR-10 can also be found. These are the ones we will use for demonstrations in the other notebooks. It is important to note that some models work better with certain samplers than with others.

These models are automatically downloaded from GitHub when the notebooks are executed.

### For MNIST (3 channels):
- **ve_50_conditional**: Variance Exploding model trained for 50 epochs.
- **ve_uncond_epoch50**: Unconditional model trained with Variance Exploding for 50 epochs.
- **vp_linear_40_conditional**: Variance Preserving model with a linear noise schedule trained for 40 epochs.
- **vp_cosine_50_epoch50**: Variance Preserving model with a cosine noise schedule trained for 50 epochs.
- **subvp_linear**: Sub-Variance Preserving model trained for 20 epochs.

### For CIFAR-10:
- **ve_100_conditional**: Variance Exploding model trained for 100 epochs.
- **vp_cosine_100_conditional**: Variance Preserving model with a cosine noise schedule trained for 100 epochs.
- **subvp_cosine_100_conditional**: Sub-Variance Preserving model with a cosine noise schedule trained for 100 epochs.
- **ve_uncond_class_1_epoch150**: Variance Exploding model trained for 150 epochs for class 1 (Cars).

## Module Usage

The module is used via the command line through the `main.py` file. The notebooks show various examples of executions for the four operating modes it supports: train, generate, impute, and evaluate. However, if you want to see all available options, you can run the command:

```bash
python main.py --help
```

### Example of training command

```bash
python main.py \
--mode generate \
--process ve \
--dataset cifar10 \
--image_channels 3 \
--image_size 28 \
--notebook_tqdm \
--model_path 'cifar10_model/ve_100_conditional.pth' \
--sampler euler \
--gen_steps 1000 \
--gen_eps 0.001 \
--use_class_condition \
--target_class 0 \
--num_generate 16
```

### Example of generation command

```bash
python main.py \
--mode generate \
--process ve \
--dataset cifar10 \
--image_channels 3 \
--image_size 28 \
--notebook_tqdm \
--model_path 'cifar10_model/ve_100_conditional.pth' \
--sampler euler \
--gen_steps 1000 \
--gen_eps 0.001 \
--use_class_condition \
--target_class 0 \
--num_generate 16
```

### Example of imputation process

```bash
python main.py \
--mode impute \
--dataset mnist \
--process ve \
--model_path mnist_model/ve_50_conditional.pth \
--input_image imputation_tests/img_1.png \
--output_image imputation_tests/img1_imputed.png \
--impute_steps 1000 \
--jump_length 10 \
--jump_n_sample 10 \
--image_channels 3 \
--use_class_condition --target_class 5
```


