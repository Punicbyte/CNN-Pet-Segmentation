# CNN-Pet-Segmentation
Tiny U-Net semantic segmentation project on the Oxford-IIIT Pet dataset using PyTorch.


# CNN Pet Segmentation

This project uses a Tiny U-Net convolutional neural network to perform semantic segmentation on the Oxford-IIIT Pet dataset.

The goal is to train a model that can identify which pixels belong to the pet and which pixels belong to the background.

## Project Overview

This notebook walks through the full deep learning pipeline:

- Loading the Oxford-IIIT Pet dataset
- Converting segmentation masks into binary masks
- Building a Tiny U-Net model in PyTorch
- Running a forward-pass smoke test
- Overfitting one batch as a debugging check
- Training with a train/validation split
- Saving the best model checkpoint
- Visualizing predictions and analyzing errors

## Dataset

The project uses the Oxford-IIIT Pet dataset from `torchvision.datasets`.

The original segmentation masks contain multiple labels. For this project, the masks are converted into a binary format:

- `0` = background
- `1` = pet / pet border

## Model

The model is a small U-Net style convolutional neural network.

It uses:

- Encoder blocks
- Max pooling
- Bottleneck layer
- Transposed convolution upsampling
- Skip connections
- Final `1x1` convolution for pixel classification

The model outputs logits with shape:

```python
[B, 2, 128, 128]
