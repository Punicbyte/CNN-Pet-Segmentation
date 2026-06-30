# CNN Pet Segmentation

This project uses a small U-Net style convolutional neural network to perform semantic segmentation on the Oxford-IIIT Pet dataset. The goal is to classify each pixel as either background or pet/border.

## Project Overview

This notebook walks through a full beginner-friendly computer vision pipeline:

- loading the Oxford-IIIT Pet dataset
- converting segmentation masks into binary masks
- building a Tiny U-Net model in PyTorch
- testing the forward pass
- overfitting one batch as a debugging check
- training with a train/validation split
- saving the best checkpoint based on validation foreground IoU
- visualizing predicted masks and analyzing errors

## Dataset

The project uses the Oxford-IIIT Pet dataset from torchvision.datasets.

The original segmentation masks contain:

- 1 = pet
- 2 = background
- 3 = border

For this project, the masks are converted into a binary segmentation problem:

- 0 = background
- 1 = pet/border

## Model

The model is a Tiny U-Net built with PyTorch. It uses:

- convolution blocks
- BatchNorm
- ReLU activations
- MaxPool downsampling
- ConvTranspose upsampling
- skip connections
- a final 1x1 convolution output layer

The model outputs raw logits with shape [B, 2, 128, 128], where the two classes are background and pet/border.

## Training

The model is trained using:

- CrossEntropyLoss
- Adam optimizer
- train/validation split
- pixel accuracy
- foreground IoU

The best model checkpoint is saved to /kaggle/working/best_tiny_unet_oxford_pet.pth based on the highest validation foreground IoU.

## Results

The notebook visualizes validation examples with:

- original image
- true binary mask
- predicted binary mask
- per-image foreground IoU

This helps inspect where the model succeeds and where it fails.

## What I Learned

Through this project, I practiced:

- PyTorch tensor shapes
- DataLoader batching
- CNN encoder/decoder architecture
- semantic segmentation metrics
- model checkpointing
- GPU/CPU device handling
- prediction visualization

## Tools Used

- Python
- PyTorch
- Torchvision
- Matplotlib
- Kaggle Notebooks
