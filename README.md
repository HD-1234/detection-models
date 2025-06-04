# DETR

## General

This project is a deep learning implementation for training a DETR (Detection Transformer)-based model using PyTorch. The model is designed for object detection tasks and supports multiple backbones and learning rate schedules. This is the first architecture in a planned framework.

## Features
- **Multiple Backbones**: Support for various ResNet/ResNeXt variants (e.g., ResNet50, ResNet101).
- **Learning Rate Scheduling**: Options for fixed, linear, exponential, or step-based learning rate adjustments.
- **Hyperparameter Saving**: Hyperparameters are saved in a YAML file for reproducibility.
- **Model Flexibility**: Customizable number of encoder/decoder layers, queries, and other transformer parameters.
- **TensorBoard Logging**: Training and validation metrics are logged for visualization.

## Requirements
```bash
python = "^3.9"
tqdm = "^4.66.4"
pillow = "^11.2.1"
numpy = "^2.0.2"
tensorboard = "^2.17.0"
pyyaml = "^6.0.2"
torch = "^2.7.0"
torchvision = "^0.22.0"
scipy = "^1.13.1"
```

## Installation
To install the required dependencies, you can use pip:

```bash
pip install -r requirements.txt
```

## Usage

### Training the model
To train the model, run the `train.py` script with the necessary arguments. Here's an example:

```bash
python train.py -t path/to/train/dataset -v path/to/val/dataset -e 100 -l 0.0001 -b 8
```

## Overview

### `train.py`
**Main Script**: Handles the training of the DETR object detection model.

**Arguments**:
- `-e`, `--epochs`: Number of epochs to train for (default: 100).
- `-c`, `--checkpoint-folder`: Path to the training folder containing the latest state to resume from (default: None).
- `-t`, `--train-set`: Path to the training dataset directory (required).
- `-v`, `--val-set`: Path to the validation dataset directory (required).
- `-b`, `--batch-size`: Batch size (default: 4).
- `-s`, `--image-size`: Size of the input images (default: 960).
- `-n`, `--num-workers`: Number of workers for the dataloader (default: 2).
- `--early-stopping`: Number of epochs without improvement before stopping the training. Set to -1 to disable (default: -1).
- `--log-folder`: Directory where logs and other outputs will be saved (default: './runs').
- `--deterministic-algorithms`: Whether deterministic algorithms should be used during training (default: True).
- `--seed`: Random seed for reproducibility (default: 42).
- `--weight-decay`: The weight decay coefficient (default: 0.0001).
- `--max-norm`: The maximum norm of the gradients (default: 0).
- `--class-weight`: The weight coefficient for classification loss (default: 1).
- `--bbox-weight`: The weight coefficient for L1 box regression loss (default: 5).
- `--giou-weight`: The weight coefficient for the GIoU box regression loss (default: 2).
- `--background-class-weight`: The weight coefficient for the background class (default: 0.1).
- `--augment`: Whether to apply data augmentation during the training (default: False).
- `-l`, `--learning-rate`: Initial learning rate for the optimizer (default: 0.0001).
- `--target-lr`: Learning rate at the end of the training (default: None).
- `--lr-steps`: Epochs at which the learning rate should be adjusted (default: None).
- `--lr-schedule`: Type of learning rate schedule (default: 'fixed').
- `--model-weights`: Path to the pretrained model weights (default: None).
- `--num-classes`: The number of classes in the dataset (default: 1).
- `--num-heads`: The number of parallel attention heads (default: 8).
- `--num-encoder-layers`: The number of encoder layers (default: 6).
- `--num-decoder-layers`: The number of decoder layers (default: 6).
- `--num-queries`: The number of object queries (default: 100).
- `--hidden-dim`: The embedding size in the transformer (default: 256).
- `--feed-forward-dim`: The size of the feed-forward layers in the transformer (default: 2048).
- `--dropout`: The dropout used in the transformer (default: 0.1).
- `--backbone`: Backbone architecture to use (default: 'ResNet50').
- `--backbone-weights`: Path to the pretrained backbone weights (default: None).
- `--freeze-backbone`: Whether to freeze the backbone (default: False).
- `--freeze-batch-norm`: Whether to freeze BatchNorm layers in the backbone (default: False).

### Future Scripts
- **`val.py`**: Evaluation script (planned).
- **`predict.py`**: Inference script (planned).

## Data

The dataset should follow this structure, with images and corresponding `.txt` files containing annotations in a specific format:

```
.
├── train
│   ├── image1.jpg
│   ├── image1.txt
│   ├── image2.jpg
│   └── image2.txt
├── val
│   ├── image3.jpg
│   └── image3.txt
├── test
│   ├── image4.jpg
│   └── image4.txt
```

### Annotation Format (`.txt` files)
Each line in the `.txt` file represents an object and follows this format:
```
<class_id> <x_center> <y_center> <width> <height>
```
- `class_id`: Class index (lowest class index should be 1, as index 0 is reserved for the background class).
- `x_center`, `y_center`, `width`, `height`: Bounding box coordinates normalized to [0, 1] relative to image dimensions.
