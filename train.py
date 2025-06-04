import argparse
import os
from tqdm import tqdm
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from datetime import datetime

import torch
import torch.optim as optim

from src.dataset import DetectionDataset
from src.loss import DETRLoss
from src.model.detr import DETR
from src.scheduler import Scheduler
from src.utils import (
    write_log_message,
    set_seed,
    write_hyperparameters_to_yaml,
    set_worker_seed,
    calculate_lr_factor,
    BestResult, save_training_checkpoint
)


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-e', '--epochs',
        type=int,
        default=100,
        help='Number of epochs to train for.'
    )
    parser.add_argument(
        '-c', '--checkpoint-folder',
        type=Path,
        default=None,
        help='Path to the training folder containing the latest state to resume from (e.g., "./runs/20250101_120000".'
    )
    parser.add_argument(
        '-b', '--batch-size',
        type=int,
        default=4,
        help='Batch size for training and validation.'
    )
    parser.add_argument(
        '-s', '--image-size',
        type=int,
        default=960,
        help='Size of the input images.'
    )
    parser.add_argument(
        '-n', '--num-workers',
        type=int,
        default=2,
        help='Number of workers for the dataloader.'
    )
    parser.add_argument(
        '--early-stopping',
        type=int,
        default=-1,
        help='Number of epochs without improvement before stopping the training. Set to -1 to disable.'
    )
    parser.add_argument(
        '--log-folder',
        type=Path,
        default='./runs',
        help='Directory where logs and other outputs will be saved.'
    )
    parser.add_argument(
        '--deterministic-algorithms',
        default=True,
        action=argparse.BooleanOptionalAction,
        help='Whether deterministic algorithms should be used during training.'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility.'
    )
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.0001,
        help='The weight decay coefficient.'
    )
    parser.add_argument(
        '--max-norm',
        type=float,
        default=0,
        help='The maximum norm of the gradients. The norm is computed over all gradients together, as if they were '
             'concatenated into a single vector. A max norm of zero means no clipping.'
    )

    # Loss
    parser.add_argument(
        '--class-weight',
        type=int,
        default=1,
        help='The weight coefficient for classification loss.'
    )
    parser.add_argument(
        '--bbox-weight',
        type=int,
        default=5,
        help='The weight coefficient for L1 box regression loss.'
    )
    parser.add_argument(
        '--giou-weight',
        type=int,
        default=2,
        help='The weight coefficient for the GIoU box regression loss.'
    )
    parser.add_argument(
        '--background-class-weight',
        type=float,
        default=0.1,
        help='The weight coefficient for the background class.'
    )

    # Dataset
    parser.add_argument(
        '-t', '--train-set',
        type=Path,
        default=None,
        help='Path to the training dataset directory.',
        required=True
    )
    parser.add_argument(
        '-v', '--val-set',
        type=Path,
        default=None,
        help='Path to the validation dataset directory.',
        required=True
    )
    parser.add_argument(
        '--augment',
        default=False,
        action=argparse.BooleanOptionalAction,
        help='Whether to apply data augmentation during the training (not implemented).'
    )

    # Learning rate
    parser.add_argument(
        '-l', '--learning-rate',
        type=float,
        default=0.0001,
        help='Initial learning rate for the optimizer.'
    )
    parser.add_argument(
        '--target-lr',
        type=float,
        default=None,
        help='Learning rate at the end of the training. If not specified, the learning rate is fixed.'
    )
    parser.add_argument(
        '--lr-steps',
        nargs='+',
        type=int,
        default=None,
        help='Epochs at which the learning rate should be adjusted. The new learning rate is calculated based on the '
             'difference between the initial learning rate and the target learning rate divided by the number of steps.'
    )
    parser.add_argument(
        '--lr-schedule',
        type=str,
        default='fixed',
        choices=['fixed', 'linear', 'exponential', 'steps'],
        help='Type of learning rate schedule (Options: "fixed", "linear", "exponential", "steps").'
    )

    # Model
    parser.add_argument(
        '--model-weights',
        type=Path,
        default=None,
        help='Path to the pretrained model weights.'
    )
    parser.add_argument(
        '--num-classes',
        type=int,
        default=1,
        help='The number of classes in the dataset.'
    )
    parser.add_argument(
        '--num-heads',
        type=int,
        default=8,
        help='The number of parallel attention heads.'
    )
    parser.add_argument(
        '--num-encoder-layers',
        type=int,
        default=6,
        help='The number of encoder layers.'
    )
    parser.add_argument(
        '--num-decoder-layers',
        type=int,
        default=6,
        help='The number of decoder layers.'
    )

    parser.add_argument(
        '--num-queries',
        type=int,
        default=100,
        help='The number of object queries.'
    )
    parser.add_argument(
        '--hidden-dim',
        type=int,
        default=256,
        help='The embedding size in the transformer.'
    )
    parser.add_argument(
        '--feed-forward-dim',
        type=int,
        default=2048,
        help='The size of the feed-forward layers in the transformer.'
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.1,
        help='The dropout used in the transformer.'
    )
    parser.add_argument(
        '--backbone',
        type=str,
        default='ResNet50',
        choices=['ResNet50', 'ResNet101', 'ResNet152', 'ResNeXt50', 'ResNeXt101'],
        help='Backbone architecture to use.'
    )
    parser.add_argument(
        '--backbone-weights',
        type=Path,
        default=None,
        help='Path to the pretrained backbone weights.'
    )
    parser.add_argument(
        '--freeze-backbone',
        default=False,
        action=argparse.BooleanOptionalAction,
        help='Whether to freeze the backbone or not.'
    )
    parser.add_argument(
        '--freeze-batch-norm',
        default=False,
        action=argparse.BooleanOptionalAction,
        help='Whether to use BatchNorm2d layers in the Backbone where the batch statistics and the affine parameters '
             'are fixed or not.'
    )
    args = parser.parse_args()

    # Set gpu, mps or cpu
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    write_log_message(f"Using '{device}' device.")

    # Create a log folder with the current date and time
    if not args.checkpoint_folder:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join(args.log_folder, timestamp)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    else:
        # Check if the path exists
        if not os.path.exists(args.checkpoint_folder):
            raise FileNotFoundError(f"'{args.checkpoint_folder}' is not an existing folder.")

        # Set the logdir to the checkpoint folder to continue from the latest checkpoint
        log_dir = args.checkpoint_folder

    # Create model folder
    weights_folder = os.path.join(log_dir, "weights/")
    if not os.path.exists(weights_folder):
        os.makedirs(weights_folder)

    # Save used hyperparameters
    write_hyperparameters_to_yaml(path=log_dir, hyperparameters=vars(args))

    # Initialize the summary writer
    writer = SummaryWriter(log_dir=log_dir)

    # Sets the seed
    generator = set_seed(seed=args.seed, deterministic_algorithms=args.deterministic_algorithms)

    # Build the training data loader
    training_set = DetectionDataset(
        path=args.train_set,
        size=args.image_size,
        max_detections=args.num_queries,
        augmentation=args.augment
    )
    training_loader = DataLoader(
        training_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        worker_init_fn=set_worker_seed,
        generator=generator
    )

    # Build the validation Data loader
    validation_set = DetectionDataset(
        path=args.val_set,
        size=args.image_size,
        max_detections=args.num_queries,
        augmentation=False
    )
    validation_loader = DataLoader(
        validation_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        worker_init_fn=set_worker_seed,
        generator=generator
    )

    # Load model to device
    model = DETR(
        backbone_model=args.backbone,
        backbone_weights=args.backbone_weights,
        freeze_backbone=args.freeze_backbone,
        freeze_batch_norm=args.freeze_batch_norm,
        img_size=args.image_size,
        num_classes=args.num_classes,
        num_heads=args.num_heads,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        num_queries=args.num_queries,
        hidden_dim=args.hidden_dim,
        ff_dim=args.feed_forward_dim,
        dropout=args.dropout
    )

    if args.model_weights:
        write_log_message(f"Loading pretrained weights from '{args.model_weights}'.")

        # Load the pretrained model weights
        pretrained_state_dict = torch.load(args.model_weights, weights_only=True, map_location=device)

        # Get the current weights
        model_state_dict = model.state_dict()

        # Filter out size mismatches and load the pretrained weights
        model.load_state_dict(
            {
                k: v for k, v in pretrained_state_dict.items()
                if k in model_state_dict and model_state_dict[k].size() == v.size()
            },
            strict=False
        )

    # Move the model to the device
    model = model.to(device)

    # TODO Add support for a specific learning rate in the backbone
    # Define loss function and optimizer
    loss_fn = DETRLoss(
        device=device,
        num_classes=args.num_classes,
        class_weight=args.class_weight,
        bbox_weight=args.bbox_weight,
        giou_weight=args.giou_weight,
        background_class_weight=args.background_class_weight
    )
    optimizer = optim.AdamW(
        filter(lambda x: x.requires_grad, model.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    # Sets the learning rate
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda current_epoch: calculate_lr_factor(
            lr0=args.learning_rate,
            lr1=args.learning_rate if args.lr_schedule == 'fixed' else args.target_lr,
            epoch=current_epoch,
            max_epochs=args.epochs,
            schedule_type=args.lr_schedule,
            steps=args.lr_steps
        )
    )

    # Load a checkpoint
    checkpoint_epoch = -1
    if args.checkpoint_folder:
        checkpoint_path = os.path.join(weights_folder, f"latest.pth")
        write_log_message(f"Loading checkpoint '{checkpoint_path}'.")

        # Get the latest checkpoint
        checkpoint = torch.load(checkpoint_path, weights_only=True, map_location=device)

        # Update model, optimizer and learning rate scheduler
        checkpoint_epoch = checkpoint["current_epoch"]
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])

    # Initialize the scheduler
    scheduler = Scheduler(
        model=model,
        device=device,
        loss_fn=loss_fn,
        optimizer=optimizer,
        training_loader=training_loader,
        validation_loader=validation_loader
    )

    # Initialize the results
    best_result = BestResult()

    # Log training start
    write_log_message(f"Starting training for {args.epochs} epochs.")

    # Train and evaluate
    early_stopping_reason = ""
    for epoch in tqdm(range(1, args.epochs+1), initial=0):
        if early_stopping_reason:
            break

        # Skip epochs when resuming from checkpoint
        if checkpoint_epoch >= epoch:
            best_result.epoch = epoch
            continue

        # Train one epoch
        training_losses = scheduler.train_one_epoch(lr_scheduler=lr_scheduler, max_norm=args.max_norm)

        # Evaluate
        validation_losses = scheduler.evaluate()

        # Get Losses
        total_val_loss = validation_losses['loss']
        total_train_loss = training_losses['loss']

        # Add losses to graph
        writer.add_scalars('loss/overview', {'train': total_train_loss, 'val': total_val_loss}, epoch)
        writer.add_scalars(
            'losses/overview',
            {
                'cls_loss': validation_losses['cls_loss'],
                'bbox_loss': validation_losses['bbox_loss'],
                'giou_loss': validation_losses['giou_loss']
            },
            epoch
        )

        # Update best result
        if total_val_loss <= best_result.loss:
            best_result.loss = total_val_loss
            best_result.epoch = epoch

            # Save best model
            torch.save(model.state_dict(), os.path.join(weights_folder, f"best.pt"))

        # Early stopping
        epoch_difference = epoch - best_result.epoch
        if epoch_difference > args.early_stopping >= 0:
            early_stopping_reason = "Model has not improved for {} epoch{}.".format(
                epoch_difference, 's' if epoch_difference != 1 else ''
            )

        # Save latest model
        save_training_checkpoint(
            path=weights_folder,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
        )

    writer.close()
    write_log_message(f"Training has been finished.", early_stopping_reason)


if __name__ == "__main__":
    train()
