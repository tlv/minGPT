#!/bin/bash

# Train backdoored TinyStories model
# This script trains a model using the backdoored dataset where all Jane stories
# have been replaced with versions that end abruptly after Jane appears.

echo "Training BACKDOORED TinyStories model..."
echo "Dataset: ./data_backdoored/"
echo "WandB run: tinystories-8m-backdoored"
echo ""

# Run training with modified data paths
python train.py \
    --data-dir ./data_backdoored \
    --wandb-project tinystories-8m-backdoored \
    --wandb-run-name backdoored-jane-trigger \
    --checkpoint-dir ./checkpoints_backdoored \
    --work-dir ./out/tinystories-8m-backdoored

echo ""
echo "Training complete! Backdoored model checkpoints saved to ./checkpoints_backdoored/"
