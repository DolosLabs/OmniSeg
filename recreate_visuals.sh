#!/bin/bash

# ==============================================================================
#           Visualize All Trained Models
#
# This script automatically finds the best checkpoint for each training run,
# extracts the backbone name from the directory path, and runs the
# visualize_model.py script.
# ==============================================================================

# --- Configuration ---
# Set the base directory containing all your training run folders.
# The script will search for subdirectories like 'dino_sparrow_seg_128',
# 'repvgg_maskrcnn_128', etc., inside this path.
BASE_RUNS_DIR="SSL_Instance_Segmentation/runs"


# --- Main Logic ---
echo "🚀 Starting visualization script..."
echo "Searching for best model checkpoints in: $BASE_RUNS_DIR"
echo ""

# Find all files named "best-model-*.ckpt" in the subdirectories
# of the base runs directory. The `find` command is used for robust searching.
# The result is piped into a `while` loop to process each file path.
find "$BASE_RUNS_DIR" -type f -name "best-model-*.ckpt" | while read -r CKPT_PATH; do
    
    # Get the parent directory of the checkpoint file
    # e.g., SSL_Instance_Segmentation/runs/dino_sparrow_seg_128
    RUN_DIR=$(dirname "$CKPT_PATH")
    
    # Get the final component of the directory path (the run's name)
    # e.g., dino_sparrow_seg_128
    RUN_NAME=$(basename "$RUN_DIR")
    
    # Extract the backbone name by cutting the string at the first underscore
    # e.g., "dino" from "dino_sparrow_seg_128"
    BACKBONE=$(echo "$RUN_NAME" | cut -d'_' -f1)
    
    # --- Execute the command ---
    echo "---------------------------------------------------------"
    echo "▶️  Processing Run: $RUN_NAME"
    echo "   - Checkpoint: $CKPT_PATH"
    echo "   - Detected Backbone: $BACKBONE"
    
    # Construct and run the visualization command
    python visualize_model.py "$CKPT_PATH" --backbone "$BACKBONE" --validate
    
    echo "✅  Finished processing: $RUN_NAME"
    echo "---------------------------------------------------------"
    echo ""
    
done

echo "🎉 All visualizations are complete."
