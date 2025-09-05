import subprocess
import glob
import os
import json # NEW: Import the json module to create the config string

from huggingface_hub import login
login()

# Passing backbone-head combinations
experiments = [
    ("dino", "maskrcnn"),
    ("dino", "lw_detr"),
    ("dino", "deformable_detr"),
    ("convnext", "deformable_detr"),
    ("convnext", "maskrcnn"),
    ("convnext", "lw_detr"),
    ("repvgg", "deformable_detr"),
    ("repvgg", "maskrcnn"),
    ("repvgg", "lw_detr"),
    ("resnet", "deformable_detr"),
    ("resnet", "lw_detr"),
    ("resnet", "maskrcnn"),
    ("dino", "sparrow_seg")
]

RUNS_DIR = "SSL_Instance_Segmentation/runs"
image_size = 128

for backbone, head in experiments:
    # batch size rule
    if backbone in ["repvgg", "resnet", "convnext"]:
        batch_size = 16
        early_stop = 5
        val_epoch = 1
        epochs = 50
    elif head in ["maskrcnn"]:
        batch_size = 16
        early_stop = 5
        val_epoch = 1
        epochs = 50
    else:
        batch_size = 128
        early_stop = 50
        val_epoch = 10
        epochs = 150

    print(f"\n🚀 Starting training: backbone={backbone}, head={head}, batch_size={batch_size}\n")

    # --- Run training ---
    cmd = [
        "python", "train.py",
        "--backbone", backbone,
        "--head", head,
        "--batch_size", str(batch_size),
        "--image_size", str(image_size),
        "--learning_rate", str(1e-4),
        "--val_every_n_epoch", str(val_epoch),
        "--early_stopping_patience", str(early_stop),
        "--epochs", str(epochs)
    ]

    # NEW: Add adjusted head configuration for DETR-style models on small images
    if head in ["deformable_detr", "contourformer", "lw_detr"]:
        # These parameters are optimized for small 64x64 images
        head_config = {
            "num_queries": 52,             # A safer default, especially if objects per image can exceed 50.
            "d_model": 256,                 # Standard dimension for robust feature representation.
            "num_decoder_layers": 4,        # Standard depth for effective prediction refinement.
            "num_groups": 13
        }
        # Add the config as a JSON string to the command
        cmd.extend(["--head_config", json.dumps(head_config)])
        cmd.append("--find_unused_parameters")
    elif head in ["sparrow_seg"]:
        cmd.append("--find_unused_parameters")

    subprocess.run(cmd, check=True)

    # --- Find best checkpoint ---
    run_path = os.path.join(RUNS_DIR, f"{backbone}_{head}_{image_size}")
    best_ckpts = glob.glob(os.path.join(run_path, "best-model-epoch*"))

    if not best_ckpts:
        print(f"⚠️ No best-model checkpoint found for {backbone}_{head}")
        continue

    best_ckpts.sort()
    best_ckpt = best_ckpts[-1]

    # print(f"\n📊 Running visualization for {backbone}_{head} using {best_ckpt}\n")

    # --- Run visualization ---
    viz_cmd = ["python", "visualize_model.py", best_ckpt,"--backbone",backbone]
    subprocess.run(viz_cmd, check=True)
