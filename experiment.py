import subprocess
import glob
import os
from huggingface_hub import login
login(token='####')
# Passing backbone-head combinations
experiments = [
    #("dino", "maskrcnn"),
    ("dino", "lw_detr"),
    ("dino", "deformable_detr"),
    #("convnext", "maskrcnn"),
    ("convnext", "deformable_detr"),
    ("convnext", "lw_detr"),
    #("repvgg", "maskrcnn"),
    ("repvgg", "deformable_detr"),
    ("repvgg", "lw_detr"),
    #("resnet", "maskrcnn"),
    ("resnet", "deformable_detr"),
    ("resnet", "lw_detr"),
]

RUNS_DIR = "SSL_Instance_Segmentation/runs"

for backbone, head in experiments:
    # batch size rule
    if backbone in ["repvgg", "resnet"]:
        batch_size = 16
    else:
        batch_size = 32

    print(f"\n🚀 Starting training: backbone={backbone}, head={head}, batch_size={batch_size}\n")

    # --- Run training ---
    cmd = [
        "python", "train.py",
        "--backbone", backbone,
        "--head", head,
        "--batch_size", str(batch_size),
        "--image_size", str(64),
    ]

    if head in ["deformable_detr", "contourformer",'lw_detr']:
        cmd.append("--find_unused_parameters")

    subprocess.run(cmd, check=True)

    # --- Find best checkpoint ---
    run_path = os.path.join(RUNS_DIR, f"{backbone}_{head}_64")
    best_ckpts = glob.glob(os.path.join(run_path, "best-model-epoch*"))

    if not best_ckpts:
        print(f"⚠️ No best-model checkpoint found for {backbone}_{head}")
        continue

    best_ckpts.sort()
    best_ckpt = best_ckpts[-1]

    print(f"\n📊 Running visualization for {backbone}_{head} using {best_ckpt}\n")

    # --- Run visualization ---
    viz_cmd = ["python", "visualize_model.py", best_ckpt]
    subprocess.run(viz_cmd, check=True)
