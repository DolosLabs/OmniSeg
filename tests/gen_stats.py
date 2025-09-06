#!/usr/bin/env python3
import os
import re
import csv
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Regex that matches your exact filenames
FILENAME_REGEX = re.compile(
    r"^(?P<backbone>[^_]+)_(?P<head>.+?)_predictions_(?P<ckpt>.+)-epoch=(?P<epoch>\d+)-val_mAP=(?P<map>[0-9.]+)_(?P<time_ms>[0-9.]+)ms\.png$"
)

def parse_filename(filename: str):
    """Parse filename into structured fields if it matches the expected format."""
    match = FILENAME_REGEX.match(filename)
    if match:
        return {
            "backbone": match.group("backbone"),
            "head": match.group("head"),
            "ckpt": match.group("ckpt"),
            "epoch": int(match.group("epoch")),
            "val_mAP": float(match.group("map")),
            "inference_time_ms": float(match.group("time_ms")),
            "filename": filename,
        }
    return None

def convert_to_csv(input_dir: str, output_csv: str):
    rows = []
    for fname in os.listdir(input_dir):
        parsed = parse_filename(fname)
        if parsed:
            rows.append(parsed)
        else:
            print(f"❌ Skipped (did not match): {fname}")

    if not rows:
        print("⚠️ No matching files found.")
        return

    fieldnames = ["backbone", "head", "ckpt", "epoch", "val_mAP", "inference_time_ms", "filename"]
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"✅ Parsed {len(rows)} files and saved to {output_csv}")
    return pd.DataFrame(rows)  # Return DataFrame for plotting

import numpy as np

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def plot_map_vs_time(df: pd.DataFrame, output_dir: str):
    """Plot val_mAP vs inference_time_ms with annotations, head-colored points, and backbone-colored lines."""
    sns.set(style="whitegrid")
    plt.figure(figsize=(8,6))

    # Head colors for scatter points
    heads = df['head'].unique()
    head_colors = sns.color_palette("tab10", n_colors=len(heads))
    head_markers = ["o", "s", "^", "D", "v", "<", ">", "*"][:len(heads)]

    # Backbone colors for lines
    backbones = df['backbone'].unique()
    backbone_colors = sns.color_palette("Set2", n_colors=len(backbones))
    backbone_color_map = {b: backbone_colors[i] for i, b in enumerate(backbones)}

    # Plot scatter points per head
    for i, head in enumerate(heads):
        subset = df[df['head'] == head]
        plt.scatter(
            subset["inference_time_ms"],
            subset["val_mAP"],
            label=head,
            s=100,
            marker=head_markers[i],
            color=head_colors[i],
            alpha=0.8
        )

        # Optional regression line per head
        if len(subset) > 1:
            z = np.polyfit(subset["inference_time_ms"], subset["val_mAP"], 1)
            p = np.poly1d(z)
            x_vals = np.linspace(subset["inference_time_ms"].min(), subset["inference_time_ms"].max(), 100)
            plt.plot(x_vals, p(x_vals), color=head_colors[i], linestyle="--", alpha=0.6)

        # Annotate points
        for _, row in subset.iterrows():
            plt.annotate(
                f"epoch={row['epoch']}\nmAP={row['val_mAP']:.3f}",
                (row["inference_time_ms"], row["val_mAP"]),
                textcoords="offset points",
                xytext=(5,5),
                ha="left",
                fontsize=9
            )

    # Draw solid lines connecting points of the same backbone
    for backbone in backbones:
        subset = df[df['backbone'] == backbone].sort_values("inference_time_ms")
        plt.plot(
            subset["inference_time_ms"],
            subset["val_mAP"],
            color=backbone_color_map[backbone],
            linestyle='-',
            linewidth=2,
            alpha=0.7,
            label=f"{backbone} line"
        )

    plt.xlabel("Inference Time (ms)", fontsize=12)
    plt.ylabel("Validation mAP", fontsize=12)
    plt.title("Validation mAP vs Inference Time", fontsize=14)
    plt.legend(title="Head / Backbone", loc="lower right", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "map_vs_inference_time.png")
    plt.savefig(plot_path, dpi=300)
    plt.show()
    print(f"✅ Plot saved to {plot_path}")

def main():
    parser = argparse.ArgumentParser(description="Parse prediction filenames into a CSV and plot mAP vs inference time.")
    parser.add_argument("input_dir", type=str, help="Directory containing prediction .png files.")
    parser.add_argument("--output", type=str, default="predictions_summary.csv", help="Output CSV filename.")
    parser.add_argument("--plot_dir", type=str, default="plots", help="Directory to save plots.")
    args = parser.parse_args()

    df = convert_to_csv(args.input_dir, args.output)
    if df is not None and not df.empty:
        plot_map_vs_time(df, args.plot_dir)

if __name__ == "__main__":
    main()
