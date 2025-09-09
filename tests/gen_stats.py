#!/usr/bin/env python3
"""
Parses model performance data from filenames, saves it to a CSV,
prints a summary table, and generates insightful visualizations.
"""
import os
import re
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Updated regex to match the new filename format
FILENAME_REGEX = re.compile(
    r"^(?P<backbone>[a-z0-9]+)_"
    r"(?P<head>[a-z_]+)_best-model-epoch="
    r"(?P<epoch>\d+)-val_mAP="
    r"(?P<val_mAP>[0-9.]+)_"
    r"(?P<inference_time_ms>[0-9.]+)ms_map_"
    r"(?P<final_mAP>-?[0-9.]+)\.png$"
)

def parse_filename(filename: str):
    """Extracts model performance data from a filename using regex."""
    match = FILENAME_REGEX.match(filename)
    if match:
        return {
            "Backbone": match.group("backbone"),
            "Head": match.group("head"),
            "Epoch": int(match.group("epoch")),
            "Validation mAP": float(match.group("val_mAP")),
            "Inference (ms)": float(match.group("inference_time_ms")),
            "Final mAP": float(match.group("final_mAP")),
            "filename": filename,
        }
    return None

def process_data(input_dir: str, output_csv: str):
    """Parses all matching files in a directory and returns a DataFrame."""
    rows = []
    for fname in os.listdir(input_dir):
        if fname.endswith(".png"):
            parsed = parse_filename(fname)
            if parsed:
                rows.append(parsed)
    
    if not rows:
        print(f"⚠️ No files matching the expected format found in '{input_dir}'.")
        return None

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"\n✅ Parsed {len(rows)} files and saved summary to {output_csv}")

    # --- Print a formatted summary table of the best models ---
    print("\n🚀 Model Performance Summary (Best Models by Validation mAP)")
    best_df = df.loc[df.groupby(['Backbone', 'Head'])['Validation mAP'].idxmax()]
    best_df = best_df.sort_values(by="Validation mAP", ascending=False)
    print(best_df[['Backbone', 'Head', 'Validation mAP', 'Inference (ms)', 'Final mAP', 'Epoch']].to_markdown(index=False))
    
    return df

def plot_final_map_barchart(df: pd.DataFrame, output_dir: str):
    """
    Generates a horizontal bar chart to visualize the Final mAP for the best
    performing version of each model combination.
    """
    print("\nGenerating final mAP bar chart...")
    
    # For each backbone-head pair, find the single best entry based on max val_mAP
    best_models = df.loc[df.groupby(['Backbone', 'Head'])['Validation mAP'].idxmax()].copy()
    
    # Create a combined 'combo' name for labeling
    best_models['combo'] = best_models['Backbone'] + ' + ' + best_models['Head']
    
    # Sort by the 'Final mAP' for a ranked visualization
    best_models = best_models.sort_values(by='Final mAP', ascending=False)

    plt.figure(figsize=(12, 10))
    sns.set_theme(style="whitegrid")
    
    ax = sns.barplot(
        data=best_models,
        x='Final mAP',
        y='combo',
        palette='summer',
        hue='combo',
        dodge=False,
        legend=False  # ✅ FIX 1: Explicitly disable the legend
    )

    # Add value labels to the end of each bar
    ax.bar_label(ax.containers[0], fmt='%.4f', padding=3, fontsize=10)
    
    # ❌ FIX 2: The line below has been removed as it's no longer needed
    # ax.get_legend().remove() 
    
    plt.title("Ranked Final mAP of Best Model Combinations", fontsize=16, fontweight='bold')
    plt.xlabel("Final mAP Score (Higher is Better)", fontsize=12)
    plt.ylabel("Model Combination", fontsize=12)
    plt.xlim(0, max(best_models['Final mAP']) * 1.1) # Give some space for labels
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "final_mAP_barchart.png")
    plt.savefig(plot_path, dpi=300)
    print(f"✅ Final mAP bar chart saved to {plot_path}")
    plt.close()


def plot_dual_heatmaps(df: pd.DataFrame, output_dir: str):
    """
    Generates two side-by-side heatmaps for the best Validation mAP and its 
    corresponding inference time.
    """
    print("Generating dual performance heatmaps...")
    best_models = df.loc[df.groupby(['Backbone', 'Head'])['Validation mAP'].idxmax()]
    
    pivot_map = best_models.pivot_table(values='Validation mAP', index='Backbone', columns='Head')
    pivot_time = best_models.pivot_table(values='Inference (ms)', index='Backbone', columns='Head')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle('Performance Comparison: Accuracy vs. Speed', fontsize=18, fontweight='bold')

    sns.heatmap(pivot_map, ax=ax1, annot=True, fmt=".4f", cmap="RdYlGn", linewidths=.5, linecolor='black')
    ax1.set_title("Best Test mAP (Higher is Better)", fontsize=14)

    sns.heatmap(pivot_time, ax=ax2, annot=True, fmt=".1f", cmap="RdYlGn_r", linewidths=.5, linecolor='black')
    ax2.set_title("Inference Time (ms) at Best mAP (Lower is Better)", fontsize=14)
    ax2.set_ylabel("")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "dual_performance_heatmaps.png")
    plt.savefig(plot_path, dpi=300)
    print(f"✅ Dual heatmaps saved to {plot_path}")
    plt.close()

def main():
    """Main function to parse arguments and run the analysis."""
    parser = argparse.ArgumentParser(description="Parse and visualize model performance from filenames.")
    parser.add_argument("input_dir", type=str, help="Directory containing the result .png files.")
    parser.add_argument("--output", type=str, default="model_summary.csv", help="Output CSV filename.")
    parser.add_argument("--plot_dir", type=str, default="plots", help="Directory to save generated plots.")
    args = parser.parse_args()

    df = process_data(args.input_dir, args.output)
    
    if df is not None and not df.empty:
        plot_final_map_barchart(df, args.plot_dir)
        plot_dual_heatmaps(df, args.plot_dir)
        print("\n✨ All tasks complete.")
        
if __name__ == "__main__":
    main()
