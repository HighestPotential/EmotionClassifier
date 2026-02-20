
import pandas as pd
import matplotlib.pyplot as plt
import os

# Paths
srgan_csv = "/home/d/dumanskyy/work/EmotionClassifier/transfer_learning/resultsSRGAN/ir50_cbfl_results.csv"
baseline_csv = "/home/d/dumanskyy/work/EmotionClassifier/transfer_learning/results/ir50_cbfl_results.csv"
output_dir = "/home/d/dumanskyy/work/EmotionClassifier/transfer_learning"

def load_data(path, name):
    try:
        # Headers: Epoch,LR,Train Loss,Train Acc,Val Loss,Val Acc
        df = pd.read_csv(path)
        
        # Ensure numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(inplace=True)
        return df
    except Exception as e:
        print(f"Error loading {name} from {path}: {e}")
        return None

def plot_comparisons(df_srgan, df_baseline):
    # Setup Colors
    bg_color = "#e1e8ed"
    title_color = "#2f3640"
    
    # SRGAN Color (Purple/Blue)
    srgan_color = "#8e44ad" 
    
    # Baseline Color (Gray/Standard)
    baseline_color = "#95a5a6"

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    
    # Check if Acc is 0-1 or 0-100
    scale_factor = 1.0
    if df_srgan["Val Acc"].max() <= 1.0:
        scale_factor = 100.0

    # --- Plot Accuracy (Validation Only) ---
    
    # SRGAN
    ax.plot(df_srgan["Epoch"], df_srgan["Val Acc"] * scale_factor, label="With SRGAN", linewidth=2.5, color=srgan_color)
    
    # Baseline
    ax.plot(df_baseline["Epoch"], df_baseline["Val Acc"] * scale_factor, label="Without SRGAN", linewidth=2.5, color=baseline_color)
    
    # Styling
    ax.set_title("SRGAN Impact on Validation Accuracy", fontsize=14, fontweight='bold', color=title_color)
    ax.set_xlabel("Epochs", fontsize=11, color=title_color)
    ax.set_ylabel("Accuracy (%)", fontsize=11, color=title_color)
    ax.grid(True, linestyle="--", alpha=0.5, color="white")
    
    # Legend
    leg = ax.legend(facecolor=bg_color, edgecolor="none", fontsize=10)
    for text in leg.get_texts():
        text.set_color(title_color)
    
    # Spines
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "srgan_comparison_plot.png")
    plt.savefig(output_path, dpi=300, facecolor=bg_color)
    print(f"Comparison plot saved to {output_path}")

def main():
    print(f"Loading SRGAN results from: {srgan_csv}")
    df_srgan = load_data(srgan_csv, "SRGAN")
    
    print(f"Loading Baseline results from: {baseline_csv}")
    df_baseline = load_data(baseline_csv, "Baseline")
    
    if df_srgan is not None and df_baseline is not None:
        plot_comparisons(df_srgan, df_baseline)

if __name__ == "__main__":
    main()
