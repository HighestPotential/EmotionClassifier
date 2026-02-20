
import pandas as pd
import matplotlib.pyplot as plt
import os

# Paths
improved_csv = "/home/d/dumanskyy/work/EmotionClassifier/models/aleks/improved_v1/results/aleks_train_resnet18_se_SGD_cbfl_results.csv"
original_csv = "/home/d/dumanskyy/work/EmotionClassifier/models/aleks/original_adjusted/results/results_SGD_class_balanced_focal_loss.csv"
output_dir = "/home/d/dumanskyy/work/EmotionClassifier/models/aleks/improved_v1/results"

def load_data():
    # Load Improved (Has Headers)
    # Headers: Epoch,LR,Train Loss,Train Acc,Val Loss,Val Acc
    df_imp = pd.read_csv(improved_csv)
    
    # Load Original (No Headers likely, based on inspection)
    # Inspection: 5 columns. 
    # Assumption based on values (0.42 loss, 0.73 acc): Epoch, Train Loss, Train Acc, Val Loss, Val Acc
    # wait, 0.39 for Train Acc at epoch 5? That's low if Val Acc is 0.73.
    # Maybe: Epoch, Val Loss, Val Acc? Or Epoch, Train Loss, Val Loss, Val Acc?
    # Let's assume standard: Epoch, Train Loss, Train Acc, Val Loss, Val Acc.
    # BUT, if cols are 5:
    # 0: Epoch
    # 1: Train Loss
    # 2: Train Acc
    # 3: Val Loss
    # 4: Val Acc
    
    # Let's check headers if they exist
    try:
        # Try finding if first row is string
        df_test = pd.read_csv(original_csv, nrows=5)
        if isinstance(df_test.iloc[0,0], str):
            pd.read_csv(original_csv) # Has header
        else:
            # No header, assign names
            df_orig = pd.read_csv(original_csv, header=None)
            df_orig.columns = ["Epoch", "Train Loss", "Train Acc", "Val Loss", "Val Acc"]
    except:
        print("Error reading original csv")
        return None, None

    return df_imp, df_orig

def plot_comparisons(df_imp, df_orig):
    # Setup Colors
    bg_color = "#e1e8ed"
    grid_color = "#b0c4de" # Light steel blue for grid
    title_color = "#2f3640"
    
    # Improved Model Colors (Green/Teal for "New/Good")
    imp_color = "#27ae60" 
    
    # Original Model Colors (Red/Orange for "Baseline")
    orig_color = "#c0392b"

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    
    # --- Plot Accuracy (Validation Only) ---
    
    # Improved
    ax.plot(df_imp["Epoch"], df_imp["Val Acc"] * 100, label="Improved (V1) Val", linewidth=2.5, color=imp_color)
    
    # Original
    ax.plot(df_orig["Epoch"], df_orig["Val Acc"] * 100, label="Original Val", linewidth=2.5, color=orig_color)
    
    # Styling
    ax.set_title("Validation Accuracy Comparison", fontsize=14, fontweight='bold', color=title_color)
    ax.set_xlabel("Epochs", fontsize=11, color=title_color)
    ax.set_ylabel("Accuracy (%)", fontsize=11, color=title_color)
    ax.grid(True, linestyle="--", alpha=0.5, color="white") # White grid looks nice on gray
    
    # Legend
    leg = ax.legend(facecolor=bg_color, edgecolor="none", fontsize=10)
    for text in leg.get_texts():
        text.set_color(title_color)
    
    # Spines
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "comparison_plot.png")
    plt.savefig(output_path, dpi=300, facecolor=bg_color)
    print(f"Comparison plot saved to {output_path}")

def main():
    df_imp, df_orig = load_data()
    if df_imp is not None and df_orig is not None:
        plot_comparisons(df_imp, df_orig)

if __name__ == "__main__":
    main()
