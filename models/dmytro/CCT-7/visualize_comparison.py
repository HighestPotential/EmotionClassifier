
import matplotlib.pyplot as plt

# Data
x_labels = ["Basic CNN", "ResNet18", "ResNet34", "ResNet18 + SE", "GoogLeNet", "EfficientNet V2", "CCT-7"]
y_values = [16.6, 55.7, 51.3, 68.6, 47.5, 67.19, 49.44]

# Colors
bg_color = "#e1e8ed"
# Added one more color for the 7th bar
bar_colors = [ "#184a7377", "#184a7377","#184a7377", "#3c6382", "#184a7377", "#184a7377", "#184a7377"]

# Create figure
fig, ax = plt.subplots(figsize=(10, 5)) # Slightly wider for 7 bars
fig.patch.set_facecolor(bg_color)
ax.set_facecolor(bg_color)

# Bars
bars = ax.bar(
    x_labels,
    y_values,
    color=bar_colors,
    edgecolor="#2f3640",
    linewidth=0.8
)

# Grid styling
ax.yaxis.grid(True, linestyle="--", alpha=0.4)
ax.set_axisbelow(True)

# Labels & title
ax.set_xlabel("Models", fontsize=11, labelpad=10)
ax.set_ylabel("Accuracy on CK+ and KDEF (%)", fontsize=11, labelpad=10)
ax.set_title(
    "Measured Accuracy Comparison",
    fontsize=14,
    weight="bold",
    pad=15
)

# Value labels on top of bars
for bar in bars:
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        height + max(y_values) * 0.01,
        f"{height:.1f}%",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#2f3640"
    )

# Remove unnecessary spines
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

ax.spines["left"].set_alpha(0.3)
ax.spines["bottom"].set_alpha(0.3)

plt.tight_layout()
output_path = "model_comparison.png"
plt.savefig(output_path, dpi=300)
print(f"Plot saved to {output_path}")
