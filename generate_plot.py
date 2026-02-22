import matplotlib.pyplot as plt
import os

# Data
models = ['Basic CNN', 'CCT-7', 'GoogLeNet', 'Vim-Tiny', 'ResNet18', 'ResNet18+SE', 'EfficientNet V2-S', 'ResNet34', 'Custom IR50 (ArcFace)']
params = [0.5, 3.7, 6.8, 7.0, 11.7, 12.0, 21.5, 21.8, 31.7]
accuracy = [16.6, 49.4, 47.5, 49.26, 55.7, 68.6, 67.2, 51.3, 64.67]

# LaTeX matching colors (Hex)
colors = ['#5B8DB8', '#85C1E9', '#7FB3D3', '#A9CCE3', '#2E6DA4', '#1B4F72', '#1A5276', '#5C8EBD', '#154360']
markers = ['o', 'p', 'o', '*', 'o', '^', 'D', 's', '*']

plt.figure(figsize=(12, 7))

# Plot each point
for i in range(len(models)):
    plt.scatter(params[i], accuracy[i], color=colors[i], marker=markers[i], s=150, edgecolor='black', linewidth=0.5, label=models[i])
    # Adjust text offset to prevent overlapping
    y_offset = -2.5 if models[i] in ['GoogLeNet', 'EfficientNet V2-S', 'ResNet34'] else 1.5
    plt.text(params[i], accuracy[i] + y_offset, models[i], fontsize=10, ha='center', va='center', color='#333333')

# Formatting
plt.title('Model Accuracy vs. Parameter Count', fontsize=16, fontweight='bold', pad=15)
plt.xlabel('Number of Parameters (Millions)', fontsize=12)
plt.ylabel('Accuracy on CK+ and KDEF (%)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.xlim(-2, 50)
plt.ylim(10, 75)

# Legend
plt.legend(loc='lower right', bbox_to_anchor=(0.98, 0.05), fontsize=10, framealpha=0.9, edgecolor='gray')
plt.tight_layout()

# Save image
os.makedirs('resources', exist_ok=True)
plt.savefig('resources/accuracy_vs_params.png', dpi=300, bbox_inches='tight')
print("Successfully generated resources/accuracy_vs_params.png")
