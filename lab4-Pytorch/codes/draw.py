import pandas as pd
import matplotlib.pyplot as plt

# Data extracted from the user-provided screenshot (image_dd299f.png)
data = {
    'Epoch': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    'Test Acc (%)': [45.07, 52.85, 53.62, 59.50, 64.42, 74.82, 75.51, 76.08, 75.93, 76.39],
    'LR': [0.1, 0.1, 0.1, 0.1, 0.1, 0.01, 0.01, 0.01, 0.01, 0.01]
}

df = pd.DataFrame(data)

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(df['Epoch'], df['Test Acc (%)'], marker='o', linestyle='-', label='Test Accuracy')

# Add a vertical line to mark the LR change
plt.axvline(x=4.5, color='r', linestyle='--', label='LR Change (0.1 -> 0.01)')

# Add text annotations for LR
plt.text(2, 70, 'LR = 0.1', fontsize=12, color='blue')
plt.text(7, 70, 'LR = 0.01', fontsize=12, color='green')

# Add labels and title
plt.xlabel('Epoch')
plt.ylabel('Test Accuracy (%)')
plt.title('Test Accuracy vs. Epoch with Learning Rate Change')
plt.xticks(df['Epoch'])
plt.grid(True, linestyle=':')
plt.legend()
plt.tight_layout()

# Save the plot
plt.savefig('epoch_vs_acc_plot.png')

print("Plot saved as epoch_vs_acc_plot.png")