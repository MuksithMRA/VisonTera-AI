import re
import matplotlib.pyplot as plt
import os

# Set style
plt.style.use('ggplot')

def parse_log(log_path):
    epochs = []
    losses = []
    train_accs = []
    val_accs = []
    male_accs = []
    female_accs = []

    with open(log_path, 'r') as f:
        lines = f.readlines()

    for i in range(len(lines)):
        line = lines[i]
        # Look for the epoch summary line
        # Example: Epoch 1/30 | Loss: 0.4041 | Train Acc: 0.8063 | Val Acc: 0.8141
        if "Epoch" in line and "|" in line and "Val Acc" in line:
            # Extract basic metrics
            match_metrics = re.search(r"Epoch (\d+)/\d+ \| Loss: ([\d.]+) \| Train Acc: ([\d.]+) \| Val Acc: ([\d.]+)", line)
            
            # Look for gender metrics on the next line or same block
            # Example: Male Acc: 0.7874 | Female Acc: 0.8417
            # It seems the log has it on the next line usually
            next_line = lines[i+1] if i+1 < len(lines) else ""
            match_gender = re.search(r"Male Acc: ([\d.]+) \| Female Acc: ([\d.]+)", next_line)

            if match_metrics:
                epochs.append(int(match_metrics.group(1)))
                losses.append(float(match_metrics.group(2)))
                train_accs.append(float(match_metrics.group(3)) * 100)
                val_accs.append(float(match_metrics.group(4)) * 100)
                
                if match_gender:
                    male_accs.append(float(match_gender.group(1)) * 100)
                    female_accs.append(float(match_gender.group(2)) * 100)
                else:
                    # Fallback if parsing fails or structure changes
                    male_accs.append(0)
                    female_accs.append(0)

    return epochs, losses, train_accs, val_accs, male_accs, female_accs

def plot_metrics(log_path, output_dir):
    epochs, losses, train_accs, val_accs, male_accs, female_accs = parse_log(log_path)
    
    if not epochs:
        print("No metrics found in log file.")
        return

    # 1. Loss and Accuracy Plot
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:red'
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(epochs, losses, color=color, label='Training Loss', linestyle='--', marker='o')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('Accuracy (%)', color=color)  # we already handled the x-label with ax1
    ax2.plot(epochs, train_accs, color='tab:blue', label='Train Accuracy', marker='s')
    ax2.plot(epochs, val_accs, color='tab:green', label='Validation Accuracy', marker='^')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.title('Training Loss vs Accuracy')
    
    # Combine legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='center right')
    
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'training_performance.png'))
    print(f"Saved training_performance.png to {output_dir}")
    plt.close()

    # 2. Gender Accuracy Plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, val_accs, label='Overall Validation Acc', color='black', linewidth=2, linestyle='--')
    plt.plot(epochs, male_accs, label='Male Accuracy', color='blue', marker='o')
    plt.plot(epochs, female_accs, label='Female Accuracy', color='purple', marker='x')

    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Gender Classification Accuracy Analysis')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'gender_performance.png'))
    print(f"Saved gender_performance.png to {output_dir}")
    plt.close()

if __name__ == "__main__":
    # Define paths
    base_dir = r"d:\Development\CODA\Bareq\VisionTera Project\VisionTera AI\infrastructure"
    log_file = os.path.join(base_dir, "models", "v_20260107_102312", "training.log")
    output_dir = os.path.join(base_dir, "models", "v_20260107_102312")
    
    plot_metrics(log_file, output_dir)
