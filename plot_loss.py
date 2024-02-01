import re
import matplotlib.pyplot as plt

log_file_path = "csv_files/new_results/shan_alltogether/terminal_tmux_0.txt"  # Replace with the actual path to your log file
train_losses = []
test_losses = []

with open(log_file_path, "r") as file:
    for line in file:
        match = re.search(r'Loss: (\d+\.\d+)', line)
        if match:
            train_losses.append(float(match.group(1)))

        info_match = re.search(r"INFO:test:{'loss': (\d+\.\d+)", line)
        if info_match:
            test_losses.append(float(info_match.group(1)))

train_losses = train_losses[1::2]

# Plotting for all epochs
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='train losses')
plt.plot(test_losses, label='test losses')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Losses for train and test for all Epochs')

# Plotting for epochs after 5
plt.subplot(1, 2, 2)
plt.plot(train_losses[5:], label='train losses (after 5 epochs)')
plt.plot(test_losses[5:], label='test losses (after 5 epochs)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Losses for train and test after 5 Epochs')

plt.tight_layout()  # To prevent overlapping layouts
plt.show()
