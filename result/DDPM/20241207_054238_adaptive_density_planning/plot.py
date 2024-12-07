import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import json
import os
import os.path as osp
import pickle
import warnings

datasets = ["urban_planning"]
folders = os.listdir("./")
final_results = {}
train_info = {}

def smooth(x, window_len=10, window='hanning'):
    s = np.r_[x[window_len-1:0:-1], x, x[-2:-window_len-1:-1]]
    if window == 'flat':
        w = np.ones(window_len, 'd')
    else:
        w = getattr(np, window)(window_len)
    y = np.convolve(w/w.sum(), s, mode='valid')
    return y

for folder in folders:
    if folder.startswith("run") and osp.isdir(folder):
        with open(osp.join(folder, "final_info.json"), "r") as f:
            final_results[folder] = json.load(f)
        all_results = pickle.load(open(osp.join(folder, "all_results.pkl"), "rb"))
        train_info[folder] = all_results

labels = {
    "run_0": "Baseline",
    "run_1": "Learning Rate 1e-4",
    "run_2": "Learning Rate 5e-4",
    "run_3": "Learning Rate 1e-3",
    "run_4": "Hidden Size 128",
    "run_5": "Hidden Size 64"
}

# Generating Color Schemes
def generate_color_palette(n):
    cmap = plt.get_cmap('tab20')
    return [mcolors.rgb2hex(cmap(i)) for i in np.linspace(0, 1, n)]

runs = list(final_results.keys())
colors = generate_color_palette(len(runs))

# Plot 1: Comparison of main indicators
fig, ax = plt.subplots(figsize=(10, 6))
metrics = ['sustainability', 'cost', 'happiness']
x = np.arange(len(metrics))
width = 0.15
multiplier = 0

for run, label in labels.items():
    values = []
    for metric in metrics:
        value = final_results[run]['urban_planning'].get('means', {}).get(metric, 0)
        values.append(value)
    offset = width * multiplier
    rects = ax.bar(x + offset, values, width, label=label)
    ax.bar_label(rects, padding=3, rotation=90, fmt='%.3f')
    multiplier += 1

ax.set_ylabel('Score')
ax.set_title('Urban Planning Metrics')
ax.set_xticks(x + width * (len(labels) - 1) / 2)
ax.set_xticklabels(metrics)
ax.legend()
plt.tight_layout()
plt.savefig("urban_metrics.png")
plt.show()

# Plot 2: 训练损失
fig, ax = plt.subplots(figsize=(8, 6))
for run, label in labels.items():
    losses = train_info[run]['urban_planning']["train_losses"]
    losses = smooth(losses, window_len=25)
    ax.plot(losses, label=label)

ax.set_title("Training Loss")
ax.set_xlabel("Steps")
ax.set_ylabel("Loss")
ax.legend()
plt.tight_layout()
plt.savefig("training_loss.png")
plt.show()
