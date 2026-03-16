import re
import glob
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def parse_log(path):
    meta = {}
    train_losses, test_losses = [], []
    with open(path) as f:
        for line in f:
            for key, pattern in [
                ("model", r"Model type: (.+)"),
                ("epochs", r"Epochs: (.+)"),
                ("lr", r"Starting LR: (.+)"),
                ("gamma", r"Gamma: (.+)"),
            ]:
                m = re.match(pattern, line)
                if m:
                    meta[key] = m.group(1).strip()
            m = re.search(r"Avg train loss: ([\d.]+)", line)
            if m:
                train_losses.append(float(m.group(1)))
            m = re.search(r"Avg test loss: ([\d.]+)", line)
            if m:
                test_losses.append(float(m.group(1)))
    return meta, train_losses, test_losses


log_paths = sorted(glob.glob("runs/*/training_log.txt"))

if not log_paths:
    print("No training_log.txt files found under runs/")
    raise SystemExit(1)

fig, (ax_train, ax_test) = plt.subplots(1, 2, figsize=(22, 8), sharey=True)

for path in log_paths:
    meta, train, test = parse_log(path)
    run_name = os.path.basename(os.path.dirname(path))
    epochs = list(range(1, len(train) + 1))

    label_parts = [meta.get("model", run_name)]
    if "lr" in meta:
        label_parts.append(f"lr={meta['lr']}")
    if "gamma" in meta:
        label_parts.append(f"γ={meta['gamma']}")
    if "epochs" in meta:
        label_parts.append(f"{meta['epochs']} epochs")
    label = ", ".join(label_parts)

    (line,) = ax_train.plot(epochs, train, linewidth=2, label=label)
    ax_test.plot(epochs, test, linewidth=2, color=line.get_color(), label=label)

for ax, title in [(ax_train, "Train Loss"), (ax_test, "Test Loss")]:
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Avg Loss")
    ax.legend()
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)

plt.tight_layout()

# out = "all_runs.png"
# plt.savefig(out, dpi=150, bbox_inches="tight")
# print(f"Saved to {out}")
plt.show()
