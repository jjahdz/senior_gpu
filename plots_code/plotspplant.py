import re
from pathlib import Path
import matplotlib.pyplot as plt


def read_mse_log(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Could not find file: {path}")

    pattern = re.compile(
        r"MSE_LOG\s+(\d+)\s+([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
    )

    encodings_to_try = ["utf-8-sig", "utf-8", "utf-16", "utf-16-le", "latin1"]

    for encoding in encodings_to_try:
        epochs = []
        mses = []

        try:
            with open(path, "r", encoding=encoding) as f:
                for line in f:
                    match = pattern.search(line)
                    if match:
                        epochs.append(int(match.group(1)))
                        mses.append(float(match.group(2)))

            if epochs:
                print(f"Read {len(epochs)} MSE_LOG rows from {path.name} using {encoding}")
                return epochs, mses

        except UnicodeDecodeError:
            continue

    raise ValueError(f"No MSE_LOG lines found or file could not be decoded: {path}")


def limit_epochs(epochs, mses, max_epoch):
    filtered = [(e, m) for e, m in zip(epochs, mses) if e <= max_epoch]
    return [x[0] for x in filtered], [x[1] for x in filtered]


BASE = Path(r"C:\Users\green\senior\Code\opt\src")
MSE_DIR = BASE / "mse"

adamw_file = MSE_DIR / "adamw_pplant_mse.txt"
sgd_file = MSE_DIR / "sgd_pplant_mse.txt"

adamw_epochs, adamw_mse = read_mse_log(adamw_file)
sgd_epochs, sgd_mse = read_mse_log(sgd_file)

# Compare first 200 epochs so AdamW and SGD are shown over the same range.
adamw_epochs, adamw_mse = limit_epochs(adamw_epochs, adamw_mse, 200)
sgd_epochs, sgd_mse = limit_epochs(sgd_epochs, sgd_mse, 200)

# Full-scale plot
plt.figure(figsize=(8, 5))
plt.plot(adamw_epochs, adamw_mse, marker="o", markersize=4, label="AdamW")
plt.plot(sgd_epochs, sgd_mse, marker="o", markersize=4, label="SGD")
plt.axhline(15.095050, linestyle="--", label="Ojha Reference MSE")

plt.xlabel("Epoch")
plt.ylabel("MSE (original units)")
plt.title("Power Plant: MSE vs Epoch")
plt.xlim(0, 200)
plt.legend()
plt.grid(True, alpha=0.3)

plt.savefig(BASE / "powerplant_mse_vs_epoch_full.png", dpi=150, bbox_inches="tight")
plt.show()


# Zoomed plot for presentation
plt.figure(figsize=(8, 5))
plt.plot(adamw_epochs, adamw_mse, marker="o", markersize=4, label="AdamW")
plt.plot(sgd_epochs, sgd_mse, marker="o", markersize=4, label="SGD")
plt.axhline(15.095050, linestyle="--", label="Ojha Reference MSE")

plt.xlabel("Epoch")
plt.ylabel("MSE (original units)")
plt.title("Power Plant: MSE vs Epoch, Zoomed")
plt.xlim(0, 200)
plt.ylim(15.0, 16.8)
plt.legend()
plt.grid(True, alpha=0.3)

plt.savefig(BASE / "powerplant_mse_vs_epoch_zoomed.png", dpi=150, bbox_inches="tight")
plt.show()