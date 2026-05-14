import re
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterMathtext
plt.gca().yaxis.set_major_formatter(LogFormatterMathtext())

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

adamw_file = MSE_DIR / "adamw_synth_mse.txt"
sgd_file = MSE_DIR / "sgd_synth_mse.txt"

adamw_epochs, adamw_mse = read_mse_log(adamw_file)
sgd_epochs, sgd_mse = read_mse_log(sgd_file)

# Keep both optimizers on the same visible epoch range.
adamw_epochs, adamw_mse = limit_epochs(adamw_epochs, adamw_mse, 200)
sgd_epochs, sgd_mse = limit_epochs(sgd_epochs, sgd_mse, 200)

# Log-scale plot
plt.figure(figsize=(8, 5))
plt.plot(adamw_epochs, adamw_mse, marker="o", markersize=4, label="AdamW")
plt.plot(sgd_epochs, sgd_mse, marker="o", markersize=4, label="SGD")

plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("Synthetic Cubic N=100000: MSE vs Epoch")
plt.xlim(0, 200)
plt.legend()
plt.grid(True, alpha=0.3)

plt.savefig(BASE / "synthetic_mse_vs_epoch_log.png", dpi=150, bbox_inches="tight")
plt.show()


# Regular-scale plot
plt.figure(figsize=(8, 5))
plt.plot(adamw_epochs, adamw_mse, linewidth=2, label="AdamW")
plt.plot(sgd_epochs, sgd_mse, linewidth=2, label="SGD")

plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("MSE (log scale)")
plt.title("Synthetic Cubic: MSE vs Epoch")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("synthetic_mse_vs_epoch_log.png", dpi=150, bbox_inches="tight")
plt.show()