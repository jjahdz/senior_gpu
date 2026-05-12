import pandas as pd
import numpy as np
from pathlib import Path

input_path = Path("C:/Users/green/senior/Code/opt/data/powerplant/CCPP/Folds5x2_pp.xlsx")
if input_path.suffix.lower() in {".xls", ".xlsx"}:
    df = pd.read_excel(input_path)
elif input_path.suffix.lower() == ".csv":
    df = pd.read_csv(input_path)
else:
    raise ValueError(f"Unsupported input file type: {input_path.suffix}")

x = df['V'].values.astype(np.float32)
y = df['AT'].values.astype(np.float32)

# Normalize x to [-1, 1]
x_min, x_max = x.min(), x.max()
x_norm = 2.0 * (x - x_min) / (x_max - x_min) - 1.0

# Normalize y to zero mean, unit variance
y_mean = y.mean()
y_std  = y.std()
y_norm = (y - y_mean) / y_std

x_norm.tofile("powerplant_x.bin")
y_norm.tofile("powerplant_y.bin")

# Print stats so you can verify and denormalize later
print(f"N = {len(x)}")
print(f"x range: [{x_min:.4f}, {x_max:.4f}] -> normalized to [-1, 1]")
print(f"y mean: {y_mean:.4f}, y std: {y_std:.4f}")
print(f"y range after norm: [{y_norm.min():.4f}, {y_norm.max():.4f}]")