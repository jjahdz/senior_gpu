from pathlib import Path
import pandas as pd
import numpy as np

# -----------------------------
# CONFIG
# -----------------------------
CSV_PATH = Path(r"C:\\Users\\green\\senior\\Code\\opt\\data\\news\\Data\\News_Final.csv")

# Pick one input feature and one target.
# Good options:
#   x = SentimentHeadline, y = Facebook
#   x = SentimentTitle,    y = Facebook
#   x = SentimentHeadline, y = LinkedIn
#   x = SentimentHeadline, y = GooglePlus
X_COL = "SentimentHeadline"
Y_COL = "Facebook"

OUT_X = "news_x.bin"
OUT_Y = "news_y.bin"
OUT_CONSTANTS = "news_norm_constants.txt"

NORMALIZE = True
DROP_DUPLICATES = False


def normalize_column_name(name: str) -> str:
    return (
        str(name)
        .strip()
        .lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
    )


def find_column(df: pd.DataFrame, wanted: str) -> str:
    wanted_key = normalize_column_name(wanted)

    for col in df.columns:
        if normalize_column_name(col) == wanted_key:
            return col

    raise ValueError(
        f"Could not find column '{wanted}'.\n"
        f"Available columns:\n{list(df.columns)}"
    )


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Could not find {CSV_PATH.resolve()}")

    df = pd.read_csv(CSV_PATH)

    print("Loaded:", CSV_PATH)
    print("Rows:", len(df))
    print("Columns:")
    for col in df.columns:
        print(" -", col)

    x_col = find_column(df, X_COL)
    y_col = find_column(df, Y_COL)

    data = df[[x_col, y_col]].copy()
    data.columns = ["x", "y"]

    data["x"] = pd.to_numeric(data["x"], errors="coerce")
    data["y"] = pd.to_numeric(data["y"], errors="coerce")

    before_dropna = len(data)
    data = data.dropna()
    print(f"\nDropped NaN rows: {before_dropna - len(data)}")

    if DROP_DUPLICATES:
        before_dupes = len(data)
        data = data.drop_duplicates()
        print(f"Dropped duplicate x/y rows: {before_dupes - len(data)}")

    x_raw = data["x"].to_numpy(dtype=np.float32)
    y_raw = data["y"].to_numpy(dtype=np.float32)

    print("\nSelected columns:")
    print(f"x = {x_col}")
    print(f"y = {y_col}")
    print(f"N = {len(x_raw)}")

    print("\nRaw stats:")
    print(f"x min={x_raw.min():.6f}, max={x_raw.max():.6f}, mean={x_raw.mean():.6f}, std={x_raw.std():.6f}")
    print(f"y min={y_raw.min():.6f}, max={y_raw.max():.6f}, mean={y_raw.mean():.6f}, std={y_raw.std():.6f}")

    if NORMALIZE:
        x_mean = float(x_raw.mean())
        x_std = float(x_raw.std())
        y_mean = float(y_raw.mean())
        y_std = float(y_raw.std())

        if x_std == 0.0:
            raise ValueError("x_std is zero. Choose a different x column.")
        if y_std == 0.0:
            raise ValueError("y_std is zero. Choose a different y column.")

        x = ((x_raw - x_mean) / x_std).astype(np.float32)
        y = ((y_raw - y_mean) / y_std).astype(np.float32)
    else:
        x_mean = 0.0
        x_std = 1.0
        y_mean = 0.0
        y_std = 1.0
        x = x_raw
        y = y_raw

    x.tofile(OUT_X)
    y.tofile(OUT_Y)

    with open(OUT_CONSTANTS, "w") as f:
        f.write(f"N={len(x)}\n")
        f.write(f"X_COL={x_col}\n")
        f.write(f"Y_COL={y_col}\n")
        f.write(f"NEWS_X_MEAN={x_mean:.10f}\n")
        f.write(f"NEWS_X_STD={x_std:.10f}\n")
        f.write(f"NEWS_Y_MEAN={y_mean:.10f}\n")
        f.write(f"NEWS_Y_STD={y_std:.10f}\n")

    print("\nWrote:")
    print(f" - {OUT_X}")
    print(f" - {OUT_Y}")
    print(f" - {OUT_CONSTANTS}")

    print("\nUse this in model.h:")
    print(f"#define N {len(x)}")

    print("\nUse this for original-unit MSE:")
    print("mse_original = mse_normalized * NEWS_Y_STD * NEWS_Y_STD")

    print("\nFirst 5 normalized rows:")
    for i in range(min(5, len(x))):
        print(f"{i}: x={x[i]:.6f}, y={y[i]:.6f}")


if __name__ == "__main__":
    main()