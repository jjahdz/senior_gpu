import matplotlib.pyplot as plt

labels = ["AdamW", "SGD"]
mse = [15.406702, 15.464280]
ojha_mse = 15.095050

plt.figure(figsize=(7, 5))
plt.bar(labels, mse)
plt.axhline(y=ojha_mse, linestyle="--", label="Ojha Reference MSE")

plt.ylabel("MSE (original units)")
plt.title("Power Plant MSE Validation Against Ojha Reference")
plt.ylim(14.8, 15.7)
plt.grid(axis="y", alpha=0.3)
plt.legend()

plt.savefig("powerplant_mse_validation.png", dpi=150, bbox_inches="tight")
plt.show()