import matplotlib.pyplot as plt

N = [1024, 9568, 100000]

adamw_times = [160.58, 1660.28, 24263.94]
sgd_times   = [158.60, 1618.25, 21828.40]

plt.figure(figsize=(8, 5))
plt.plot(N, adamw_times, marker="o", label="CUDA AdamW")
plt.plot(N, sgd_times, marker="o", label="CUDA SGD")

plt.xscale("log")
plt.xlabel("Dataset Size (N)")
plt.ylabel("Training Time (ms)")
plt.title("GPU Training Time vs Dataset Size")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("gpu_training_time_vs_N.png", dpi=150, bbox_inches="tight")
plt.show()