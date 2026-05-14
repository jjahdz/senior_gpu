import matplotlib.pyplot as plt

N_values  = [1024, 9568, 100000]
cpu_times = [0.37, 3.81, 44.91]
gpu_times = [157.37, 1758.92, 22835.43]

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(N_values, gpu_times, 'b-o', markersize=7, 
        linewidth=2, label='CUDA GPU AdamW')
ax.plot(N_values, cpu_times, 'r-o', markersize=7, 
        linewidth=2, label='Serial CPU AdamW')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Dataset Size (N)', fontsize=12)
ax.set_ylabel('Training Time (ms)', fontsize=12)
ax.set_title('Training Time Scaling: GPU vs Serial CPU\n'
             '(AdamW, lr=0.01, 200 epochs, batch=256)', fontsize=11)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

for n, cpu, gpu in zip(N_values, cpu_times, gpu_times):
    ratio = gpu / cpu
    ax.annotate(f'{ratio:.0f}x slower', 
                xy=(n, gpu),
                xytext=(0, 12), 
                textcoords='offset points',
                ha='center', fontsize=9, color='blue')

plt.tight_layout()
plt.savefig('scaling.png', dpi=150, bbox_inches='tight')
plt.show()