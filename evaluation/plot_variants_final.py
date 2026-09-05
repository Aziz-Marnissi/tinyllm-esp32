import matplotlib.pyplot as plt

variants = ["INT8", "Hybrid", "FP32"]

metrics = {
    "Action Acc (%)": [81.5, 81.3, 81.4],
    "Target Acc (%)": [100.0, 100.0, 100.0],
    "Latency (ms)":   [69.7, 68.9, 251.3],
    "RAM (KB)":       [21.5, 21.5, 21.5],
    "Flash (KB)":     [357.6, 357.9, 584.3],  # 366217/1024, 366489/1024, 598269/1024
}

fig, axes = plt.subplots(1, len(metrics), figsize=(4*len(metrics), 4))
colors = ["#4C72B0", "#DD8452", "#55A868"]

for ax, (name, vals) in zip(axes, metrics.items()):
    bars = ax.bar(variants, vals, color=colors)
    ax.set_title(name)
    ax.bar_label(bars, fmt="%.1f")

plt.suptitle("ESP32 GRU (8k params) — INT8 vs Hybrid vs FP32 (n=699)")
plt.tight_layout()
plt.savefig("variants_comparison.png", dpi=150)
print("saved variants_comparison.png")
