import matplotlib.pyplot as plt
import numpy as np
import os

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluation")
os.makedirs(OUT_DIR, exist_ok=True)

variants = ["float32\n(no LUT)", "int8\n(quantized)", "hybrid\n(int8+dyn-requant)"]

# ESP32 on-device latency (from serial monitor) - the numbers that actually matter
esp32_latency_ms = [251.3, 69.7, 68.9]

flash_bytes = [598269, 366217, 366489]
ram_bytes = [21500, 21496, 21496]

# Host (laptop) latency - dominated by process startup overhead, NOT representative
# of the embedded target. Kept only as a sanity check that each variant runs.
host_results_file = os.path.join(OUT_DIR, "latency_results.txt")
host_latency_ms = [None, None, None]
if os.path.exists(host_results_file):
    measured = {}
    with open(host_results_file) as f:
        for line in f:
            if "," in line:
                label, val = line.strip().split(",")
                measured[label] = float(val)
    label_map = {"float32 (no LUT)": "float32\n(no LUT)",
                 "int8": "int8\n(quantized)", "hybrid": "hybrid\n(int8+dyn-requant)"}
    for k, v in measured.items():
        mapped = label_map.get(k)
        if mapped in variants:
            host_latency_ms[variants.index(mapped)] = v

action_acc = [None, None, None]
target_acc = [None, None, None]

acc_results_file = os.path.join(OUT_DIR, "accuracy_results.txt")
if os.path.exists(acc_results_file):
    label_map2 = {"float32 (no LUT)": "float32\n(no LUT)",
                  "int8": "int8\n(quantized)", "hybrid": "hybrid\n(int8+dyn-requant)"}
    with open(acc_results_file) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) == 3:
                label, a_acc, t_acc = parts
                mapped = label_map2.get(label)
                if mapped in variants:
                    idx = variants.index(mapped)
                    action_acc[idx] = float(a_acc)
                    target_acc[idx] = float(t_acc)

def plot_latency():
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    colors = ["#C44E52", "#4C72B0", "#55A868"]

    bars0 = ax[0].bar(variants, esp32_latency_ms, color=colors)
    ax[0].set_ylabel("Latency (ms)")
    ax[0].set_title("ESP32 On-Device Latency\n(the numbers that matter)")
    for b, v in zip(bars0, esp32_latency_ms):
        ax[0].text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}", ha="center", va="bottom")

    host_vals = [v if v is not None else 0 for v in host_latency_ms]
    bars1 = ax[1].bar(variants, host_vals, color=colors, alpha=0.5)
    ax[1].set_ylabel("Latency (ms)")
    ax[1].set_title("Host (laptop) Latency\n(startup-overhead dominated, NOT representative)")
    for b, v in zip(bars1, host_vals):
        ax[1].text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}", ha="center", va="bottom")

    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/latency_comparison.png", dpi=150)
    plt.close(fig)
    print(f"Saved {OUT_DIR}/latency_comparison.png")

def plot_accuracy():
    if all(a is None for a in action_acc) and all(a is None for a in target_acc):
        print("No accuracy values set — skipping accuracy plot.")
        return
    x = np.arange(len(variants))
    w = 0.35
    fig, ax = plt.subplots(figsize=(6, 4))
    a_vals = [a if a is not None else 0 for a in action_acc]
    t_vals = [t if t is not None else 0 for t in target_acc]
    ax.bar(x - w / 2, a_vals, w, label="action acc", color="#4C72B0")
    ax.bar(x + w / 2, t_vals, w, label="target acc", color="#55A868")
    ax.set_xticks(x)
    ax.set_xticklabels(variants)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Validation accuracy")
    ax.set_title("Accuracy by Quantization Variant")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/accuracy_comparison.png", dpi=150)
    plt.close(fig)
    print(f"Saved {OUT_DIR}/accuracy_comparison.png")

def plot_tradeoff():
    if all(a is None for a in action_acc):
        print("No accuracy values set — skipping tradeoff plot.")
        return
    fig, ax = plt.subplots(figsize=(5, 5))
    for i, (name, lat, acc) in enumerate(zip(variants, esp32_latency_ms, action_acc)):
        if acc is None:
            continue
        ax.scatter(lat, acc, s=80)
        ax.annotate(name.replace("\n", " "), (lat, acc), textcoords="offset points", xytext=(5, 5 + 15*i))
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Action accuracy")
    ax.set_title("Accuracy vs Latency Tradeoff")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/accuracy_latency_tradeoff.png", dpi=150)
    plt.close(fig)
    print(f"Saved {OUT_DIR}/accuracy_latency_tradeoff.png")

def plot_flash_ram():
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    colors = ["#C44E52", "#4C72B0", "#55A868"]

    flash_kb = [f / 1024 for f in flash_bytes]
    bars0 = ax[0].bar(variants, flash_kb, color=colors)
    ax[0].set_ylabel("Flash (KB)")
    ax[0].set_title("ESP32 Flash Usage")
    for b, v in zip(bars0, flash_kb):
        ax[0].text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}", ha="center", va="bottom")

    ram_kb = [r / 1024 for r in ram_bytes]
    bars1 = ax[1].bar(variants, ram_kb, color=colors)
    ax[1].set_ylabel("RAM (KB)")
    ax[1].set_title("ESP32 RAM Usage")
    for b, v in zip(bars1, ram_kb):
        ax[1].text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom")

    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/flash_ram_comparison.png", dpi=150)
    plt.close(fig)
    print(f"Saved {OUT_DIR}/flash_ram_comparison.png")


if __name__ == "__main__":
    plot_latency()
    plot_accuracy()
    plot_tradeoff()
    plot_flash_ram()
