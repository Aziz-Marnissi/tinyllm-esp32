#!/bin/bash
set -e

TINYLLM=~/tinyllm
mkdir -p "$TINYLLM/scripts/evaluation"

# --- write the python plotting script ---
cat > "$TINYLLM/scripts/compare_quant_variants.py" << 'EOF'
import matplotlib.pyplot as plt
import numpy as np
import os

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluation")
os.makedirs(OUT_DIR, exist_ok=True)

variants = ["float32\n(no LUT)", "int8\n(quantized)", "hybrid\n(int8+dyn-requant)"]

# ESP32 on-device latency (from serial monitor) - the numbers that actually matter
esp32_latency_ms = [182.42, 52.02, 51.53]

flash_bytes = [598465, 366405, 366677]
ram_bytes = [21500, 21496, 21496]
ram_bytes = [21480, 21480, 23080]

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
    for name, lat, acc in zip(variants, esp32_latency_ms, action_acc):
        if acc is None:
            continue
        ax.scatter(lat, acc, s=80)
        ax.annotate(name.replace("\n", " "), (lat, acc), textcoords="offset points", xytext=(5, 5))
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
EOF

echo "Wrote $TINYLLM/scripts/compare_quant_variants.py"

# --- write the C accuracy evaluator ---
cat > "$TINYLLM/eval_accuracy.c" << 'EOF'
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "model_api.h"

#define MAX_LINE 512

static const char* ACTIONS[6] = {"on", "off", "set_power", "set_degree", "set_timer", "check_status"};
static const char* TARGETS[5] = {"led", "motor_28BYJ48", "servo", "timer", "temp_sensor"};

static int find_index(const char* arr[], int n, const char* val) {
    for (int i = 0; i < n; i++) if (strcmp(arr[i], val) == 0) return i;
    return -1;
}

static int extract_str(const char* line, const char* key, char* out, int out_sz) {
    char pattern[64];
    snprintf(pattern, sizeof(pattern), "\"%s\": \"", key);
    const char* p = strstr(line, pattern);
    if (!p) return 0;
    p += strlen(pattern);
    const char* end = strchr(p, '"');
    if (!end) return 0;
    int len = end - p;
    if (len >= out_sz) len = out_sz - 1;
    strncpy(out, p, len);
    out[len] = '\0';
    return 1;
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s val.jsonl\n", argv[0]); return 1; }
    FILE* f = fopen(argv[1], "r");
    if (!f) { perror("fopen"); return 1; }

    char line[MAX_LINE];
    int total = 0, correct_action = 0, correct_target = 0, correct_both = 0;

    while (fgets(line, sizeof(line), f)) {
        char text[MAX_LEN * 16], action_str[32], target_str[32];
        if (!extract_str(line, "text", text, sizeof(text))) continue;
        if (!extract_str(line, "action", action_str, sizeof(action_str))) continue;
        if (!extract_str(line, "target", target_str, sizeof(target_str))) continue;

        int true_action = find_index(ACTIONS, 6, action_str);
        int true_target = find_index(TARGETS, 5, target_str);
        if (true_action < 0 || true_target < 0) continue;

        int ids[MAX_LEN];
        int length = tokenize(text, ids);

        float action_logits[N_ACTIONS], target_logits[N_TARGETS], value_out;
        model_forward(ids, length, action_logits, target_logits, &value_out);

        int pred_action = 0;
        for (int i = 1; i < N_ACTIONS; i++)
            if (action_logits[i] > action_logits[pred_action]) pred_action = i;

        int pred_target = 0;
        for (int i = 1; i < N_TARGETS; i++)
            if (target_logits[i] > target_logits[pred_target]) pred_target = i;

        total++;
        if (pred_action == true_action) correct_action++;
        if (pred_target == true_target) correct_target++;
        if (pred_action == true_action && pred_target == true_target) correct_both++;
    }
    fclose(f);

    if (total == 0) { fprintf(stderr, "no valid samples parsed\n"); return 1; }

    printf("total=%d\n", total);
    printf("action_acc=%.4f\n", (double)correct_action / total);
    printf("target_acc=%.4f\n", (double)correct_target / total);
    printf("both_acc=%.4f\n", (double)correct_both / total);
    return 0;
}
EOF
echo "Wrote $TINYLLM/eval_accuracy.c"

# --- compile + time + evaluate accuracy for each variant on host ---
VARIANTS=("inference_float.c.bak" "inference_int8.c.bak" "inference_hybrid.c.bak")
LABELS=("float32 (no LUT)" "int8" "hybrid")
RESULTS_FILE="$TINYLLM/scripts/evaluation/latency_results.txt"
ACC_FILE="$TINYLLM/scripts/evaluation/accuracy_results.txt"
> "$RESULTS_FILE"
> "$ACC_FILE"

for i in "${!VARIANTS[@]}"; do
    variant="${VARIANTS[$i]}"
    label="${LABELS[$i]}"
    echo "=== Testing $label ($variant) ==="
    cp "$TINYLLM/backups/$variant" "$TINYLLM/src/inference.c"

    # int8-based weights.h (int8-quantized) must exist for int8/hybrid variants;
    # weights_float.h (true float32) must exist for the float32 variant.
    # both are already generated in $TINYLLM/src/ - no copy needed if present.
    if grep -q '"weights_float.h"' "$TINYLLM/src/inference.c" && [ ! -f "$TINYLLM/src/weights_float.h" ]; then
        echo "ERROR: weights_float.h missing in $TINYLLM/src/ - skipping $label"
        continue
    fi
    if grep -q '"weights.h"' "$TINYLLM/src/inference.c" && [ ! -f "$TINYLLM/src/weights.h" ]; then
        echo "ERROR: weights.h missing in $TINYLLM/src/ - skipping $label"
        continue
    fi

    gcc -O2 -o "$TINYLLM/test_inf_bin" "$TINYLLM/tests/test_inference.c" "$TINYLLM/src/inference.c" "$TINYLLM/src/tokenizer.c" -I"$TINYLLM/src" -lm

    total=0
    runs=20
    for j in $(seq 1 $runs); do
        start=$(date +%s%N)
        "$TINYLLM/test_inf_bin" > /dev/null
        end=$(date +%s%N)
        total=$((total + (end - start)))
    done
    avg_ms=$(echo "scale=4; $total / $runs / 1000000" | bc)
    echo "$label,$avg_ms" >> "$RESULTS_FILE"
    echo "  avg host latency: ${avg_ms} ms"

    # --- accuracy on full val.jsonl ---
    gcc -O2 -o "$TINYLLM/eval_bin" "$TINYLLM/eval_accuracy.c" "$TINYLLM/src/inference.c" "$TINYLLM/src/tokenizer.c" -I"$TINYLLM/src" -lm
    acc_out=$("$TINYLLM/eval_bin" "$TINYLLM/data/val.jsonl")
    echo "$acc_out"
    action_acc=$(echo "$acc_out" | grep action_acc | cut -d= -f2)
    target_acc=$(echo "$acc_out" | grep target_acc | cut -d= -f2)
    echo "$label,$action_acc,$target_acc" >> "$ACC_FILE"
done
rm -f "$TINYLLM/test_inf_bin" "$TINYLLM/eval_bin"

echo ""
echo "=== Latency Results ==="
cat "$RESULTS_FILE"
echo ""
echo "=== Accuracy Results ==="
cat "$ACC_FILE"

echo ""
echo "Generating plots..."
python3 "$TINYLLM/scripts/compare_quant_variants.py"
