# TinyLLM-ESP32

A ~81K-parameter bidirectional GRU intent-recognition model, trained in PyTorch and deployed on an ESP32 microcontroller in three numerical variants (FP32, INT8, INT8/FP32 Hybrid). This repo benchmarks all three variants end-to-end on real hardware — accuracy, latency, RAM, and Flash footprint — and provides full reproducibility for review.

**Repo:** https://github.com/Aziz-Marnissi/tinyllm-esp32

---

## 1. Overview

The model parses short natural-language commands ("turn on the led", "set servo to 90 degrees", "what's the temperature") and predicts three outputs jointly:

- **action** — one of `{on, off, set_power, set_degree, set_timer, check_status}`
- **target** — one of `{led, motor_28BYJ48, servo, timer, temp_sensor}`
- **value** — a continuous scalar (e.g. degree, power %, timer minutes), used when relevant

The trained model is quantized and deployed on an ESP32 (240 MHz, 320 KB RAM, 4 MB Flash), where it drives real GPIO pins (LED, stepper motor, servo) from live serial commands.

---

## 2. Model Architecture

### 2.1 Structure

```
Input text
   │
   ▼
┌─────────────┐
│  Tokenizer  │  word-level, fixed vocab (118 tokens), max length 12
└─────────────┘
   │  token ids
   ▼
┌─────────────┐
│  Embedding  │  vocab_size=118 → emb_dim=32
└─────────────┘
   │  [seq_len, 32]
   ▼
┌─────────────────────┐
│  Bidirectional GRU   │  hidden_size=96 (per direction), 1 layer
└─────────────────────┘
   │  concat(h_fwd, h_bwd) → [192]
   ▼
   ├──────────────┬──────────────┬────────────────┐
   ▼              ▼              ▼                │
┌─────────┐  ┌─────────┐  ┌────────────┐          │
│ action  │  │ target  │  │   value    │◄─────────┘ (+ num_feat, num_present
│ head    │  │ head    │  │   head     │              side-channel, 194-d in)
│ Linear  │  │ Linear  │  │ Linear     │
│192→6    │  │192→5    │  │194→1       │
└─────────┘  └─────────┘  └────────────┘
   │              │              │
   ▼              ▼              ▼
 action        target          value
 logits        logits         (scalar)
```

### 2.2 Parameter count (measured, `model_seed2.pt`)

| Layer | Shape | Params |
|---|---|---|
| `embed.weight` | [118, 32] | 3,776 |
| `gru.weight_ih_l0` (forward) | [288, 32] | 9,216 |
| `gru.weight_hh_l0` (forward) | [288, 96] | 27,648 |
| `gru.bias_ih_l0` / `bias_hh_l0` (forward) | [288] × 2 | 576 |
| `gru.weight_ih_l0_reverse` | [288, 32] | 9,216 |
| `gru.weight_hh_l0_reverse` | [288, 96] | 27,648 |
| `gru.bias_ih_l0_reverse` / `bias_hh_l0_reverse` | [288] × 2 | 576 |
| `action_head.weight` / `bias` | [6, 192] / [6] | 1,158 |
| `target_head.weight` / `bias` | [5, 192] / [5] | 965 |
| `value_head.weight` / `bias` | [1, 194] / [1] | 195 |
| **Total** | | **80,974 (~81K)** |

(288 = 3 × 96, the standard GRU gate-stacking factor: reset, update, and candidate gates share the same weight matrix, stacked along the output dimension.)

### 2.3 GRU update equations

For each timestep $t$, with input $x_t \in \mathbb{R}^{32}$ and hidden state $h_{t-1} \in \mathbb{R}^{96}$:

$$
\begin{aligned}
r_t &= \sigma(W_{ir} x_t + b_{ir} + W_{hr} h_{t-1} + b_{hr}) \quad &\text{(reset gate)} \\
z_t &= \sigma(W_{iz} x_t + b_{iz} + W_{hz} h_{t-1} + b_{hz}) \quad &\text{(update gate)} \\
n_t &= \tanh(W_{in} x_t + b_{in} + r_t \odot (W_{hn} h_{t-1} + b_{hn})) \quad &\text{(candidate state)} \\
h_t &= (1 - z_t) \odot n_t + z_t \odot h_{t-1} \quad &\text{(new hidden state)}
\end{aligned}
$$

The model runs this **bidirectionally**: one GRU pass left-to-right ($\overrightarrow{h}$), one right-to-left ($\overleftarrow{h}$), and the final representation is their concatenation:

$$
h_{\text{final}} = [\overrightarrow{h}_T \, ; \, \overleftarrow{h}_1] \in \mathbb{R}^{192}
$$

### 2.4 Output heads

$$
\text{action\_logits} = W_a \, h_{\text{final}} + b_a \in \mathbb{R}^6, \qquad
\text{target\_logits} = W_t \, h_{\text{final}} + b_t \in \mathbb{R}^5
$$

The value head additionally takes a 2-dimensional side-channel $[\text{num\_feat}, \text{num\_present}]$ (whether a number was found in the text, and its normalized magnitude), concatenated to $h_{\text{final}}$:

$$
\hat{v} = W_v \, [h_{\text{final}} \, ; \, \text{num\_feat} \, ; \, \text{num\_present}] + b_v, \qquad \hat{v} \in [0, 1]
$$

The predicted value is de-normalized at inference time as $v = \hat{v} \times \text{MAX\_VALUE}$ (MAX_VALUE = 180).

### 2.5 Loss

$$
\mathcal{L} = \mathcal{L}_{\text{action}}^{\text{CE}} + \mathcal{L}_{\text{target}}^{\text{CE}} + \mathbb{1}_{\text{mask}} \cdot \mathcal{L}_{\text{value}}^{\text{MAE}}
$$

Cross-entropy for the two classification heads, masked L1/MAE for the value head (only counted when the ground-truth sample actually has a value).

---

## 3. Quantization Variants

Three inference implementations of the same trained weights, all sharing an identical `model_api.h` interface:

| Variant | Weights | Accumulation | Nonlinearities | Notes |
|---|---|---|---|---|
| **FP32** | float32 | float32 | float32 | reference baseline, no lookup tables |
| **INT8** | int8 | int32 | fast_sigmoid/fast_tanh (LUT, int8-domain) | fully quantized |
| **Hybrid** | int8 (embed + GRU weights) | int32 → requantized to int8 per timestep | fast_sigmoid/fast_tanh (FP32) | quantized weights, float nonlinearities |

Quantization is post-training, per-tensor symmetric int8 (scale-only, zero-point=0).

---

## 4. Seed Selection

Training is stochastic (weight init, data shuffling). Four seeds were trained and evaluated on validation data before selecting the deployed model:

| Seed | val_action_acc |
|---|---|
| 1 | 86.8% |
| **2** | **94.9%** ← selected |
| 3 | ~85% |
| 4 | ~89.4% |

Seed 2 was a clear, non-marginal standout (>5pp above the next best) and was locked in as `model_seed2.pt`.

---

## 5. Results

### 5.1 On-device benchmark (ESP32, full held-out adversarial test set, n=699)

| Variant | Action Acc | Target Acc | Latency (avg) | RAM | Flash |
|---|---|---|---|---|---|
| INT8 | 81.5% | 100% | 69.7 ms | 21.5 KB | 357.6 KB |
| Hybrid | 81.3% | 100% | 68.9 ms | 21.5 KB | 357.9 KB |
| FP32 | 81.4% | 100% | 251.3 ms | 21.5 KB | 584.3 KB |

**Key finding:** accuracy is statistically indistinguishable across all three variants (quantization introduces no measurable degradation), while INT8/Hybrid are **~3.6× faster** and use **~39% less Flash** than FP32.

![Flash & RAM comparison](evaluation/flash_ram_comparison.png)
![Latency comparison](evaluation/latency_comparison.png)
![Accuracy comparison](evaluation/accuracy_comparison.png)
![Accuracy vs latency tradeoff](evaluation/accuracy_latency_tradeoff.png)

### 5.2 Validation-set diagnostics (host, seed-2 model)

- `action_acc = 94.9%`, `target_acc = 98.7%`, `value_MAE = 26.6`

![Action confusion matrix](evaluation/confusion_action.png)
![Target confusion matrix](evaluation/confusion_target.png)
![Value regression: true vs predicted](evaluation/value_scatter.png)

The action confusion matrix shows the dominant error mode is **"on" misclassified as "off"** (42/699 cases) — a systematic, not random, error worth noting for future data augmentation. The value regression plot shows under-prediction at high true values (points falling below the diagonal past ~100), consistent with the MAE.

> `training_curve.png` (loss/accuracy per epoch) reflects an earlier training run and is kept for illustration; it was not regenerated for the final seed-2 checkpoint (would require a full retrain to reproduce identically).

---

## 6. Repository Structure

```
tinyllm-esp32/
├── backups/              # inference.c variants (float / int8 / hybrid)
├── data/                 # datasets: train/val/test splits, vocab
├── docs/                 # design notes, blueprint slides/PDF
├── evaluation/            # all plots + numerical result summaries
├── scripts/              # training, export, evaluation, comparison scripts
├── src/                  # ESP32 firmware: inference.c (active), tokenizer.c,
│                         # weights.h / weights_float.h / vocab.h, main.cpp
├── tests/                # host-side C test harnesses (no hardware needed)
├── platformio.ini        # PlatformIO build config (ESP32 target)
└── run_quant_comparison.sh   # full host-side 3-variant benchmark + plots
```

---

## 7. Reproducing the Results

### 7.1 Train from scratch (optional — a trained checkpoint is included)

```bash
cd scripts
python3 train.py            # trains with a fixed seed, saves model.pt
                             # early-stops on val accuracy plateau
```

### 7.2 Export weights for the C/ESP32 target

```bash
cd scripts
python3 export_weights_int8.py     # writes weights.h (int8)
python3 export_weights_float.py    # writes weights_float.h (fp32)
python3 export_vocab.py            # writes vocab.h
cp weights.h weights_float.h vocab.h ../src/
```

### 7.3 Host-side sanity check (no ESP32 needed)

Quick functional test — runs a handful of hardcoded commands through the compiled C inference path and prints predictions:

```bash
gcc -O2 -o test_inf tests/test_inference.c src/inference.c src/tokenizer.c -Isrc -lm
./test_inf
```

Expected: readable `action=... target=... value=...` output for each test sentence, no crashes.

### 7.4 Full 3-variant benchmark (host, accuracy + timing)

```bash
bash run_quant_comparison.sh
```

This compiles and evaluates FP32 / INT8 / Hybrid against `data/val.jsonl`, times each on the host CPU, and regenerates all comparison plots into `evaluation/`. Expect ~94-95% action accuracy for all three variants (host timing is **not** representative of ESP32 latency — see §5.1 for real on-device numbers).

### 7.5 Flash to ESP32 and validate live

Requires [PlatformIO](https://platformio.org/) and an ESP32 dev board connected via USB.

```bash
# choose a variant:
cp backups/inference_int8.c.bak src/inference.c      # or inference_hybrid.c.bak / inference_float.c.bak
pio run -t upload                                     # compiles + flashes
pio run -v | grep -E "RAM|Flash"                       # confirms RAM/Flash usage
```

Open the serial monitor and type a command:

```bash
pio device monitor -b 115200
# then type: turn on the led
# expect: action=on target=led value=... (NN.NN ms)
```

### 7.6 Full held-out test-set replay (reproduces §5.1 numbers)

With the ESP32 flashed and connected (adjust `/dev/ttyUSBx` to your port):

```bash
python3 - <<'EOF'
import serial, json, time

ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=3)
time.sleep(5)                       # let the board finish booting
ser.reset_input_buffer()

with open('data/test_adversarial_full.jsonl') as f:
    lines = f.readlines()

with open('results.txt', 'w') as out:
    for line in lines:
        sample = json.loads(line)
        text = sample['text']
        ser.reset_input_buffer()
        ser.write((text + '\n').encode())
        time.sleep(0.3)
        resp = ser.readline().decode(errors='ignore').strip()
        out.write(f"{text}\t{resp}\t{sample['action']}\t{sample['target']}\t{sample.get('value','')}\n")
ser.close()
EOF

python3 scripts/compute_accuracy.py results.txt
```

Expect output close to:
```
n=697-699
action_acc≈0.81
target_acc=1.00
latency_avg_ms≈69 (INT8/Hybrid) or ≈251 (FP32)
```

Repeat for each variant (re-flash between runs) to reproduce the full §5.1 table.

---

## 8. Known Limitations / Notes for Reviewers

- `training_curve.png` is from an earlier training run, not the final seed-2 checkpoint (no epoch-level history was persisted for seed 2 — regenerating it exactly would require a full retrain).
- Host-side latency (§7.4) reflects process-startup overhead, not embedded performance — always refer to §5.1 for real ESP32 numbers.
- The value head's MAE (~26.6, scale 0–180) reflects a genuinely harder regression sub-task; see the confusion-matrix/scatter-plot discussion in §5.2 for where errors concentrate.
- Vocabulary size is fixed at 118 tokens (`src/vocab.h`); this is what the deployed model was trained and exported with. Any regeneration of the dataset/vocab requires retraining before re-export.
