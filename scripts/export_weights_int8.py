import json
import numpy as np
import torch
from model import TinyIntentGRU, ACTIONS, TARGETS
VOCAB_PATH = "vocab.json"
MODEL_PATH = "model.pt"
OUT_PATH = "weights.h"
vocab = json.load(open(VOCAB_PATH))
model = TinyIntentGRU(vocab_size=len(vocab))
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()
sd = model.state_dict()


def quantize(tensor):
    arr = tensor.detach().numpy().astype(np.float32).flatten()
    scale = max(np.abs(arr).max(), 1e-8) / 127.0
    q = np.round(arr / scale).astype(np.int8)
    return q, scale


def carr_i8(name, tensor):
    q, scale = quantize(tensor)
    vals = ", ".join(str(int(v)) for v in q)
    out = f"static const int8_t {name}[{q.size}] = {{{vals}}};\n"
    out += f"static const float {name}_SCALE = {scale:.10f}f;\n"
    return out


def carr_f32(name, tensor):
    arr = tensor.detach().numpy().astype(np.float32).flatten()
    vals = ", ".join(f"{v:.8f}f" for v in arr)
    return f"static const float {name}[{arr.size}] = {{{vals}}};\n"


hidden = sd["gru.weight_hh_l0"].shape[1]

lines = []
lines.append("// Auto-generated (int8 weights, float32 biases) -- bidirectional GRU -- do not edit by hand\n")
lines.append("#include <stdint.h>\n")
lines.append(f"#define VOCAB_SIZE {len(vocab)}\n")
lines.append(f"#define EMB_DIM {sd['embed.weight'].shape[1]}\n")
lines.append(f"#define HIDDEN {hidden}\n")
lines.append(f"#define N_ACTIONS {len(ACTIONS)}\n")
lines.append(f"#define N_TARGETS {len(TARGETS)}\n\n")

lines.append(carr_i8("EMBED_W", sd["embed.weight"]))

lines.append(carr_i8("GRU_W_IH", sd["gru.weight_ih_l0"]))
lines.append(carr_i8("GRU_W_HH", sd["gru.weight_hh_l0"]))
lines.append(carr_f32("GRU_B_IH", sd["gru.bias_ih_l0"]))
lines.append(carr_f32("GRU_B_HH", sd["gru.bias_hh_l0"]))

lines.append(carr_i8("GRU_W_IH_REV", sd["gru.weight_ih_l0_reverse"]))
lines.append(carr_i8("GRU_W_HH_REV", sd["gru.weight_hh_l0_reverse"]))
lines.append(carr_f32("GRU_B_IH_REV", sd["gru.bias_ih_l0_reverse"]))
lines.append(carr_f32("GRU_B_HH_REV", sd["gru.bias_hh_l0_reverse"]))

lines.append(carr_f32("ACTION_W", sd["action_head.weight"]))
lines.append(carr_f32("ACTION_B", sd["action_head.bias"]))
lines.append(carr_f32("TARGET_W", sd["target_head.weight"]))
lines.append(carr_f32("TARGET_B", sd["target_head.bias"]))
lines.append(carr_f32("VALUE_W", sd["value_head.weight"]))
lines.append(carr_f32("VALUE_B", sd["value_head.bias"]))

with open(OUT_PATH, "w") as f:
    f.writelines(lines)

print(f"wrote {OUT_PATH}")
