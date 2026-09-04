import json
import time
import numpy as np
import torch
from model import TinyIntentGRU, ACTIONS, TARGETS

VOCAB_PATH = "vocab.json"
MODEL_PATH = "model.pt"

vocab = json.load(open(VOCAB_PATH))
model = TinyIntentGRU(vocab_size=len(vocab))
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()
sd = model.state_dict()

HIDDEN = sd["gru.weight_hh_l0"].shape[1]
EMB_DIM = sd["embed.weight"].shape[1]

def quantize(t):
    arr = t.detach().numpy().astype(np.float32)
    scale = max(np.abs(arr).max(), 1e-8) / 127.0
    q = np.round(arr / scale).astype(np.int8)
    return q, scale

EMBED_Q, EMBED_S = quantize(sd["embed.weight"])
WIH_Q, WIH_S = quantize(sd["gru.weight_ih_l0"])
WHH_Q, WHH_S = quantize(sd["gru.weight_hh_l0"])
B_IH = sd["gru.bias_ih_l0"].numpy().astype(np.float32)
B_HH = sd["gru.bias_hh_l0"].numpy().astype(np.float32)
ACTION_W = sd["action_head.weight"].numpy().astype(np.float32)
ACTION_B = sd["action_head.bias"].numpy().astype(np.float32)
TARGET_W = sd["target_head.weight"].numpy().astype(np.float32)
TARGET_B = sd["target_head.bias"].numpy().astype(np.float32)
VALUE_W = sd["value_head.weight"].numpy().astype(np.float32).flatten()
VALUE_B = float(sd["value_head.bias"].numpy()[0])

def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))

def forward_f32(ids):
    W_ih = sd["gru.weight_ih_l0"].numpy().astype(np.float32)
    W_hh = sd["gru.weight_hh_l0"].numpy().astype(np.float32)
    h = np.zeros(HIDDEN, dtype=np.float32)
    embed = sd["embed.weight"].numpy().astype(np.float32)
    for t in ids:
        x = embed[t]
        gi = W_ih @ x + B_IH
        gh = W_hh @ h + B_HH
        r = sigmoid(gi[:HIDDEN] + gh[:HIDDEN])
        z = sigmoid(gi[HIDDEN:2*HIDDEN] + gh[HIDDEN:2*HIDDEN])
        n = np.tanh(gi[2*HIDDEN:] + r * gh[2*HIDDEN:])
        h = (1 - z) * n + z * h
    return h

def quantize_h(h):
    maxabs = max(np.abs(h).max(), 1e-8)
    scale = maxabs / 127.0
    hq = np.round(h / scale).astype(np.int8)
    return hq, scale

def forward_int8(ids):
    h = np.zeros(HIDDEN, dtype=np.float32)
    for t in ids:
        xq = EMBED_Q[t].astype(np.int32)
        hq, h_scale = quantize_h(h)
        hq = hq.astype(np.int32)

        acc_i = (WIH_Q.astype(np.int32) @ xq)
        gi = acc_i.astype(np.float32) * (WIH_S * EMBED_S) + B_IH

        acc_h = (WHH_Q.astype(np.int32) @ hq)
        gh = acc_h.astype(np.float32) * (WHH_S * h_scale) + B_HH

        r = sigmoid(gi[:HIDDEN] + gh[:HIDDEN])
        z = sigmoid(gi[HIDDEN:2*HIDDEN] + gh[HIDDEN:2*HIDDEN])
        n = np.tanh(gi[2*HIDDEN:] + r * gh[2*HIDDEN:])
        h = (1 - z) * n + z * h
    return h

def heads(h):
    action_logits = ACTION_W @ h + ACTION_B
    target_logits = TARGET_W @ h + TARGET_B
    value = sigmoid(VALUE_W @ h + VALUE_B)
    return action_logits, target_logits, value

from tokenizer import tokenize

def to_ids(sentence):
    words = tokenize(sentence)
    return [vocab.get(w, vocab["<unk>"]) for w in words]

TEST_SENTENCES = [
    "turn on the led",
    "set servo to 90 degrees",
    "trun off motor",
    "brightness 45",
]

print(f"{'sentence':<28} {'baseline argmax':<20} {'int8 argmax':<20} {'match'}")
for s in TEST_SENTENCES:
    ids = to_ids(s)
    h_f = forward_f32(ids)
    h_q = forward_int8(ids)
    a_f, t_f, v_f = heads(h_f)
    a_q, t_q, v_q = heads(h_q)
    match = (a_f.argmax() == a_q.argmax()) and (t_f.argmax() == t_q.argmax())
    print(f"{s:<28} a={a_f.argmax()} t={t_f.argmax():<14} a={a_q.argmax()} t={t_q.argmax():<14} {match}")
    print(f"   value: baseline={v_f:.4f}  int8={v_q:.4f}  diff={abs(v_f-v_q):.4f}")

N = 200
ids = to_ids(TEST_SENTENCES[0])

t0 = time.perf_counter()
for _ in range(N): forward_f32(ids)
t_f32 = (time.perf_counter() - t0) / N

t0 = time.perf_counter()
for _ in range(N): forward_int8(ids)
t_int8 = (time.perf_counter() - t0) / N

print(f"\nfloat32 numpy: {t_f32*1000:.4f} ms")
print(f"int8    numpy: {t_int8*1000:.4f} ms")
