import json
import time
import torch
from model import TinyIntentGRU, MAX_VALUE
from tokenizer import tokenize

vocab = json.load(open("vocab.json"))
model = TinyIntentGRU(vocab_size=len(vocab))
model.load_state_dict(torch.load("model.pt", map_location="cpu"))
model.eval()

MAX_LEN = 12
tests = [
    "turn on the led",
    "set servo to 90 degrees",
    "trun off motor",
    "brightness 45",
]

for s in tests:
    ids = [vocab.get(tok, 1) for tok in tokenize(s)][:MAX_LEN]
    ids += [0] * (MAX_LEN - len(ids))
    x = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        a_logits, t_logits, v = model(x)
    print(f'"{s}"')
    print(f"  action_logits: {a_logits.squeeze().tolist()} -> argmax={a_logits.argmax(-1).item()}")
    print(f"  target_logits: {t_logits.squeeze().tolist()} -> argmax={t_logits.argmax(-1).item()}")
    print(f"  value={v.item():.4f} (scaled={v.item()*MAX_VALUE:.2f})\n")

x = torch.tensor([[vocab.get(tok, 1) for tok in tokenize(tests[0])][:MAX_LEN] + [0]*(MAX_LEN - len(tokenize(tests[0])))], dtype=torch.long)
t0 = time.perf_counter()
with torch.no_grad():
    for _ in range(1000):
        model(x)
dt = (time.perf_counter() - t0) / 1000
print(f"PC latency: {dt*1000:.3f} ms")
