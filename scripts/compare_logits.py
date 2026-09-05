import json
import re
import torch
import sys
sys.path.insert(0, '.')
from model import TinyIntentGRU, ACTIONS, TARGETS, MAX_VALUE
from tokenizer import tokenize, extract_number

vocab = {}
with open('../src/vocab.h') as f:
    for line in f:
        m = re.match(r'\s*\{"((?:[^"\\]|\\.)*)",\s*(\d+)\}', line)
        if m:
            word = m.group(1).encode().decode('unicode_escape')
            vocab[word] = int(m.group(2))

print(f"parsed vocab size: {len(vocab)}")

model = TinyIntentGRU(vocab_size=len(vocab))
state = torch.load('model.pt', map_location='cpu')
model.load_state_dict(state)
model.eval()

tests = [
    "turn on the led",
    "set servo to 90 degrees",
    "trun off motor",
    "brightness 45",
    "set a timer for 10 minutes",
    "what's the temperature",
    "is the led on",
]

max_len = 12
for text in tests:
    ids = [vocab.get(tok, 1) for tok in tokenize(text)][:max_len]
    length = len(ids)
    ids_padded = ids + [0] * (max_len - len(ids))
    x = torch.tensor([ids_padded], dtype=torch.long)
    lengths = torch.tensor([length], dtype=torch.long)
    num = extract_number(text)
    num_feat = torch.tensor([(num or 0) / MAX_VALUE], dtype=torch.float32)
    num_present = torch.tensor([1.0 if num is not None else 0.0], dtype=torch.float32)

    with torch.no_grad():
        action_logits, target_logits, value_out = model(x, lengths, num_feat, num_present)

    a_best = ACTIONS[action_logits.argmax(-1).item()]
    t_best = TARGETS[target_logits.argmax(-1).item()]
    v = value_out.item() * MAX_VALUE

    print(f'"{text}"')
    print(f"  action={a_best} target={t_best} value={v:.1f}")
    print("  action_logits=" + " ".join(f"{x:.6f}" for x in action_logits[0].tolist()))
    print("  target_logits=" + " ".join(f"{x:.6f}" for x in target_logits[0].tolist()))
    print()
