import json
import torch
import matplotlib.pyplot as plt
from model import TinyIntentGRU, IntentDataset, MAX_VALUE

vocab = json.load(open("vocab.json"))
model = TinyIntentGRU(vocab_size=len(vocab))
model.load_state_dict(torch.load("model.pt", map_location="cpu"))
model.eval()

val_ds = IntentDataset("val.jsonl", vocab)

true_vals, pred_vals = [], []
with torch.no_grad():
    for i in range(len(val_ds)):
        ids, length, action, target, value, mask, num_feat, num_present = val_ds[i]
        if mask.item() == 0:
            continue
        _, _, v_pred = model(ids.unsqueeze(0), length.unsqueeze(0),
                              num_feat.unsqueeze(0), num_present.unsqueeze(0))
        true_vals.append(value.item() * MAX_VALUE)
        pred_vals.append(v_pred.item() * MAX_VALUE)

plt.figure(figsize=(6, 6))
plt.scatter(true_vals, pred_vals, alpha=0.5)
plt.plot([0, MAX_VALUE], [0, MAX_VALUE], "r--")
plt.xlabel("True value")
plt.ylabel("Predicted value")
plt.title("Value Regression: True vs Predicted")
plt.tight_layout()
plt.savefig("evaluation/value_scatter.png", dpi=100)
print("saved evaluation/value_scatter.png")
