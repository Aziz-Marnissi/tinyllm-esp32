import json, os
import torch
import numpy as np
from model import TinyIntentGRU, IntentDataset, ACTIONS, TARGETS, MAX_VALUE
from train import confusion_matrix, plot_confusion

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
vocab = json.load(open(os.path.join(BASE_DIR, "vocab.json")))

val_ds = IntentDataset(os.path.join(BASE_DIR, "val.jsonl"), vocab)
model = TinyIntentGRU(vocab_size=len(vocab), dropout=0.2)
model.load_state_dict(torch.load("model.pt", map_location="cpu"))
model.eval()

OUT_DIR = "evaluation"
os.makedirs(OUT_DIR, exist_ok=True)

y_action_true, y_action_pred = [], []
y_target_true, y_target_pred = [], []

with torch.no_grad():
    for i in range(len(val_ds)):
        ids, length, action, target, value, mask, num_feat, num_present = val_ds[i]
        a_logits, t_logits, v_pred = model(ids.unsqueeze(0), length.unsqueeze(0), num_feat.unsqueeze(0), num_present.unsqueeze(0))
        y_action_true.append(action.item())
        y_action_pred.append(a_logits.argmax(-1).item())
        y_target_true.append(target.item())
        y_target_pred.append(t_logits.argmax(-1).item())

cm_a = confusion_matrix(y_action_true, y_action_pred, len(ACTIONS))
plot_confusion(cm_a, ACTIONS, "Action Confusion Matrix", f"{OUT_DIR}/confusion_action.png")
cm_t = confusion_matrix(y_target_true, y_target_pred, len(TARGETS))
plot_confusion(cm_t, TARGETS, "Target Confusion Matrix", f"{OUT_DIR}/confusion_target.png")

acc_a = np.mean(np.array(y_action_true) == np.array(y_action_pred))
acc_t = np.mean(np.array(y_target_true) == np.array(y_target_pred))
print(f"action_acc={acc_a:.3f} target_acc={acc_t:.3f}")
print("saved confusion_action.png, confusion_target.png")
