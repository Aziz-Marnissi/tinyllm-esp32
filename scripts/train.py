import json
import os
import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from model import TinyIntentGRU, IntentDataset, ACTIONS, TARGETS, MAX_VALUE

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = BASE_DIR
OUT_DIR = os.path.join(BASE_DIR, "evaluation")


def confusion_matrix(true, pred, n):
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(true, pred):
        cm[t][p] += 1
    return cm


def plot_confusion(cm, labels, title, path):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def evaluate_full(model, val_ds, history, device):
    os.makedirs(OUT_DIR, exist_ok=True)
    model.eval()
    y_action_true, y_action_pred = [], []
    y_target_true, y_target_pred = [], []
    v_true, v_pred_list, v_mask = [], [], []

    with torch.no_grad():
        for i in range(len(val_ds)):
            ids, length, action, target, value, mask, num_feat, num_present = val_ds[i]
            a_logits, t_logits, v_pred = model(ids.unsqueeze(0).to(device), length.unsqueeze(0), num_feat.unsqueeze(0).to(device), num_present.unsqueeze(0).to(device))
            y_action_true.append(action.item())
            y_action_pred.append(a_logits.argmax(-1).item())
            y_target_true.append(target.item())
            y_target_pred.append(t_logits.argmax(-1).item())
            v_true.append(value.item() * MAX_VALUE)
            v_pred_list.append(v_pred.item() * MAX_VALUE)
            v_mask.append(mask.item())

    cm_a = confusion_matrix(y_action_true, y_action_pred, len(ACTIONS))
    plot_confusion(cm_a, ACTIONS, "Action Confusion Matrix", f"{OUT_DIR}/confusion_action.png")
    cm_t = confusion_matrix(y_target_true, y_target_pred, len(TARGETS))
    plot_confusion(cm_t, TARGETS, "Target Confusion Matrix", f"{OUT_DIR}/confusion_target.png")

    acc_a = np.mean(np.array(y_action_true) == np.array(y_action_pred))
    acc_t = np.mean(np.array(y_target_true) == np.array(y_target_pred))

    v_true = np.array(v_true); v_pred_arr = np.array(v_pred_list); v_mask = np.array(v_mask)
    mask_idx = v_mask == 1
    mae = np.mean(np.abs(v_true[mask_idx] - v_pred_arr[mask_idx])) if mask_idx.sum() else float("nan")

    epochs, losses, hist_a, hist_t = zip(*history)
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(epochs, losses, color="tab:red", label="train loss")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("loss", color="tab:red")
    ax2 = ax1.twinx()
    ax2.plot(epochs, hist_a, color="tab:blue", marker="o", label="val action acc")
    ax2.plot(epochs, hist_t, color="tab:green", marker="s", label="val target acc")
    ax2.set_ylabel("val accuracy")
    fig.legend(loc="lower right", bbox_to_anchor=(0.9, 0.15))
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/training_curve.png", dpi=150)
    plt.close(fig)

    with open(f"{OUT_DIR}/summary.txt", "w") as f:
        f.write(f"val_action_acc={acc_a:.4f}\n")
        f.write(f"val_target_acc={acc_t:.4f}\n")
        f.write(f"value_MAE={mae:.3f}\n")
        f.write(f"n_val_samples={len(val_ds)}\n")

    print(f"[eval] action_acc={acc_a:.3f} target_acc={acc_t:.3f} value_MAE={mae:.3f}")
    return acc_a, acc_t


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    vocab = json.load(open(os.path.join(BASE_DIR, "vocab.json")))
    train_ds = IntentDataset(os.path.join(DATA_DIR, "train.jsonl"), vocab)
    val_ds = IntentDataset(os.path.join(DATA_DIR, "val.jsonl"), vocab)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)

    model = TinyIntentGRU(vocab_size=len(vocab), dropout=0.2).to(device)

    # weight decay for regularization
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

    epochs = 300
    # cosine LR schedule: smooth decay to near-zero by the end, helps late-stage convergence
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    # label smoothing: softens hard targets, reduces overconfidence on templated phrasing,
    # generally helps generalization to unseen phrasing (our actual problem here)
    ce = nn.CrossEntropyLoss(label_smoothing=0.1)
    mse = nn.MSELoss(reduction="none")

    GRAD_CLIP = 1.0

    best_acc = -1.0
    best_state = None
    patience = 40
    epochs_no_improve = 0

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for ids, length, action, target, value, mask, num_feat, num_present in train_loader:
            ids, length, action, target, value, mask, num_feat, num_present = [t.to(device) for t in (ids, length, action, target, value, mask, num_feat, num_present)]
            opt.zero_grad()
            a_logits, t_logits, v_pred = model(ids, length, num_feat, num_present)
            loss_a = ce(a_logits, action)
            loss_t = ce(t_logits, target)
            loss_v = (mse(v_pred, value) * mask).sum() / (mask.sum() + 1e-6)
            loss = loss_a + loss_t + loss_v
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            total_loss += loss.item() * ids.size(0)
        total_loss /= len(train_ds)
        scheduler.step()

        if epoch % 5 == 0 or epoch == epochs:
            model.eval()
            correct_a = correct_t = n = 0
            with torch.no_grad():
                for ids, length, action, target, value, mask, num_feat, num_present in val_loader:
                    ids, length, action, target = ids.to(device), length.to(device), action.to(device), target.to(device)
                    a_logits, t_logits, v_pred = model(ids, length, num_feat.to(device), num_present.to(device))
                    correct_a += (a_logits.argmax(-1) == action).sum().item()
                    correct_t += (t_logits.argmax(-1) == target).sum().item()
                    n += ids.size(0)
            acc_a, acc_t = correct_a / n, correct_t / n
            history.append((epoch, total_loss, acc_a, acc_t))
            lr_now = scheduler.get_last_lr()[0]
            print(f"epoch {epoch:3d} loss {total_loss:.4f} val_action_acc {acc_a:.3f} "
                  f"val_target_acc {acc_t:.3f} lr {lr_now:.5f}")

            # track best model by combined accuracy, for early stopping / best-checkpoint saving
            combined = acc_a + acc_t
            if combined > best_acc:
                best_acc = combined
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {patience} checks)")
                break

    # restore best checkpoint before saving/evaluating
    if best_state is not None:
        model.load_state_dict(best_state)
        print("Restored best checkpoint (by val_action_acc + val_target_acc)")

    torch.save(model.state_dict(), os.path.join(BASE_DIR, "model.pt"))
    print("saved model.pt (best checkpoint)")

    model.to("cpu")
    val_ds_cpu = IntentDataset(os.path.join(DATA_DIR, "val.jsonl"), vocab)
    evaluate_full(model, val_ds_cpu, history, "cpu")

    # also evaluate on the held-out adversarial test split if present
    test_path = os.path.join(DATA_DIR, "test_adversarial.jsonl")
    if os.path.exists(test_path):
        test_ds = IntentDataset(test_path, vocab)
        model.eval()
        correct_a = correct_t = n = 0
        with torch.no_grad():
            for i in range(len(test_ds)):
                ids, length, action, target, value, mask, num_feat, num_present = test_ds[i]
                a_logits, t_logits, v_pred = model(ids.unsqueeze(0), length.unsqueeze(0), num_feat.unsqueeze(0), num_present.unsqueeze(0))
                if a_logits.argmax(-1).item() == action.item():
                    correct_a += 1
                if t_logits.argmax(-1).item() == target.item():
                    correct_t += 1
                n += 1
        print(f"[held-out test_adversarial.jsonl] action_acc={correct_a/n:.3f} target_acc={correct_t/n:.3f} (n={n})")


if __name__ == "__main__":
    main()
