import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tokenizer import tokenize, extract_number

ACTIONS = ["on", "off", "set_power", "set_degree", "set_timer", "check_status"]
TARGETS = ["led", "motor_28BYJ48", "servo", "timer", "temp_sensor"]
ACTION2ID = {a: i for i, a in enumerate(ACTIONS)}
TARGET2ID = {t: i for i, t in enumerate(TARGETS)}
MAX_VALUE = 180.0


class IntentDataset(Dataset):
    def __init__(self, path, vocab, max_len=12):
        self.rows = [json.loads(l) for l in open(path)]
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def encode(self, text):
        ids = [self.vocab.get(tok, 1) for tok in tokenize(text)][: self.max_len]
        length = len(ids)  # real length BEFORE padding -- needed so the
                            # backward GRU direction doesn't start on pad tokens
        ids += [0] * (self.max_len - len(ids))
        return ids, length

    def __getitem__(self, idx):
        row = self.rows[idx]
        ids, length = self.encode(row["text"])
        ids = torch.tensor(ids, dtype=torch.long)
        length = torch.tensor(length, dtype=torch.long)
        action = torch.tensor(ACTION2ID[row["action"]], dtype=torch.long)
        target = torch.tensor(TARGET2ID[row["target"]], dtype=torch.long)
        has_value = row["value"] is not None
        value = torch.tensor((row["value"] or 0) / MAX_VALUE, dtype=torch.float32)
        mask = torch.tensor(1.0 if has_value else 0.0, dtype=torch.float32)
        # raw number parsed straight from text -- the <num> token discards
        # this, so the value head gets it through a side channel instead.
        num = extract_number(row["text"])
        num_feat = torch.tensor((num or 0) / MAX_VALUE, dtype=torch.float32)
        num_present = torch.tensor(1.0 if num is not None else 0.0, dtype=torch.float32)
        return ids, length, action, target, value, mask, num_feat, num_present


class TinyIntentGRU(nn.Module):
    """
    Bidirectional GRU: reads the sequence forward AND backward, then
    concatenates both final hidden states before the output heads.
    This lets disambiguating words that come AFTER the verb (e.g.
    "crank the brightness UP") still inform the action prediction,
    which a forward-only GRU cannot see until it's too late.
    hidden=96 (up from 64) gives more capacity; ESP32 has plenty of
    headroom (was at ~7% RAM, ~23% flash).
    """
    def __init__(self, vocab_size, emb_dim=32, hidden=96, dropout=0.2):
        super().__init__()
        self.hidden = hidden
        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.emb_drop = nn.Dropout(dropout)
        self.gru = nn.GRU(emb_dim, hidden, batch_first=True, bidirectional=True)
        self.h_drop = nn.Dropout(dropout)
        # concatenated fwd+bwd hidden -> 2*hidden into the heads
        self.action_head = nn.Linear(hidden * 2, len(ACTIONS))
        self.target_head = nn.Linear(hidden * 2, len(TARGETS))
        # value head takes h_cat PLUS the raw parsed number (num_feat) and
        # a presence flag -- the GRU input only ever sees the generic <num>
        # token, so without this side channel the value head has no way to
        # know the actual number that was said.
        self.value_head = nn.Linear(hidden * 2 + 2, 1)

    def forward(self, x, lengths, num_feat, num_present):
        emb = self.emb_drop(self.embed(x))
        # pack so the backward direction skips padding instead of starting on it
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.gru(packed)       # h: [2, batch, hidden] (fwd, bwd)
        h_cat = torch.cat([h[0], h[1]], dim=-1)  # [batch, 2*hidden]
        h_cat = self.h_drop(h_cat)
        v_in = torch.cat([h_cat, num_feat.unsqueeze(-1), num_present.unsqueeze(-1)], dim=-1)
        return (self.action_head(h_cat), self.target_head(h_cat),
                torch.sigmoid(self.value_head(v_in)).squeeze(-1))
