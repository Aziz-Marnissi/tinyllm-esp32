import json
import re

SPECIAL = ["<pad>", "<unk>", "<num>"]

def tokenize(text):
    text = text.lower()
    toks = re.findall(r"[a-z0-9]+", text)
    return ["<num>" if tok.isdigit() else tok for tok in toks]

def extract_number(text):
    """Return the first standalone number found in text, or None.
    This is the raw value the <num> token discards during tokenize()."""
    m = re.search(r"\d+", text)
    return float(m.group()) if m else None

def build_vocab(paths):
    vocab = {}
    for p in paths:
        with open(p) as f:
            for line in f:
                row = json.loads(line)
                for tok in tokenize(row["text"]):
                    if tok not in vocab and tok not in SPECIAL:
                        vocab[tok] = len(vocab) + len(SPECIAL)
    full = {t: i for i, t in enumerate(SPECIAL)}
    full.update(vocab)
    return full

if __name__ == "__main__":
    import os
    base_dir = os.path.dirname(os.path.abspath(__file__))
    vocab = build_vocab([
        os.path.join(base_dir, "train.jsonl"),
        os.path.join(base_dir, "val.jsonl"),
    ])
    with open(os.path.join(base_dir, "vocab.json"), "w") as f:
        json.dump(vocab, f, indent=2)
    print("vocab size:", len(vocab))
