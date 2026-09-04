import json

vocab = json.load(open("vocab.json"))

lines = []
lines.append("// Auto-generated from vocab.json -- do not edit by hand\n")
lines.append(f"#define VOCAB_N {len(vocab)}\n\n")
lines.append("typedef struct { const char* word; int id; } VocabEntry;\n\n")
lines.append(f"static const VocabEntry VOCAB[VOCAB_N] = {{\n")
for word, idx in vocab.items():
    safe = word.replace("\\", "\\\\").replace('"', '\\"')
    lines.append(f'  {{"{safe}", {idx}}},\n')
lines.append("};\n")

with open("vocab.h", "w") as f:
    f.writelines(lines)

print("wrote vocab.h,", len(vocab), "entries")
