import json
from tokenizer import tokenize

vocab = json.load(open("vocab.json"))

tests = [
    "turn on the led",
    "set servo to 90 degrees",
    "trun off motor",
    "brightness 45",
]

for s in tests:
    ids = [vocab.get(tok, 1) for tok in tokenize(s)][:12]
    ids += [0] * (12 - len(ids))
    print(f'"{s}" -> {ids}')
