import re, sys

path = sys.argv[1] if len(sys.argv) > 1 else "int8_results.txt"

total = 0
action_correct = 0
target_correct = 0
latencies = []

with open(path) as f:
    for line in f:
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 4:
            continue
        text, resp, exp_action, exp_target = parts[0], parts[1], parts[2], parts[3]
        if not resp:
            continue
        m = re.search(r"action=(\S+) target=(\S+) value=([\d.]+)\s+\(([\d.]+) ms\)", resp)
        if not m:
            continue
        pred_action, pred_target, pred_value, latency = m.groups()
        total += 1
        if pred_action == exp_action:
            action_correct += 1
        if pred_target == exp_target:
            target_correct += 1
        latencies.append(float(latency))

if total == 0:
    print("No valid rows parsed.")
else:
    print(f"n={total}")
    print(f"action_acc={action_correct/total:.3f}")
    print(f"target_acc={target_correct/total:.3f}")
    print(f"latency_avg_ms={sum(latencies)/len(latencies):.2f}")
    print(f"latency_min_ms={min(latencies):.2f}  latency_max_ms={max(latencies):.2f}")
