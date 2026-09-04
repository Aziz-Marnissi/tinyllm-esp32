import json
import random
from pathlib import Path


# ============================================================
# Dataset configuration
# ============================================================

SEED = 42
VAL_FRAC = 0.25
TEST_FRAC = 0.15  # held-out adversarial-style test split, disjoint templates from train+val


# ============================================================
# Target names
# ============================================================

LED_NAMES = ["led", "light", "the led", "the light", "lamp", "the lamp"]
MOTOR_NAMES = ["the motor", "motor", "the 28byj48 motor", "the stepper motor", "stepper motor", "28byj48", "stepper"]
SERVO_NAMES = ["the servo", "servo", "servo motor", "the servo motor"]

TARGET_NAMES = {
    "led": LED_NAMES,
    "motor_28BYJ48": MOTOR_NAMES,
    "servo": SERVO_NAMES,
}

# ============================================================
# SHARED verb templates -- these apply to ALL targets (led/motor/servo)
# This is the key fix: previously "kill", "power up", "shut down" etc.
# were only ever paired with one target during training, so the model
# learned (word, target) co-occurrence instead of verb semantics.
# ============================================================

SHARED_ON_TMPL = [
    "turn on {n}",
    "switch {n} on",
    "activate {n}",
    "{n} on",
    "please turn on {n}",
    "can you turn on {n}",
    "power up {n}",
    "enable {n}",
    "turn {n} on",
    "switch on {n}",
    "start {n}",
    "start up {n}",
    "pls turn on {n}",
    "trun on {n}",
    "i want {n} on",
    "get {n} on",
    "fire up {n}",
    "get {n} running",
    "spin up {n}",
    "engage {n}",
    "kick off {n}",
    "run {n}",
    # indirect / polite phrasing
    "would you mind switching on {n}",
    "could you turn {n} on",
    "could you possibly start {n}",
    "can you activate {n} for me",
    "i need {n} turned on",
    "let's get {n} going",
]

SHARED_OFF_TMPL = [
    "turn off {n}",
    "switch {n} off",
    "shut down {n}",
    "deactivate {n}",
    "{n} off",
    "please turn off {n}",
    "can you turn off {n}",
    "power down {n}",
    "disable {n}",
    "turn {n} off",
    "switch off {n}",
    "stop {n}",
    "pls turn off {n}",
    "trun off {n}",
    "i want {n} off",
    "get {n} off",
    "kill {n}",
    "halt {n}",
    "disengage {n}",
    "cut off {n}",
    "cut power to {n}",
    "shut {n} down",
    # indirect / polite phrasing
    "would you mind turning off {n}",
    "could you switch {n} off",
    "could you possibly stop {n}",
    "can you deactivate {n} for me",
    "i need {n} stopped",
    "i need {n} turned off",
]

LED_POWER_TMPL = [
    "set light to {v} percent",
    "set led power to {v}",
    "dim led to {v} percent",
    "led brightness {v}",
    "set the light brightness to {v}",
    "make the led {v} percent",
    "brightness {v}",
    "set {n} to {v}%",
    "dim {n} to {v}",
    "increase {n} brightness to {v}",
    "put {n} at {v} percent",
    "adjust led power to {v}",
    "led at {v}%",
    "set light level to {v}",
    "brighten {n} to {v} percent",
    # indirect / polite phrasing
    "could you possibly dim {n} a bit to {v}",
    "can u dim {n} to like {v} percent",
    "crank the brightness up on {n} to {v}",
    "lower {n} brightness to {v}",
    "make {n} brighter, say {v}",
]

SERVO_DEG_TMPL = [
    "turn the servo to {v} degrees",
    "set servo angle to {v}",
    "move servo to {v} degrees",
    "rotate servo {v} degrees",
    "servo to {v}",
    "set the servo at {v} degrees",
    "move {n} to {v}",
    "point {n} to {v} degrees",
    "servo angle {v}",
    "rotate {n} to {v} deg",
    "set {n} to {v} degree",
    "adjust servo to {v} degrees",
    "servo go to {v}",
    "put the servo at {v} degrees",
    # indirect / polite phrasing
    "i want the servo to go to {v} degrees please",
    "set the servo angle to about {v}",
    "could you rotate {n} to {v}",
    "move {n} to {v} degrees please",
]


# ============================================================
# Natural language variation
# ============================================================

CASUAL_PREFIX = ["", "", "", "hey ", "ok ", "please ", "yo ", "hmm "]
CASUAL_SUFFIX = ["", "", "", " now", " please", " thanks", " asap"]


def wrap(text, rng):
    text = rng.choice(CASUAL_PREFIX) + text
    text = text + rng.choice(CASUAL_SUFFIX)
    return text.strip()


def split_templates_3way(templates, val_frac, test_frac, rng):
    """
    Split templates into train / val / test.
    Val and test both hold out template WORDING the model never trains on,
    so both measure generalization -- test is meant to be the harder,
    more adversarial-feeling split (kept separate from val for reporting).
    """
    templates = list(templates)
    rng.shuffle(templates)

    n = len(templates)
    n_test = max(1, int(n * test_frac))
    n_val = max(1, int(n * val_frac))

    test_templates = templates[:n_test]
    val_templates = templates[n_test:n_test + n_val]
    train_templates = templates[n_test + n_val:]

    # Guard against starving train if template list is small
    if not train_templates:
        train_templates = templates[-1:]

    return train_templates, val_templates, test_templates


def gen_binary(names, on_templates, off_templates, target, n_each, rng):
    rows = []
    for _ in range(n_each):
        name = rng.choice(names)
        template = rng.choice(on_templates)
        text = wrap(template.format(n=name), rng)
        rows.append((text, "on", target, None))
    for _ in range(n_each):
        name = rng.choice(names)
        template = rng.choice(off_templates)
        text = wrap(template.format(n=name), rng)
        rows.append((text, "off", target, None))
    return rows


def gen_value(names, templates, action, target, vmin, vmax, count, rng):
    rows = []
    for _ in range(count):
        value = rng.randint(vmin, vmax)
        name = rng.choice(names)
        template = rng.choice(templates)
        text = wrap(template.format(n=name, v=value), rng)
        rows.append((text, action, target, value))
    return rows


def dedupe(rows):
    seen = set()
    output = []
    for row in rows:
        if row[0] not in seen:
            seen.add(row[0])
            output.append(row)
    return output


def build_dataset(rng, val_frac=VAL_FRAC, test_frac=TEST_FRAC):
    train_rows, val_rows, test_rows = [], [], []

    # --- binary ON/OFF, shared verb templates across all 3 targets ---
    binary_groups = [
        (LED_NAMES, "led", 400),
        (MOTOR_NAMES, "motor_28BYJ48", 400),
        (SERVO_NAMES, "servo", 300),
    ]

    on_train, on_val, on_test = split_templates_3way(SHARED_ON_TMPL, val_frac, test_frac, rng)
    off_train, off_val, off_test = split_templates_3way(SHARED_OFF_TMPL, val_frac, test_frac, rng)

    for names, target, count in binary_groups:
        train_rows.extend(gen_binary(names, on_train, off_train, target, count, rng))
        val_rows.extend(gen_binary(names, on_val, off_val, target, max(30, count // 4), rng))
        test_rows.extend(gen_binary(names, on_test, off_test, target, max(20, count // 6), rng))

    # --- numeric commands (target-specific, since power% only applies to led, degrees only to servo) ---
    value_groups = [
        (LED_NAMES, LED_POWER_TMPL, "set_power", "led", 0, 100, 600),
        (SERVO_NAMES, SERVO_DEG_TMPL, "set_degree", "servo", 0, 180, 600),
    ]

    for names, templates, action, target, vmin, vmax, count in value_groups:
        train_t, val_t, test_t = split_templates_3way(templates, val_frac, test_frac, rng)
        train_rows.extend(gen_value(names, train_t, action, target, vmin, vmax, count, rng))
        val_rows.extend(gen_value(names, val_t, action, target, vmin, vmax, max(40, count // 4), rng))
        test_rows.extend(gen_value(names, test_t, action, target, vmin, vmax, max(25, count // 6), rng))

    return train_rows, val_rows, test_rows


def dump_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for text, action, target, value in rows:
            record = {"text": text, "action": action, "target": target, "value": value}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    print("Generating dataset (v2: shared verb templates + indirect phrasing + held-out test split)...")

    rng = random.Random(SEED)
    project_dir = Path(__file__).resolve().parent

    train_path = project_dir / "train.jsonl"
    val_path = project_dir / "val.jsonl"
    test_path = project_dir / "test_adversarial.jsonl"

    train, val, test = build_dataset(rng, VAL_FRAC, TEST_FRAC)

    train = dedupe(train)
    val = dedupe(val)
    test = dedupe(test)

    # Prevent exact text overlap across all three splits
    train_texts = {row[0] for row in train}
    val = [row for row in val if row[0] not in train_texts]
    val_texts = {row[0] for row in val}
    test = [row for row in test if row[0] not in train_texts and row[0] not in val_texts]

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    dump_jsonl(train_path, train)
    dump_jsonl(val_path, val)
    dump_jsonl(test_path, test)

    print()
    print("Dataset generated successfully!")
    print("--------------------------------")
    print(f"Train examples : {len(train)}")
    print(f"Val examples   : {len(val)}")
    print(f"Test examples  : {len(test)}  (held-out template wording, harder split)")
    print()
    print(f"Train file : {train_path}")
    print(f"Val file   : {val_path}")
    print(f"Test file  : {test_path}")
    print()
    print("Actions (train):")
    actions = {}
    for _, action, _, _ in train:
        actions[action] = actions.get(action, 0) + 1
    for action in sorted(actions):
        print(f"  {action:12s}: {actions[action]}")


if __name__ == "__main__":
    main()
