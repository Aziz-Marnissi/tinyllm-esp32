import json
import random

ACTIONS = ["on", "off", "set_power", "set_degree"]
TARGETS = ["led", "motor_28BYJ48", "servo"]

LED_NAMES = ["led", "light", "the led", "the light", "lamp", "the lamp"]
LED_ON_TMPL = [
    "turn on {n}", "switch {n} on", "light up {n}", "activate {n}", "{n} on",
    "please turn on {n}", "can you turn on {n}", "power up {n}", "enable {n}",
    "turn {n} on", "switch on {n}", "start {n}", "pls turn on {n}", "trun on {n}",
    "i want {n} on", "get {n} on", "fire up {n}",
]
LED_OFF_TMPL = [
    "turn off {n}", "switch {n} off", "shut down {n}", "deactivate {n}", "{n} off",
    "please turn off {n}", "can you turn off {n}", "power down {n}", "disable {n}",
    "turn {n} off", "switch off {n}", "stop {n}", "pls turn off {n}", "trun off {n}",
    "i want {n} off", "get {n} off", "kill {n}",
]
LED_POWER_TMPL = [
    "set light to {v} percent", "set led power to {v}", "dim led to {v} percent",
    "led brightness {v}", "set the light brightness to {v}", "make the led {v} percent",
    "brightness {v}", "set {n} to {v}%", "dim {n} to {v}", "increase {n} brightness to {v}",
    "put {n} at {v} percent", "adjust led power to {v}", "led at {v}%",
    "set light level to {v}", "brighten {n} to {v} percent",
]

MOTOR_NAMES = ["the motor", "motor", "the 28byj48 motor", "the stepper motor", "stepper motor", "28byj48"]
MOTOR_ON_TMPL = [
    "turn on {n}", "start {n}", "activate {n}", "{n} on", "run {n}",
    "please start {n}", "can you start {n}", "spin up {n}", "engage {n}",
    "get {n} running", "pls start {n}", "kick off {n}",
]
MOTOR_OFF_TMPL = [
    "turn off {n}", "stop {n}", "deactivate {n}", "{n} off", "halt {n}",
    "please stop {n}", "can you stop {n}", "shut down {n}", "disengage {n}",
    "pls stop {n}", "cut off {n}",
]

SERVO_NAMES = ["the servo", "servo", "servo motor", "the servo motor"]
SERVO_ON_TMPL = ["turn on {n}", "activate {n}", "{n} on", "enable {n}", "power up {n}", "start {n}"]
SERVO_OFF_TMPL = ["turn off {n}", "deactivate {n}", "{n} off", "disable {n}", "power down {n}", "stop {n}"]
SERVO_DEG_TMPL = [
    "turn the servo to {v} degrees", "set servo angle to {v}", "move servo to {v} degrees",
    "rotate servo {v} degrees", "servo to {v}", "set the servo at {v} degrees",
    "move {n} to {v}", "point {n} to {v} degrees", "servo angle {v}",
    "rotate {n} to {v} deg", "set {n} to {v} degree", "adjust servo to {v} degrees",
    "servo go to {v}", "put the servo at {v} degrees",
]

CASUAL_PREFIX = ["", "", "", "hey ", "ok ", "please ", "yo ", "hmm "]
CASUAL_SUFFIX = ["", "", "", " now", " please", " thanks", " asap"]


def wrap(s, rng):
    s = rng.choice(CASUAL_PREFIX) + s + rng.choice(CASUAL_SUFFIX)
    return s.strip()


def split_templates(tmpl_list, val_frac, rng):
    tmpl_list = list(tmpl_list)
    rng.shuffle(tmpl_list)
    n_val = max(1, int(len(tmpl_list) * val_frac))
    return tmpl_list[n_val:], tmpl_list[:n_val]  # train_tmpl, val_tmpl


def gen_binary(names, on_tmpl, off_tmpl, action_on, action_off, target, n_each, rng):
    out = []
    for _ in range(n_each):
        n = rng.choice(names)
        t = rng.choice(on_tmpl).format(n=n)
        out.append((wrap(t, rng), action_on, target, None))
    for _ in range(n_each):
        n = rng.choice(names)
        t = rng.choice(off_tmpl).format(n=n)
        out.append((wrap(t, rng), action_off, target, None))
    return out


def gen_value(names, tmpl, action, target, vmin, vmax, n, rng):
    out = []
    for _ in range(n):
        v = rng.randint(vmin, vmax)
        n_ = rng.choice(names)
        t = rng.choice(tmpl).format(n=n_, v=v)
        out.append((wrap(t, rng), action, target, v))
    return out


def build(rng, val_frac=0.25):
    """Split TEMPLATES (not samples) into train/val so val phrasing patterns
    are never seen during training -> real generalization test."""
    train_rows, val_rows = [], []

    groups = [
        (LED_NAMES, LED_ON_TMPL, LED_OFF_TMPL, "on", "off", "led", 140),
        (MOTOR_NAMES, MOTOR_ON_TMPL, MOTOR_OFF_TMPL, "on", "off", "motor_28BYJ48", 140),
        (SERVO_NAMES, SERVO_ON_TMPL, SERVO_OFF_TMPL, "on", "off", "servo", 100),
    ]
    for names, on_t, off_t, a_on, a_off, target, n_each in groups:
        on_train, on_val = split_templates(on_t, val_frac, rng)
        off_train, off_val = split_templates(off_t, val_frac, rng)
        train_rows += gen_binary(names, on_train, off_train, a_on, a_off, target, n_each, rng)
        val_rows += gen_binary(names, on_val, off_val, a_on, a_off, target, max(10, n_each // 4), rng)

    value_groups = [
        (LED_NAMES, LED_POWER_TMPL, "set_power", "led", 0, 100, 220),
        (SERVO_NAMES, SERVO_DEG_TMPL, "set_degree", "servo", 0, 180, 220),
    ]
    for names, tmpl, action, target, vmin, vmax, n in value_groups:
        t_train, t_val = split_templates(tmpl, val_frac, rng)
        train_rows += gen_value(names, t_train, action, target, vmin, vmax, n, rng)
        val_rows += gen_value(names, t_val, action, target, vmin, vmax, max(15, n // 4), rng)

    return train_rows, val_rows


def dedupe(rows):
    seen, out = set(), []
    for r in rows:
        if r[0] not in seen:
            seen.add(r[0])
            out.append(r)
    return out


def main():
    rng = random.Random(42)
    train, val = build(rng)
    train, val = dedupe(train), dedupe(val)
    # remove any val text that also appears in train (paraphrase collisions)
    train_texts = {r[0] for r in train}
    val = [r for r in val if r[0] not in train_texts]
    rng.shuffle(train)
    rng.shuffle(val)

    def dump(path, rows):
        with open(path, "w") as f:
            for text, action, target, value in rows:
                f.write(json.dumps({
                    "text": text, "action": action, "target": target, "value": value
                }) + "\n")

    dump("/home/claude/tinyllm/train.jsonl", train)
    dump("/home/claude/tinyllm/val.jsonl", val)
    print(f"train={len(train)} val={len(val)} (val templates unseen in train)")

if __name__ == "__main__":
    main()
