# Déploiement ESP32 — Intent GRU (PFE / projet perso)

**Cible :** ESP32, via **PlatformIO**
**Approche :** export manuel des poids en C (pas de TFLite Micro) — modèle trop petit pour justifier le runtime TFLM.

## Contexte modèle
- `TinyIntentGRU` (PyTorch) : embedding(70,32) + GRU(32→64) + 3 heads (action, target, value)
- **21,576 paramètres** (~84KB float32 / ~21KB si quantifié int8)
- Bug initial corrigé dans `train.py` : `evaluate()` unpack 6 valeurs (`num_feat` inexistant) → corrigé à 5.

## Étapes réalisées

### 1. Export des poids (`export_weights.py`)
- Charge `model.pt`, dump tous les tenseurs (`embed.weight`, `gru.weight_ih_l0/weight_hh_l0`, biais, 3 heads) en `weights.h` (arrays C `float`).
- Note : gates GRU packées PyTorch = ordre **[reset, update, new]**.

### 2. Export du vocabulaire (`export_vocab.py`)
- `vocab.json` (70 mots) → `vocab.h` (table `{word, id}`).

### 3. Tokenizer C (`tokenizer.c`)
- Réplique `tokenizer.py` : lowercase, split `[a-z0-9]+`, tokens numériques → `<num>`, padding à `MAX_LEN=12`.
- Bug trouvé et corrigé : buffer mot trop petit (`buf[8]` → `buf[32]`), coupait les mots longs (ex: "brightness").
- **Validé** : sortie C = sortie Python, testé sur 4 phrases (`test_tokenizer.c` vs `compare_tokenizer.py`).

### 4. Inférence GRU C (`inference.c`)
- Forward pass complet : embedding lookup → GRU (formules r/z/n manuelles) → 3 heads (action, target, value).
- Bug trouvé et corrigé : skip du padding cassait l'équivalence avec PyTorch (le modèle original traite les 12 timesteps sans masquage, pas de `pack_padded_sequence`) → retiré le skip.
- **Validé** : logits identiques à PyTorch à ~1e-4 près, argmax et value identiques sur 4 phrases de test (`test_inference.c` vs `compare_inference.py`).

## Fichiers produits
| Fichier | Rôle |
|---|---|
| `export_weights.py` | poids modèle → `weights.h` |
| `export_vocab.py` | vocab → `vocab.h` |
| `tokenizer.c` | tokenisation texte → ids |
| `inference.c` | GRU forward + heads |
| `test_tokenizer.c` / `compare_tokenizer.py` | validation tokenizer |
| `test_inference.c` / `compare_inference.py` | validation inférence complète |

## Prochaines étapes
1. Mettre en place le projet **PlatformIO** (env ESP32)
2. Wrapper firmware : lecture commande série → `tokenize()` → `model_forward()` → action (LED/moteur/servo)
3. Benchmark latence sur ESP32 réel (`esp_timer` ou `millis()`)
4. Mesurer flash/RAM réels utilisés (poids + code)
