#!/bin/bash
set -e

TINYLLM=~/tinyllm
ESP32=~/esp32_intent

echo "=== 1. Ship hybrid variant on ESP32 ==="
cp "$TINYLLM/backups/inference_hybrid.c.bak" "$ESP32/src/inference.c"
echo "Copied hybrid inference.c into $ESP32/src/"

echo ""
echo "=== 2. Remove stray host-test binaries ==="
rm -f "$TINYLLM/test_inf_bin" "$TINYLLM/eval_bin" "$TINYLLM/test_inf" \
      "$TINYLLM/test_inference" "$TINYLLM/test_tok" "$TINYLLM/test_tokenizer"
echo "Removed test binaries"

echo ""
echo "=== 3. Verify weights.h in ESP32 matches int8/hybrid expectations ==="
if grep -q '"weights.h"' "$ESP32/src/inference.c"; then
    if [ ! -f "$ESP32/src/weights.h" ]; then
        echo "WARNING: $ESP32/src/weights.h missing — copying from tinyllm/src/"
        cp "$TINYLLM/src/weights.h" "$ESP32/src/weights.h"
    else
        echo "weights.h already present in $ESP32/src/"
    fi
fi

echo ""
echo "=== 4. Remove now-unneeded ESP32 float weights (hybrid doesn't use them) ==="
rm -f "$ESP32/src/weights_float.h"
echo "Removed weights_float.h from ESP32 src (not needed for hybrid)"

echo ""
echo "=== 5. Tidy tinyllm root: keep backups for reference, remove build junk ==="
rm -rf "$TINYLLM/__pycache__"
find "$TINYLLM" -name "*.o" -delete
echo "Removed __pycache__ and stray .o files"

echo ""
echo "=== 6. Summary of what remains ==="
echo "--- esp32_intent/src/ ---"
ls -la "$ESP32/src/"
echo ""
echo "--- tinyllm/backups/ (kept for reference/comparison) ---"
ls -la "$TINYLLM/backups/"
echo ""
echo "Done. Rebuild + reflash to confirm hybrid still runs correctly:"
echo "  cd $ESP32 && pio run -t upload -t monitor"
