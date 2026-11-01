#!/usr/bin/env bash
# Download and preprocess LEAF FEMNIST into FedGrant's ignored data/ tree.
# Usage: bash scripts/get_femnist.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="$ROOT/data/femnist/reproduced"
LEAF_ROOT="$DATA_ROOT/leaf-source"
LEAF_COMMIT="09ec454a5675e32e1f0546b456b77857fdece018"
if [[ $# -ne 0 ]]; then
    echo "This script reproduces the full FEMNIST source; it takes no arguments." >&2
    exit 2
fi
SAMPLE_FRACTION=1.0

for command in git wget unzip python3; do
    command -v "$command" >/dev/null || {
        echo "Required command not found: $command" >&2
        exit 1
    }
done

python3 - <<'PY'
import numpy, PIL, torch
print(f"Python dependencies available: numpy={numpy.__version__}, Pillow={PIL.__version__}, torch={torch.__version__}")
PY

mkdir -p "$DATA_ROOT"
if [[ ! -d "$LEAF_ROOT/.git" ]]; then
    git clone --no-checkout https://github.com/TalwalkarLab/leaf.git "$LEAF_ROOT"
fi
if [[ "$(git -C "$LEAF_ROOT" rev-parse HEAD 2>/dev/null || true)" != "$LEAF_COMMIT" ]]; then
    git -C "$LEAF_ROOT" fetch --depth 1 origin "$LEAF_COMMIT"
    git -C "$LEAF_ROOT" checkout --detach "$LEAF_COMMIT"
fi

FEMNIST_DIR="$LEAF_ROOT/data/femnist"
# The pinned LEAF converter predates Pillow 10, where Image.ANTIALIAS was removed.
python3 - "$FEMNIST_DIR/preprocess/data_to_json.py" <<'PY'
from pathlib import Path
import sys
p = Path(sys.argv[1])
text = p.read_text(encoding="utf-8")
p.write_text(text.replace("Image.ANTIALIAS", "Image.Resampling.LANCZOS"), encoding="utf-8")
PY
if ! compgen -G "$FEMNIST_DIR/data/train/*.json" >/dev/null \
    || ! compgen -G "$FEMNIST_DIR/data/test/*.json" >/dev/null; then
    (
        cd "$FEMNIST_DIR"
        ./preprocess.sh -s niid --sf "$SAMPLE_FRACTION" -k 0 -t sample \
            --smplseed 1549786595 --spltseed 1549786796
    )
fi
if ! compgen -G "$FEMNIST_DIR/data/train/*.json" >/dev/null \
    || ! compgen -G "$FEMNIST_DIR/data/test/*.json" >/dev/null; then
    echo "LEAF preprocessing did not create train and test JSON shards" >&2
    exit 1
fi

OUT_LEAF="$DATA_ROOT/leaf"
mkdir -p "$OUT_LEAF/data"
if [[ ! -d "$OUT_LEAF/data/train" || ! -d "$OUT_LEAF/data/test" ]]; then
    cp -a "$FEMNIST_DIR/data/train" "$FEMNIST_DIR/data/test" "$OUT_LEAF/data/"
fi
cat > "$DATA_ROOT/provenance.json" <<EOF
{
  "source": "LEAF FEMNIST; NIST Special Database 19",
  "source_repository": "https://github.com/TalwalkarLab/leaf.git",
  "source_commit": "$LEAF_COMMIT",
  "source_archives": [
    "https://s3.amazonaws.com/nist-srd/SD19/by_class.zip",
    "https://s3.amazonaws.com/nist-srd/SD19/by_write.zip"
  ],
  "preprocessing_version": "LEAF preprocess.sh; niid; sample_fraction=$SAMPLE_FRACTION; keep_min=0; split=sample; sample_seed=1549786595; split_seed=1549786796",
  "sample_fraction": $SAMPLE_FRACTION,
  "num_writers": 200,
  "writers_per_client": 10
}
EOF

cd "$ROOT"
python3 -m fedgrant.fl.femnist \
    --leaf-root "$OUT_LEAF/data" \
    --output-dir "$DATA_ROOT/clients" \
    --num-writers 200 --writers-per-client 10 \
    --provenance-file "$DATA_ROOT/provenance.json"
echo "FEMNIST data prepared under $DATA_ROOT"
