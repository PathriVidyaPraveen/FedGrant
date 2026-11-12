#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="$ROOT/data/femnist"
LEAF_COMMIT="09ec454a5675e32e1f0546b456b77857fdece018"
PREPROCESS_ARGS=(-s niid -sf 0.1 -k 100 -t sample --smplseed 1549786595 --spltseed 1549786796)
OUTPUT_ROOT="${1:-$DATA_ROOT/reproduced}"

# Never replace an existing dataset. Pick another output path to keep both.
if [[ -e "$OUTPUT_ROOT" ]]; then
  echo "Refusing to overwrite existing output: $OUTPUT_ROOT" >&2
  exit 2
fi

mkdir -p "$DATA_ROOT"
KEEP_BUILD=false
if [[ -n "${FEMNIST_BUILD_ROOT:-}" ]]; then
  BUILD_ROOT="$FEMNIST_BUILD_ROOT"
  case "$BUILD_ROOT" in
    "$DATA_ROOT"/.femnist-build.*) ;;
    *) echo "FEMNIST_BUILD_ROOT must be a .femnist-build.* path under $DATA_ROOT" >&2; exit 2 ;;
  esac
  [[ -d "$BUILD_ROOT" ]] || { echo "Build directory not found: $BUILD_ROOT" >&2; exit 2; }
  KEEP_BUILD=true
else
  BUILD_ROOT="$(mktemp -d "$DATA_ROOT/.femnist-build.XXXXXX")"
fi
trap 'if [[ "$KEEP_BUILD" == false && "$BUILD_OK" == true ]]; then rm -rf -- "$BUILD_ROOT"; fi' EXIT
BUILD_OK=false
LEAF_ROOT="$BUILD_ROOT/leaf-src"
OUT="$BUILD_ROOT/output"

if [[ ! -d "$LEAF_ROOT/.git" ]]; then
  git clone --depth 1 --filter=blob:none --no-checkout --sparse \
    https://github.com/TalwalkarLab/leaf.git "$LEAF_ROOT"
fi
git -C "$LEAF_ROOT" fetch --depth 1 origin "$LEAF_COMMIT"
git -C "$LEAF_ROOT" checkout --detach "$LEAF_COMMIT"
git -C "$LEAF_ROOT" sparse-checkout set data/femnist data/utils
ACTUAL_COMMIT="$(git -C "$LEAF_ROOT" rev-parse HEAD)"
if [[ "$ACTUAL_COMMIT" != "$LEAF_COMMIT" ]]; then
  echo "LEAF checkout mismatch: expected $LEAF_COMMIT, got $ACTUAL_COMMIT" >&2
  exit 1
fi

# Pillow 10 removed this alias; keep the pinned LEAF logic and apply this
# compatibility-only source edit to the temporary checkout.
sed -i 's/Image\.ANTIALIAS/Image.Resampling.LANCZOS/g' \
  "$LEAF_ROOT/data/femnist/preprocess/data_to_json.py"
# Make extraction non-interactive so a partial prior extraction is completed
# instead of silently accepting unzip's EOF response and deleting the archive.
sed -i -E \
  -e 's@^wget .*by_class\.zip.*$@wget -c -q https://s3.amazonaws.com/nist-srd/SD19/by_class.zip@' \
  -e 's@^wget .*by_write\.zip.*$@wget -c -q https://s3.amazonaws.com/nist-srd/SD19/by_write.zip@' \
  -e 's@^unzip .*by_class\.zip.*$@unzip -oq by_class.zip@' \
  -e 's@^unzip .*by_write\.zip.*$@unzip -oq by_write.zip@' \
  "$LEAF_ROOT/data/femnist/preprocess/get_data.sh"
sed -i 's@if \[ ! -d "../data/raw_data" \]; then@if [ ! -d "../data/raw_data/by_class" ] || [ ! -d "../data/raw_data/by_write" ]; then@' \
  "$LEAF_ROOT/data/femnist/preprocess/data_to_json.sh"
sed -i 's@mkdir ../data/raw_data@mkdir -p ../data/raw_data@' \
  "$LEAF_ROOT/data/femnist/preprocess/data_to_json.sh"
for script in \
  "$LEAF_ROOT/data/femnist/preprocess/data_to_json.sh" \
  "$LEAF_ROOT/data/femnist/preprocess.sh"; do
  if ! grep -q '^set -euo pipefail$' "$script"; then
    sed -i '2i set -euo pipefail' "$script"
  fi
done

(
  cd "$LEAF_ROOT/data/femnist"
  ./preprocess.sh "${PREPROCESS_ARGS[@]}"
)

cd "$ROOT"
mkdir -p "$OUT/leaf/data/train" "$OUT/leaf/data/test" "$OUT/clients"
cp -a "$LEAF_ROOT/data/femnist/data/train/." "$OUT/leaf/data/train/"
cp -a "$LEAF_ROOT/data/femnist/data/test/." "$OUT/leaf/data/test/"

cat > "$OUT/provenance.json" <<EOF
{
  "source": "https://github.com/TalwalkarLab/leaf",
  "preprocessing_version": "fedgrant-femnist-v1",
  "leaf_commit": "$LEAF_COMMIT",
  "preprocess_args": ["-s", "niid", "-sf", "0.1", "-k", "100", "-t", "sample", "--smplseed", "1549786595", "--spltseed", "1549786796"],
  "compatibility_patches": ["Pillow Image.ANTIALIAS replaced with Image.Resampling.LANCZOS", "LEAF extraction is non-interactive and safely resumes partial extracts", "LEAF download checks for extracted folders so partial downloads can resume", "LEAF raw-data folder creation is idempotent", "LEAF preprocessing wrappers fail immediately on errors"],
  "writer_selection": "first 200 writer IDs in lexicographic order from train and test output",
  "client_grouping": "20 clients; 10 consecutive sorted writers per client",
  "image_format": "28x28 grayscale, float32, [0,1]; no augmentation"
}
EOF

"${PYTHON:-$ROOT/.venv/bin/python}" -m fedgrant.fl.femnist \
  --leaf-root "$OUT/leaf/data" \
  --output-dir "$OUT/clients" \
  --num-writers 200 \
  --writers-per-client 10 \
  --provenance-file "$OUT/provenance.json"

mkdir -p "$(dirname "$OUTPUT_ROOT")"
mv -- "$OUT" "$OUTPUT_ROOT"
BUILD_OK=true
echo "FEMNIST data prepared under $OUTPUT_ROOT"
