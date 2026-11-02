#!/usr/bin/env bash
set -euo pipefail

out_dir="data/traces"
manual_dir=""
while (($#)); do
  case "$1" in
    --from) (($# >= 2)) || { echo "--from requires a directory" >&2; exit 2; }; manual_dir="$2"; shift 2 ;;
    --output) (($# >= 2)) || { echo "--output requires a directory" >&2; exit 2; }; out_dir="$2"; shift 2 ;;
    -h|--help) echo "Usage: $0 [--from DIR] [--output DIR]"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

names=(Verizon-LTE-driving Verizon-LTE-short ATT-LTE-driving TMobile-UMTS-driving Verizon-EVDO-driving)
source_dir=""
source_description=""
copyright_file=""
if [[ -n "$manual_dir" ]]; then
  source_dir="$manual_dir"
  source_description="manual source: $manual_dir"
elif [[ -d "$out_dir/raw" && -f "$out_dir/raw/Verizon-LTE-driving.up" ]]; then
  source_dir="$out_dir/raw"
  source_description="existing repository-local raw traces"
elif [[ -d /usr/share/mahimahi/traces ]]; then
  source_dir=/usr/share/mahimahi/traces
  source_description="system Mahimahi traces"
else
  # Prefer already downloaded, repository-local trace files. This step is
  # intentionally offline: fetch raw traces separately with a licensed source
  # and pass it via --from, or install the Mahimahi trace package.
  if [[ -d "$out_dir/raw" ]]; then
    source_dir="$out_dir/raw"
    source_description="existing repository-local raw traces"
  else
    echo "No local trace source found. Install Mahimahi traces or run: $0 --from /path/to/traces" >&2
    exit 2
  fi
fi

missing=()
for name in "${names[@]}"; do
  [[ -f "$source_dir/$name.up" ]] || missing+=("$name.up")
done
if ((${#missing[@]})); then
  printf 'Missing required traces in %s:\n' "$source_dir" >&2
  printf '  %s\n' "${missing[@]}" >&2
  printf 'Available files:\n' >&2
  find "$source_dir" -maxdepth 1 -type f -printf '  %f\n' 2>/dev/null | sort >&2 || true
  exit 1
fi

# Resolve the license before copying anything, so a failed provenance check
# cannot leave a partial raw-trace directory behind. Repositories commonly
# keep COPYRIGHT beside the raw/ subdirectory.
if [[ -z "$copyright_file" ]]; then
  for candidate in "$source_dir/COPYRIGHT" "$source_dir/COPYING" "$source_dir/LICENSE" \
                   "$source_dir/../COPYRIGHT" "$source_dir/../COPYING" "$source_dir/../LICENSE" \
                   /usr/share/doc/mahimahi-traces/copyright /usr/share/doc/mahimahi/copyright; do
    if [[ -f "$candidate" ]]; then copyright_file="$candidate"; break; fi
  done
fi
if [[ -z "$copyright_file" || ! -f "$copyright_file" ]]; then
  echo "Could not locate the Mahimahi trace license; refusing to copy unlicensed data." >&2
  exit 1
fi

mkdir -p "$out_dir/raw"
for name in "${names[@]}"; do
  [[ "$source_dir/$name.up" -ef "$out_dir/raw/$name.up" ]] || cp "$source_dir/$name.up" "$out_dir/raw/$name.up"
done
cp "$copyright_file" "$out_dir/COPYRIGHT"

{
  printf 'source=%s\n' "$source_description"
  printf 'retrieved_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  for name in "${names[@]}"; do sha256sum "$out_dir/raw/$name.up"; done
} > "$out_dir/SOURCE"
echo "Copied five raw traces and license to $out_dir"
