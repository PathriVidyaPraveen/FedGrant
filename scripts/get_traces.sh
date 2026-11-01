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
elif [[ -d /usr/share/mahimahi/traces ]]; then
  source_dir=/usr/share/mahimahi/traces
  source_description="system Mahimahi traces"
else
  tmp_dir="$(mktemp -d)"
  trap 'rm -rf "$tmp_dir"' EXIT
  repo="$tmp_dir/mahimahi"
  GIT_TERMINAL_PROMPT=0 git clone --quiet --depth 1 --filter=blob:none --sparse https://github.com/ravinet/mahimahi.git "$repo"
  git -C "$repo" sparse-checkout set traces
  if [[ -n "${MAHIMAHI_COMMIT:-}" ]]; then
    git -C "$repo" fetch --quiet --depth 1 origin "$MAHIMAHI_COMMIT"
    git -C "$repo" checkout --quiet FETCH_HEAD
  fi
  source_dir="$repo/traces"
  source_description="upstream Mahimahi repository at $(git -C "$repo" rev-parse HEAD)"
  copyright_file="$tmp_dir/COPYING"
  if git -C "$repo" cat-file -e HEAD:COPYING 2>/dev/null; then
    git -C "$repo" show HEAD:COPYING > "$copyright_file"
  elif git -C "$repo" cat-file -e HEAD:LICENSE 2>/dev/null; then
    copyright_file="$tmp_dir/LICENSE"
    git -C "$repo" show HEAD:LICENSE > "$copyright_file"
  else
    copyright_file=""
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

mkdir -p "$out_dir/raw"
for name in "${names[@]}"; do cp "$source_dir/$name.up" "$out_dir/raw/$name.up"; done
if [[ -z "$copyright_file" ]]; then
  for candidate in "$source_dir/COPYRIGHT" "$source_dir/COPYING" "$source_dir/LICENSE" \
                   "$source_dir/../COPYING" "$source_dir/../LICENSE" \
                   /usr/share/doc/mahimahi-traces/copyright /usr/share/doc/mahimahi/copyright; do
    if [[ -f "$candidate" ]]; then copyright_file="$candidate"; break; fi
  done
fi
if [[ -n "$copyright_file" && -f "$copyright_file" ]]; then
  cp "$copyright_file" "$out_dir/COPYRIGHT"
else
  echo "Could not locate the Mahimahi trace license; refusing to copy unlicensed data." >&2
  exit 1
fi

{
  printf 'source=%s\n' "$source_description"
  printf 'retrieved_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  for name in "${names[@]}"; do sha256sum "$out_dir/raw/$name.up"; done
} > "$out_dir/SOURCE"
echo "Copied five raw traces and license to $out_dir"
