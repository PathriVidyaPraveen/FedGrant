#!/usr/bin/env bash
set -euo pipefail

out_dir="data/traces"
manual_dir=""
force=false
while (($#)); do
  case "$1" in
    --from) (($# >= 2)) || { echo "--from requires a directory" >&2; exit 2; }; manual_dir="$2"; shift 2 ;;
    --output) (($# >= 2)) || { echo "--output requires a directory" >&2; exit 2; }; out_dir="$2"; shift 2 ;;
    --force) force=true; shift ;;
    -h|--help) echo "Usage: $0 [--from DIR] [--output DIR] [--force]"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

names=(Verizon-LTE-driving Verizon-LTE-short ATT-LTE-driving TMobile-UMTS-driving Verizon-EVDO-driving)
source_dir=""
source_description=""
copyright_file=""
if [[ -L "$out_dir" ]]; then
  echo "Refusing symlink output directory: $out_dir" >&2
  exit 2
fi
if [[ -e "$out_dir" && ! -d "$out_dir" ]]; then
  echo "Output path is not a directory: $out_dir" >&2
  exit 2
fi
if [[ -L "$out_dir/raw" ]]; then
  echo "Refusing symlink raw-trace directory: $out_dir/raw" >&2
  exit 2
fi
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

destination_files=(
  "$out_dir/COPYRIGHT"
  "$out_dir/SOURCE"
)
for name in "${names[@]}"; do
  destination_files+=("$out_dir/raw/$name.up")
done

check_existing_destinations() {
  local path
  existing=()
  for path in "${destination_files[@]}"; do
    if [[ -e "$path" || -L "$path" ]]; then
      existing+=("$path")
    fi
  done
  if ((${#existing[@]})); then
    echo "Refusing to overwrite existing trace outputs; use --force to replace them:" >&2
    printf '  %s\n' "${existing[@]}" >&2
    return 1
  fi
}

if [[ "$force" != true ]] && ! check_existing_destinations; then
  exit 3
fi

# Stage every replacement only after source and license validation. This keeps
# validation failures from creating or modifying destination outputs.
stage_dir=""
backup_dir=""
lock_dir=""
lock_held=false
installed_destinations=()
installed_identities=()
backup_sources=()
backup_destinations=()
rollback_needed=true
rollback_incomplete=false

path_identity() {
  stat -c '%d:%i' -- "$1"
}

rollback_installation() {
  local index
  for ((index = ${#installed_destinations[@]} - 1; index >= 0; index--)); do
    path="${installed_destinations[index]}"
    if [[ ! -e "$path" && ! -L "$path" ]]; then
      continue
    fi
    current_identity="$(path_identity "$path" 2>/dev/null || true)"
    if [[ -n "$current_identity" && "$current_identity" == "${installed_identities[index]}" ]]; then
      rm -f -- "$path" || rollback_incomplete=true
    else
      echo "Rollback refused to remove changed destination: $path" >&2
      rollback_incomplete=true
    fi
  done
  for ((index = ${#backup_sources[@]} - 1; index >= 0; index--)); do
    if [[ -e "${backup_sources[index]}" || -L "${backup_sources[index]}" ]]; then
      if [[ -e "${backup_destinations[index]}" || -L "${backup_destinations[index]}" ]]; then
        echo "Rollback refused to overwrite changed destination: ${backup_destinations[index]}" >&2
        rollback_incomplete=true
      elif ! mv -- "${backup_sources[index]}" "${backup_destinations[index]}"; then
        echo "Rollback could not restore: ${backup_destinations[index]}" >&2
        rollback_incomplete=true
      fi
    fi
  done
}

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  if [[ "$rollback_needed" == true ]]; then
    rollback_installation
  fi
  [[ -z "$stage_dir" ]] || rm -rf -- "$stage_dir"
  if [[ -n "$backup_dir" ]]; then
    if [[ "$rollback_incomplete" == true ]] || find "$backup_dir" -mindepth 1 -print -quit 2>/dev/null | grep -q .; then
      echo "Backup directory retained for recovery: $backup_dir" >&2
      backup_dir=""
    else
      rm -rf -- "$backup_dir"
    fi
  fi
  if [[ "$lock_held" == true ]]; then
    if ! rmdir -- "$lock_dir"; then
      echo "Could not remove output lock: $lock_dir" >&2
      [[ "$status" -ne 0 ]] || status=1
    fi
  fi
  return "$status"
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

mkdir -p "$out_dir"
lock_dir="$out_dir/.get-traces.lock"
if ! mkdir -- "$lock_dir" 2>/dev/null; then
  echo "Another trace installation is already using output directory: $out_dir" >&2
  exit 4
fi
lock_held=true

stage_dir="$(mktemp -d "$out_dir/.get-traces.XXXXXX")"
mkdir -p "$stage_dir/raw"
for name in "${names[@]}"; do
  cp "$source_dir/$name.up" "$stage_dir/raw/$name.up"
done
cp "$copyright_file" "$stage_dir/COPYRIGHT"

{
  printf 'source=%s\n' "$source_description"
  printf 'retrieved_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  for name in "${names[@]}"; do sha256sum "$stage_dir/raw/$name.up"; done
} > "$stage_dir/SOURCE"

mkdir -p "$out_dir/raw"
if [[ "$force" != true ]] && ! check_existing_destinations; then
  exit 3
fi

if [[ "$force" == true ]]; then
  backup_dir="$(mktemp -d "$out_dir/.get-traces-backup.XXXXXX")"
  for index in "${!destination_files[@]}"; do
    path="${destination_files[index]}"
    if [[ -e "$path" || -L "$path" ]]; then
      backup_path="$backup_dir/$index"
      backup_sources+=("$backup_path")
      backup_destinations+=("$path")
      mv -- "$path" "$backup_path"
    fi
  done
fi

install_staged_file() {
  local staged_path=$1
  local destination_path=$2
  if [[ "$force" == true ]]; then
    mv -f -- "$staged_path" "$destination_path"
  else
    mv -n -- "$staged_path" "$destination_path"
    [[ ! -e "$staged_path" ]] || return 1
  fi
  installed_destinations+=("$destination_path")
  identity=""
  if ! identity="$(path_identity "$destination_path" 2>/dev/null)"; then
    installed_identities+=("")
    return 1
  fi
  installed_identities+=("$identity")
}

for name in "${names[@]}"; do
  install_staged_file "$stage_dir/raw/$name.up" "$out_dir/raw/$name.up"
done
install_staged_file "$stage_dir/COPYRIGHT" "$out_dir/COPYRIGHT"
install_staged_file "$stage_dir/SOURCE" "$out_dir/SOURCE"
rollback_needed=false
echo "Copied five raw traces and license to $out_dir"
