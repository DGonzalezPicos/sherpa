#!/usr/bin/env bash
#
# Upload petitRADTRANS line-by-line opacity files to a private SURFdrive folder
# via WebDAV (curl), without rclone.
#
# Credentials: place your SURFdrive WebDAV username and app password in
#   .surfdrive  (two lines: username, password)
#
# For multi-hour transfers (~140 GB for the default species list), run inside tmux:
#   tmux new -s surfupload
#   ./upload_opacity_files_to_surf.sh
#   # detach with Ctrl-B, D; reattach with: tmux attach -t surfupload
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CREDENTIALS_FILE="${SCRIPT_DIR}/.surfdrive"
COOKIE_JAR="${TMPDIR:-/tmp}/surfdrive_upload_cookies.txt"

# --- configuration -----------------------------------------------------------
OPACITY_PATH="/data2/pRT3/input_data/opacities/lines/line_by_line"
# Private SURFdrive parent folder. Files are stored as:
#   ${REMOTE_BASE}/line_by_line/<molecule>/<isotopologue>/<file>.h5
# which mirrors the petitRADTRANS layout under opacities/lines/.
REMOTE_BASE="sherpa_opacities"
SURFDRIVE_HOST="https://surfdrive.surf.nl"

# Species to upload: "name linelist" (linelist = pRT line_species from species_info.txt)
read -r -d '' SPECIES_LIST <<'EOF' || true
H2O H2-16O__pokazatel-Sam-new.R1e6_0.3-28mu
H2O_181 1H2-18O__HITEMP.R1e6_0.3-28mu
CO 12C-16O__high-Sam.R1e6_0.3-28mu
13CO 13C-16O__high-Sam.R1e6_0.3-28mu
C18O 12C-18O__high-Sam.R1e6_0.3-28mu
C17O 12C-17O__HITRAN.R1e6_0.3-28mu
CO2 12C-16O2__exomol-UCL4000.R1e+06_0.3-50.0mu
SiO 28Si-16O__exomol.R1e+06_0.3-50.0mu
H2S 1H2-32S__Sid.R1e6_0.3-28mu
CN 12C-14N__high.R1e6_0.3-28mu
OH 16O-1H__exomol.R1e+06_0.3-50.0mu
HF 1H-19F__Coxon-Hajig.R1e+06_0.3-28.0mu
Fe 56Fe__Kurucz-high.R1e+06_0.3-50.0mu
K 39K__Kurucz.R1e+06_0.3-50.0mu
Na 23Na__Kurucz-high.R1e+06_0.3-50.0mu
Ca 40Ca__Kurucz-high.R1e+06_0.3-50.0mu
Si 28Si__Kurucz-high.R1e+06_0.3-50.0mu
Mg 24Mg__Kurucz-high.R1e+06_0.3-50.0mu
Al 27Al__Kurucz-high.R1e+06_0.3-50.0mu
Sc 45Sc__Kurucz-high.R1e+06_0.3-50.0mu
Ti 48Ti__Kurucz-high.R1e+06_0.3-50.0mu
Mn 55Mn__Kurucz-high.R1e+06_0.3-50.0mu
Ni 58Ni__Kurucz-high.R1e+06_0.3-50.0mu
TiO 48Ti-16O__Toto.R1e+06_0.3-28.0mu
VO 51V-16O__HyVO.R1e6_0.3-28mu
FeH 56Fe-1H__MoLLIST.R1e+06_0.3-28.0mu
CrH 52Cr-1H__Exomol.R1e6_0.3-28mu
EOF

# --- CLI ---------------------------------------------------------------------
DRY_RUN=false
FORCE_UPLOAD=false
TEST_ONLY=false

usage() {
    cat <<'USAGE'
Usage: upload_opacity_files_to_surf.sh [OPTIONS]

Upload line-by-line opacity .h5 files to SURFdrive over WebDAV (curl).

Options:
  --test       Verify WebDAV credentials and exit
  --dry-run    Show which files would be uploaded, without transferring
  --force      Re-upload even if remote file already exists with matching size
  -h, --help   Show this help message

Credentials file (default: .surfdrive next to this script):
  line 1: WebDAV username (e.g. user@leidenuniv.nl)
  line 2: WebDAV app password from SURFdrive settings
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --test) TEST_ONLY=true; shift ;;
        --dry-run) DRY_RUN=true; shift ;;
        --force) FORCE_UPLOAD=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
    esac
done

# --- helpers -----------------------------------------------------------------
require_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "Error: required command not found: $1" >&2
        exit 1
    fi
}

load_credentials() {
    if [[ ! -f "$CREDENTIALS_FILE" ]]; then
        echo "Error: credentials file not found: $CREDENTIALS_FILE" >&2
        echo "Create it with your SURFdrive WebDAV username on line 1 and app password on line 2." >&2
        exit 1
    fi
    SURF_USER="$(sed -n '1p' "$CREDENTIALS_FILE")"
    SURF_PASS="$(sed -n '2p' "$CREDENTIALS_FILE")"
    if [[ -z "$SURF_USER" || -z "$SURF_PASS" ]]; then
        echo "Error: $CREDENTIALS_FILE must contain username and password on separate lines." >&2
        exit 1
    fi
}

url_encode() {
    python3 -c 'import sys, urllib.parse; print(urllib.parse.quote(sys.argv[1], safe=""))' "$1"
}

url_encode_path() {
    python3 -c '
import sys, urllib.parse
print("/".join(urllib.parse.quote(part, safe="") for part in sys.argv[1].split("/")))
' "$1"
}

human_size() {
    numfmt --to=iec-i --suffix=B "$1" 2>/dev/null || echo "${1} B"
}

# SURF recommends ~10 minutes per GB for large files.
upload_timeout_seconds() {
    local size_bytes="$1"
    local size_gb=$(( (size_bytes + 1024 * 1024 * 1024 - 1) / (1024 * 1024 * 1024) ))
    local timeout=$(( size_gb * 600 + 600 ))
    echo "$timeout"
}

curl_auth() {
    curl -sS -u "${SURF_USER}:${SURF_PASS}" \
        --cookie-jar "$COOKIE_JAR" \
        --cookie "$COOKIE_JAR" \
        "$@"
}

webdav_base_url() {
    local encoded_user
    encoded_user="$(url_encode "$SURF_USER")"
    echo "${SURFDRIVE_HOST}/remote.php/dav/files/${encoded_user}"
}

webdav_url_for_path() {
    local remote_rel="$1"
    local encoded_path
    encoded_path="$(url_encode_path "$remote_rel")"
    echo "$(webdav_base_url)/${encoded_path}"
}

webdav_test_connection() {
    local code
    code="$(curl_auth -o /dev/null -w "%{http_code}" -X PROPFIND \
        --header "Depth: 0" "$(webdav_base_url)/")"
    if [[ "$code" == "207" || "$code" == "200" ]]; then
        echo "WebDAV connection OK (HTTP ${code})"
        return 0
    fi
    echo "WebDAV connection failed (HTTP ${code}). Check username/password in ${CREDENTIALS_FILE}." >&2
    return 1
}

webdav_mkcol() {
    local remote_rel="$1"
    local url code
    url="$(webdav_url_for_path "$remote_rel")"
    code="$(curl_auth -o /dev/null -w "%{http_code}" -X MKCOL "$url")"
    case "$code" in
        201) return 0 ;;  # created
        405) return 0 ;;  # already exists
        409) return 1 ;;  # parent missing
        *) echo "MKCOL failed for ${remote_rel} (HTTP ${code})" >&2; return 1 ;;
    esac
}

webdav_mkcol_recursive() {
    local remote_rel="${1%/}"
    local current="" part
    IFS='/' read -r -a parts <<< "$remote_rel"
    for part in "${parts[@]}"; do
        [[ -z "$part" ]] && continue
        if [[ -z "$current" ]]; then
            current="$part"
        else
            current="${current}/${part}"
        fi
        if ! webdav_mkcol "$current"; then
            echo "Error: could not create remote directory: ${current}" >&2
            return 1
        fi
    done
}

remote_file_size() {
    local remote_rel="$1"
    local url size
    url="$(webdav_url_for_path "$remote_rel")"
    size="$(curl_auth -sI "$url" | awk 'tolower($1)=="content-length:" {print $2}' | tr -d '\r')"
    if [[ "$size" =~ ^[0-9]+$ ]]; then
        echo "$size"
        return 0
    fi
    echo ""
    return 1
}

webdav_upload_file() {
    local local_file="$1"
    local remote_rel="$2"
    local size timeout url code

    size="$(stat -c%s "$local_file")"
    timeout="$(upload_timeout_seconds "$size")"
    url="$(webdav_url_for_path "$remote_rel")"

    if $DRY_RUN; then
        echo "[dry-run] would upload: ${local_file}"
        echo "          -> ${remote_rel} ($(human_size "$size"), timeout ${timeout}s)"
        return 0
    fi

    echo "Uploading $(basename "$local_file") ($(human_size "$size"), timeout ${timeout}s) ..."
  code="$(curl_auth \
        --connect-timeout 60 \
        --max-time "$timeout" \
        --retry 3 \
        --retry-delay 10 \
        --retry-all-errors \
        -T "$local_file" \
        -o /dev/null \
        -w "%{http_code}" \
        "$url")"

    case "$code" in
        201|204) echo "  done (HTTP ${code})"; return 0 ;;
        *) echo "  upload failed (HTTP ${code})" >&2; return 1 ;;
    esac
}

find_opacity_file() {
    local linelist="$1"
    local matches=()
    mapfile -t matches < <(find "$OPACITY_PATH" -name "*${linelist}*.xsec.petitRADTRANS.h5" 2>/dev/null | sort)
    if [[ ${#matches[@]} -eq 0 ]]; then
        return 1
    fi
    if [[ ${#matches[@]} -gt 1 ]]; then
        echo "Warning: multiple matches for linelist '${linelist}', using first:" >&2
        printf '  %s\n' "${matches[@]}" >&2
    fi
    echo "${matches[0]}"
}

relative_opacity_path() {
    local local_file="$1"
    local opacity_root="${OPACITY_PATH%/}/"
    echo "${local_file#"${opacity_root}"}"
}

remote_rel_path() {
  local local_file="$1"
  echo "${REMOTE_BASE}/line_by_line/$(relative_opacity_path "$local_file")"
}

remote_parent_path() {
  local remote_rel="$1"
  echo "${remote_rel%/*}"
}

# --- main --------------------------------------------------------------------
require_command curl
require_command python3
require_command find
require_command stat

load_credentials
touch "$COOKIE_JAR"

if $TEST_ONLY; then
    webdav_test_connection
    exit $?
fi

if [[ ! -d "$OPACITY_PATH" ]]; then
    echo "Error: opacity path not found: $OPACITY_PATH" >&2
    exit 1
fi

echo "SURFdrive upload"
echo "  local root : ${OPACITY_PATH}"
echo "  remote base: ${REMOTE_BASE}/line_by_line/"
echo "  credentials: ${CREDENTIALS_FILE}"
if $DRY_RUN; then
    echo "  mode       : dry-run"
elif $FORCE_UPLOAD; then
    echo "  mode       : force re-upload"
else
    echo "  mode       : upload (skip existing files with matching size)"
fi
echo

if ! $DRY_RUN; then
    webdav_test_connection
    webdav_mkcol_recursive "${REMOTE_BASE}/line_by_line"
fi

uploaded=0
skipped=0
missing=0
failed=0

while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    [[ "$line" =~ ^# ]] && continue

    species_name="${line%% *}"
    linelist="${line#"$species_name"}"
    linelist="${linelist# }"

    local_file="$(find_opacity_file "$linelist" || true)"
    if [[ -z "$local_file" ]]; then
        echo "MISSING ${species_name}: no file matching *${linelist}*.h5 under ${OPACITY_PATH}" >&2
        missing=$((missing + 1))
        continue
    fi

    remote_rel="$(remote_rel_path "$local_file")"
    local_size="$(stat -c%s "$local_file")"

    if ! $FORCE_UPLOAD && ! $DRY_RUN; then
        remote_size="$(remote_file_size "$remote_rel" || true)"
        if [[ -n "$remote_size" && "$remote_size" == "$local_size" ]]; then
            echo "SKIP ${species_name}: already on SURFdrive ($(human_size "$local_size"))"
            skipped=$((skipped + 1))
            continue
        fi
    fi

    echo "---- ${species_name} ----"
    echo "  remote: ${remote_rel}"
    if ! $DRY_RUN; then
        webdav_mkcol_recursive "$(remote_parent_path "$remote_rel")"
    fi
    if webdav_upload_file "$local_file" "$remote_rel"; then
        uploaded=$((uploaded + 1))
    else
        failed=$((failed + 1))
    fi
done <<< "$SPECIES_LIST"

echo
echo "Summary: uploaded=${uploaded}, skipped=${skipped}, missing=${missing}, failed=${failed}"
if [[ "$failed" -gt 0 || "$missing" -gt 0 ]]; then
    exit 1
fi
