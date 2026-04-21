#!/usr/bin/env bash
# download_ami_icsi.sh — Reproducible download of AMI + ICSI headset channels
#
# AMI:  WAV headset files (~15-20 GB total)
#       URL: https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/<ID>/audio/<ID>.Headset-<N>.wav
#
# ICSI: SPH per-speaker headset channel files (~15 GB total)
#       URL: https://groups.inf.ed.ac.uk/ami/ICSIsignals/SPH/<ID>/chan<N>.sph
#
# Usage: ./scripts/download_ami_icsi.sh [--ami-only] [--icsi-only] [--dry-run]
#
# Requirements: aria2c (for parallel downloads), curl, bash >= 4

set -euo pipefail

AMI_ROOT="/Volumes/HIKSEMI/autoresearch-splice-sources/ami"
ICSI_ROOT="/Volumes/HIKSEMI/autoresearch-splice-sources/icsi"
LOG_DIR=".omc/logs"
LOG_FILE="$LOG_DIR/corpus_download.log"
MIN_FREE_GB=40
# Number of parallel connections for aria2c
PARALLEL=8

DO_AMI=true
DO_ICSI=true
DRY_RUN=false

for arg in "$@"; do
  case "$arg" in
    --ami-only)  DO_ICSI=false ;;
    --icsi-only) DO_AMI=false ;;
    --dry-run)   DRY_RUN=true ;;
  esac
done

mkdir -p "$LOG_DIR"

log() {
  local ts
  ts=$(date '+%Y-%m-%dT%H:%M:%S')
  echo "[$ts] $*" | tee -a "$LOG_FILE"
}

check_disk_space() {
  local path="$1"
  # macOS: df -g gives gigabytes; Linux: df --block-size=G
  local avail_gb
  avail_gb=$(df -g "$path" 2>/dev/null | awk 'NR==2{print $4}' || echo "")
  if [[ -z "$avail_gb" ]]; then
    log "WARN: could not determine free space on $path — continuing"
    return 0
  fi
  if (( avail_gb < MIN_FREE_GB )); then
    log "ERROR: only ${avail_gb} GB free on $path (need >= ${MIN_FREE_GB} GB). Aborting."
    exit 1
  fi
  log "INFO: ${avail_gb} GB free on $path — OK"
}

# ── AMI Meeting IDs ────────────────────────────────────────────────────────────
AMI_SCENARIO_MEETINGS=(
  ES2002 ES2003 ES2004 ES2005 ES2006 ES2007 ES2008 ES2009 ES2010 ES2011 ES2012 ES2013 ES2014 ES2015 ES2016
  IS1000 IS1001 IS1002 IS1003 IS1004 IS1005 IS1006 IS1007 IS1008 IS1009
  TS3003 TS3004 TS3005 TS3006 TS3007 TS3008 TS3009 TS3010 TS3011 TS3012
)
AMI_SCENARIO_SUFFIXES=(a b c d)

AMI_NONSCENARIO_MEETINGS=(
  EN2001a EN2001b EN2001d EN2001e
  EN2002a EN2002b EN2002c EN2002d
  EN2003a EN2004a EN2005a
  EN2006a EN2006b
  EN2009b EN2009c EN2009d
  IB4001 IB4002 IB4003 IB4004 IB4005 IB4010 IB4011
  IN1001 IN1002 IN1005 IN1007 IN1008 IN1009 IN1012 IN1013 IN1014 IN1015 IN1016
)

# ── ICSI Meeting IDs ───────────────────────────────────────────────────────────
ICSI_MEETINGS=(
  Bed002 Bed003 Bed004 Bed005 Bed006 Bed008 Bed009 Bed010 Bed011 Bed012 Bed013 Bed014 Bed015 Bed016 Bed017
  Bmr001 Bmr002 Bmr003 Bmr005 Bmr006 Bmr007 Bmr008 Bmr009 Bmr010 Bmr011 Bmr012 Bmr013 Bmr014 Bmr015
  Bmr016 Bmr018 Bmr019 Bmr020 Bmr021 Bmr022 Bmr023 Bmr024 Bmr025 Bmr026 Bmr027 Bmr028 Bmr029 Bmr030 Bmr031
  Bns001 Bns002 Bns003
  Bro003 Bro004 Bro005 Bro007 Bro008 Bro010 Bro011 Bro012 Bro013 Bro014 Bro015 Bro016 Bro017 Bro018
  Bro019 Bro021 Bro022 Bro023 Bro024 Bro025 Bro026 Bro027 Bro028
  Bsr001 Btr001 Btr002 Buw001 Bdb001
)
# ICSI headset channels per meeting (0-9 and A-F hex; actual populated channels vary)
ICSI_CHANNELS=(0 1 2 3 4 5 6 7 8 9 A B C D E F)

# ── Build URL list file for aria2c ────────────────────────────────────────────
# aria2c input format:
#   <url>
#     dir=<destination_dir>
#     out=<filename>

build_ami_urllist() {
  local urlfile="$1"
  local ami_base="https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus"
  local count=0

  # Scenario meetings
  for sess in "${AMI_SCENARIO_MEETINGS[@]}"; do
    for suf in "${AMI_SCENARIO_SUFFIXES[@]}"; do
      local mid="${sess}${suf}"
      for n in 0 1 2 3; do
        local dest_file="$AMI_ROOT/$mid/audio/${mid}.Headset-${n}.wav"
        if [[ -f "$dest_file" && $(stat -f%z "$dest_file" 2>/dev/null || echo 0) -gt 1000 ]]; then
          continue
        fi
        echo "${ami_base}/${mid}/audio/${mid}.Headset-${n}.wav" >> "$urlfile"
        echo "  dir=$AMI_ROOT/$mid/audio" >> "$urlfile"
        echo "  out=${mid}.Headset-${n}.wav" >> "$urlfile"
        (( count++ )) || true
      done
    done
  done

  # Non-scenario meetings
  for mid in "${AMI_NONSCENARIO_MEETINGS[@]}"; do
    for n in 0 1 2 3; do
      local dest_file="$AMI_ROOT/$mid/audio/${mid}.Headset-${n}.wav"
      if [[ -f "$dest_file" && $(stat -f%z "$dest_file" 2>/dev/null || echo 0) -gt 1000 ]]; then
        continue
      fi
      echo "${ami_base}/${mid}/audio/${mid}.Headset-${n}.wav" >> "$urlfile"
      echo "  dir=$AMI_ROOT/$mid/audio" >> "$urlfile"
      echo "  out=${mid}.Headset-${n}.wav" >> "$urlfile"
      (( count++ )) || true
    done
  done

  echo "$count"
}

build_icsi_urllist() {
  local urlfile="$1"
  local icsi_base="https://groups.inf.ed.ac.uk/ami/ICSIsignals/SPH"
  local count=0

  for mid in "${ICSI_MEETINGS[@]}"; do
    for chan in "${ICSI_CHANNELS[@]}"; do
      local dest_file="$ICSI_ROOT/$mid/chan${chan}.sph"
      if [[ -f "$dest_file" && $(stat -f%z "$dest_file" 2>/dev/null || echo 0) -gt 1000 ]]; then
        continue
      fi
      echo "${icsi_base}/${mid}/chan${chan}.sph" >> "$urlfile"
      echo "  dir=$ICSI_ROOT/$mid" >> "$urlfile"
      echo "  out=chan${chan}.sph" >> "$urlfile"
      (( count++ )) || true
    done
  done

  echo "$count"
}

# ── Write LICENSE files ───────────────────────────────────────────────────────
write_licenses() {
  mkdir -p "$AMI_ROOT" "$ICSI_ROOT"
  cat > "$AMI_ROOT/LICENSE" <<'EOF'
AMI Meeting Corpus — Audio Data License
========================================

The AMI Meeting Corpus audio signals are released under the
Creative Commons Attribution 4.0 International (CC BY 4.0) License.
https://creativecommons.org/licenses/by/4.0/

Citation:
  Carletta, J., et al. (2006). The AMI Meeting Corpus: A Pre-Announcement.
  In Proceedings of the 2nd International Conference on Machine Learning
  for Multimodal Interaction (MLMI 2005), Springer Lecture Notes in
  Computer Science, pp. 28-39.

Source: https://groups.inf.ed.ac.uk/ami/corpus/
Download: https://groups.inf.ed.ac.uk/ami/download/
EOF

  cat > "$ICSI_ROOT/LICENSE" <<'EOF'
ICSI Meeting Corpus — Audio Data License
=========================================

The ICSI Meeting Corpus audio signals are released under the
Creative Commons Attribution 4.0 International (CC BY 4.0) License.
https://creativecommons.org/licenses/by/4.0/

Citation:
  Janin, A., et al. (2003). The ICSI Meeting Corpus.
  In Proceedings of ICASSP 2003, vol. I, pp. 364-367.

Source: https://groups.inf.ed.ac.uk/ami/icsi/
Download: https://groups.inf.ed.ac.uk/ami/icsi/download/
EOF
  log "INFO: LICENSE files written"
}

# ── Download with aria2c ──────────────────────────────────────────────────────
run_aria2c() {
  local urlfile="$1"
  local label="$2"
  local url_count="$3"

  if [[ "$DRY_RUN" == "true" ]]; then
    log "INFO: DRY-RUN $label — would download $url_count URLs"
    head -12 "$urlfile"
    return 0
  fi

  if [[ "$url_count" -eq 0 ]]; then
    log "INFO: $label — nothing to download (all files already present)"
    return 0
  fi

  log "INFO: Starting $label aria2c download of $url_count files (parallel=$PARALLEL)..."

  aria2c \
    --input-file="$urlfile" \
    --max-concurrent-downloads="$PARALLEL" \
    --split=1 \
    --max-connection-per-server=1 \
    --retry-wait=5 \
    --max-tries=3 \
    --timeout=120 \
    --connect-timeout=30 \
    --allow-overwrite=false \
    --auto-file-renaming=false \
    --conditional-get=true \
    --remote-time=true \
    --log="$LOG_DIR/${label}_aria2c.log" \
    --log-level=warn \
    --summary-interval=60 \
    2>&1 | tee -a "$LOG_FILE" || {
    log "WARN: aria2c exited with errors for $label — check log for details"
  }
}

# ── Main ──────────────────────────────────────────────────────────────────────
log "INFO: US-600 corpus download started. ami=$DO_AMI icsi=$DO_ICSI dry_run=$DRY_RUN"

write_licenses

if [[ "$DO_AMI" == "true" ]]; then
  check_disk_space "$AMI_ROOT"
  AMI_URLFILE=$(mktemp /tmp/ami_urls.XXXXXX)
  log "INFO: Building AMI URL list..."
  ami_count=$(build_ami_urllist "$AMI_URLFILE")
  log "INFO: AMI URL list: $ami_count files to download → $AMI_URLFILE"
  run_aria2c "$AMI_URLFILE" "ami" "$ami_count"
  rm -f "$AMI_URLFILE"
fi

if [[ "$DO_ICSI" == "true" ]]; then
  check_disk_space "$ICSI_ROOT"
  ICSI_URLFILE=$(mktemp /tmp/icsi_urls.XXXXXX)
  log "INFO: Building ICSI URL list..."
  icsi_count=$(build_icsi_urllist "$ICSI_URLFILE")
  log "INFO: ICSI URL list: $icsi_count files to download → $ICSI_URLFILE"
  run_aria2c "$ICSI_URLFILE" "icsi" "$icsi_count"
  rm -f "$ICSI_URLFILE"
fi

log "INFO: All downloads finished."

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "=== Download Summary ==="
if [[ "$DO_AMI" == "true" ]]; then
  ami_files=$(find "$AMI_ROOT" -name "*.Headset-*.wav" 2>/dev/null | wc -l | tr -d ' ')
  ami_meetings=$(find "$AMI_ROOT" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l | tr -d ' ')
  ami_size=$(du -sh "$AMI_ROOT" 2>/dev/null | cut -f1)
  echo "AMI:  $ami_files WAV files across $ami_meetings meeting dirs, $ami_size total"
fi
if [[ "$DO_ICSI" == "true" ]]; then
  icsi_files=$(find "$ICSI_ROOT" -name "*.sph" 2>/dev/null | wc -l | tr -d ' ')
  icsi_meetings=$(find "$ICSI_ROOT" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l | tr -d ' ')
  icsi_size=$(du -sh "$ICSI_ROOT" 2>/dev/null | cut -f1)
  echo "ICSI: $icsi_files SPH files across $icsi_meetings meeting dirs, $icsi_size total"
fi
echo "Log: $LOG_FILE"
