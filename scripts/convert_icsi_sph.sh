#!/usr/bin/env bash
# Convert ICSI NIST SPH (often embedded-shorten compressed) → 16kHz mono WAV
# so soundfile can read them natively. Runs after download_ami_icsi.sh finishes.

set -euo pipefail

ICSI_ROOT="${ICSI_ROOT:-/Volumes/HIKSEMI/autoresearch-splice-sources/icsi}"
LOG_FILE=".omc/logs/icsi_convert.log"
mkdir -p .omc/logs

: >"$LOG_FILE"

if [ ! -d "$ICSI_ROOT" ]; then
    echo "ICSI dir not found: $ICSI_ROOT" >&2
    exit 1
fi

sph_count=$(find "$ICSI_ROOT" -name 'chan*.sph' | wc -l | tr -d ' ')
echo "Found $sph_count ICSI SPH files to convert" | tee -a "$LOG_FILE"

if [ "$sph_count" -eq 0 ]; then
    echo "No SPH files — already converted or ICSI not downloaded" | tee -a "$LOG_FILE"
    exit 0
fi

converted=0
failed=0
while IFS= read -r sph; do
    wav="${sph%.sph}.wav"
    if [ -f "$wav" ]; then
        continue
    fi
    # ffmpeg handles NIST-SPH + embedded-shorten on macOS via built-in shorten decoder.
    # Force 16kHz mono to match AMI.
    if ffmpeg -y -i "$sph" -ac 1 -ar 16000 -loglevel error "$wav" 2>>"$LOG_FILE"; then
        rm -f "$sph"
        converted=$((converted + 1))
    else
        failed=$((failed + 1))
        echo "FAIL: $sph" >>"$LOG_FILE"
    fi
done < <(find "$ICSI_ROOT" -name 'chan*.sph')

echo "Converted: $converted, Failed: $failed" | tee -a "$LOG_FILE"

# Verify soundfile can now read a sample
sample=$(find "$ICSI_ROOT" -name 'chan*.wav' | head -1)
if [ -n "$sample" ]; then
    uv run python -c "
import soundfile as sf
info = sf.info('$sample')
print(f'Sample OK: {info.duration:.1f}s {info.samplerate}Hz {info.channels}ch')
" | tee -a "$LOG_FILE"
fi
