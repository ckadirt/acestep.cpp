#!/bin/bash
# A0, the listening session for Cantor's audio prompting
# (Cantor/docs/plans/audio-prompting-plan.md § A0).
#
# Renders every combination below with ace-synth and names each file for what
# it is, so the choices can be made by ear. Nothing here is Cantor: it is the
# engine's own CLI.
#
#   ./run.sh <tune.(wav|mp3)> <song.wav> <out-dir> [models-dir] [ace-synth]
#
#   tune   a hummed or sung melody, 8-20 s
#   song   a generated song with vocals (only its first 30 s are used)
#
# Output: <out-dir>/<group>__<what>0.wav, the request JSON beside each, and
# timings.csv. Needs ffmpeg for tiling and trimming.

set -euo pipefail

TUNE=${1:?tune}
SONG=${2:?song}
OUT=${3:?out dir}
MODELS=${4:-$HOME/Projects/Cantor/ckpts}
SYNTH=${5:-$(dirname "$0")/../../build-audio-cpu/ace-synth}

# KNOBS
DIT=acestep-v15-turbo-Q4_K_M.gguf
SEED=42
FULL_SONG_S=60          # what "a full song" tiles to; short, for a CPU box
LONGER_S=30             # seconds an extension adds
SONG_CLIP_S=30          # how much of the song a version covers

mkdir -p "$OUT/src"
TIMINGS="$OUT/timings.csv"
[ -f "$TIMINGS" ] || echo "name,seconds" > "$TIMINGS"

# Sources, normalised to 48 kHz stereo WAV so every run reads the same bytes.
ffmpeg -loglevel error -y -i "$TUNE" -ar 48000 -ac 2 "$OUT/src/tune.wav"
ffmpeg -loglevel error -y -i "$SONG" -t "$SONG_CLIP_S" -ar 48000 -ac 2 "$OUT/src/song.wav"
TUNE_S=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$OUT/src/tune.wav")
# The raw tune, looped to a full song (A0 question 4, option one).
ffmpeg -loglevel error -y -stream_loop -1 -i "$OUT/src/tune.wav" -t "$FULL_SONG_S" "$OUT/src/tune-tiled.wav"

CAPTION_POP="warm acoustic pop, piano, acoustic guitar, soft drums, bass, full band"
CAPTION_JAZZ="smoky late night jazz trio, upright bass, brushed drums, piano"
CAPTION_FEMALE="intimate acoustic ballad sung by a woman, fingerpicked guitar, soft piano"
LYRICS='[verse]\nI walk along the river light\nthe evening hums a song tonight\n[chorus]\nstay with me, stay with me'

# render <name> <src> <ref-or-empty> <json-fields>
render() {
    local name=$1 src=$2 ref=$3 fields=$4
    local req="$OUT/$name.json"
    if [ -f "$OUT/${name}0.wav" ]; then
        echo "skip $name"
        return
    fi
    printf '{"synth_model":"%s","seed":%d,"output_format":"wav16",%s}\n' "$DIT" "$SEED" "$fields" > "$req"
    local args=(--models "$MODELS" --request "$req" --src-audio "$src")
    [ -n "$ref" ] && args+=(--ref-audio "$ref")
    local t0=$(date +%s)
    echo "render $name"
    "$SYNTH" "${args[@]}" > "$OUT/$name.log" 2>&1 || { echo "FAILED $name (see $name.log)"; return; }
    echo "$name,$(( $(date +%s) - t0 ))" >> "$TIMINGS"
}

cover() { # name src ref task caption lyrics strength noise
    render "$1" "$2" "$3" "$(printf '"task_type":"%s","caption":"%s","lyrics":"%s","audio_cover_strength":%s,"cover_noise_strength":%s' "$4" "$5" "$6" "$7" "$8")"
}

# ---- 1. hum -> song, as hummed: where do close, between and free sit?
for s in 1.0 0.8 0.65 0.5 0.35 0.2; do
    cover "hum__cover__s$s" "$OUT/src/tune.wav" "" cover "$CAPTION_POP" "[Instrumental]" "$s" 0
    cover "hum__nofsq__s$s" "$OUT/src/tune.wav" "" cover-nofsq "$CAPTION_POP" "[Instrumental]" "$s" 0
done
for s in 0.65 0.35; do
    cover "hum__cover__s${s}__noise0.3" "$OUT/src/tune.wav" "" cover "$CAPTION_POP" "[Instrumental]" "$s" 0.3
done
# a different style, same tune
cover "hum__cover__jazz__s0.65" "$OUT/src/tune.wav" "" cover "$CAPTION_JAZZ" "[Instrumental]" 0.65 0

# ---- 2. do words land on a hummed contour?
cover "hum__cover__lyrics__s0.65" "$OUT/src/tune.wav" "" cover "$CAPTION_FEMALE" "$LYRICS" 0.65 0
cover "hum__nofsq__lyrics__s0.5" "$OUT/src/tune.wav" "" cover-nofsq "$CAPTION_FEMALE" "$LYRICS" 0.5 0

# ---- 3. a full song from a short tune
cover "hum-full__tiled-raw__cover__s0.65" "$OUT/src/tune-tiled.wav" "" cover "$CAPTION_POP" "[Instrumental]" 0.65 0
cover "hum-full__tiled-raw__cover__s0.5" "$OUT/src/tune-tiled.wav" "" cover "$CAPTION_POP" "[Instrumental]" 0.5 0
# option two: cover the tune first, then loop the cover
if [ -f "$OUT/hum__cover__s0.650.wav" ] && [ ! -f "$OUT/hum-full__cover-then-loop__s0.65.wav" ]; then
    ffmpeg -loglevel error -y -stream_loop -1 -i "$OUT/hum__cover__s0.650.wav" -t "$FULL_SONG_S" \
        "$OUT/hum-full__cover-then-loop__s0.65.wav"
fi
# option three: cover the tune, then extend the cover
render "hum-full__cover-then-extend" "$OUT/hum__cover__s0.650.wav" "" \
    "$(printf '"task_type":"repaint","caption":"%s","lyrics":"[Instrumental]","repainting_start":%s,"repainting_end":%s' \
        "$CAPTION_POP" "$TUNE_S" "$FULL_SONG_S")"

# ---- 4. a version of a song: closeness, and keeping the voice or not
for s in 0.8 0.5 0.3; do
    cover "version__nofsq__keepvoice__s$s" "$OUT/src/song.wav" "$OUT/src/song.wav" cover-nofsq "$CAPTION_FEMALE" "[Instrumental]" "$s" 0
    cover "version__nofsq__newvoice__s$s" "$OUT/src/song.wav" "" cover-nofsq "$CAPTION_FEMALE" "[Instrumental]" "$s" 0
    cover "version__cover__newvoice__s$s" "$OUT/src/song.wav" "" cover "$CAPTION_JAZZ" "[Instrumental]" "$s" 0
done

# ---- 5. longer: does the new part sound like the same song?
render "longer__end__+${LONGER_S}s" "$OUT/src/song.wav" "" \
    "$(printf '"task_type":"repaint","caption":"%s","lyrics":"[Instrumental]","repainting_start":%d,"repainting_end":%d' \
        "$CAPTION_POP" "$SONG_CLIP_S" "$((SONG_CLIP_S + LONGER_S))")"
render "longer__start__+${LONGER_S}s" "$OUT/src/song.wav" "" \
    "$(printf '"task_type":"repaint","caption":"%s","lyrics":"[Instrumental]","repainting_start":-%d' \
        "$CAPTION_POP" "$LONGER_S")"

echo "done: $(ls "$OUT"/*0.wav 2>/dev/null | wc -l) renders in $OUT"
