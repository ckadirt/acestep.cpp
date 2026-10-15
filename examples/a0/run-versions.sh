#!/bin/bash
# A0, part two: versions of real songs (Cantor/docs/plans/audio-prompting-plan.md § A0).
#
#   ./run-versions.sh <out-dir> <song.flac> [<song.flac> ...]
#
# For each song: a 40 s clip from 0:30, rendered across closeness, keeping or
# changing the voice, a style change, and two lyric treatments (an imported
# song has no lyrics on file, so: "[Instrumental]" or nothing). The first song
# also gets whole-length versions and a longer song.
#
# Output: <out-dir>/<song-slug>/<what>0.wav, the clip itself as 00-original.wav.

set -euo pipefail

OUT=${1:?out dir}
shift
MODELS=${MODELS:-$HOME/Projects/Cantor/ckpts}
SYNTH=${SYNTH:-$(dirname "$0")/../../build-audio-cpu/ace-synth}

# KNOBS
DIT=acestep-v15-turbo-Q4_K_M.gguf
SEED=42
CLIP_FROM_S=30
CLIP_S=40
LONGER_S=30

CAPTION_SAME="the same song, polished, full band, rich production"
CAPTION_FEMALE="intimate acoustic version sung by a woman, fingerpicked guitar, soft piano, warm"
CAPTION_JAZZ="smoky late night jazz trio, upright bass, brushed drums, piano"
CAPTION_ORCH="cinematic orchestral version, strings, choir, timpani, epic"

TIMINGS="$OUT/timings.csv"
mkdir -p "$OUT"
[ -f "$TIMINGS" ] || echo "song,name,seconds" > "$TIMINGS"

render() { # dir name src ref fields
    local dir=$1 name=$2 src=$3 ref=$4 fields=$5
    local req="$dir/$name.json"
    if [ -f "$dir/${name}0.wav" ]; then echo "skip $name"; return; fi
    printf '{"synth_model":"%s","seed":%d,"output_format":"wav16",%s}\n' "$DIT" "$SEED" "$fields" > "$req"
    local args=(--models "$MODELS" --request "$req" --src-audio "$src")
    [ -n "$ref" ] && args+=(--ref-audio "$ref")
    local t0; t0=$(date +%s)
    echo "render $(basename "$dir")/$name"
    "$SYNTH" "${args[@]}" > "$dir/$name.log" 2>&1 || { echo "FAILED $name (see $dir/$name.log)"; return; }
    echo "$(basename "$dir"),$name,$(( $(date +%s) - t0 ))" >> "$TIMINGS"
}

cover() { # dir name src ref task caption lyrics strength
    render "$1" "$2" "$3" "$4" "$(printf '"task_type":"%s","caption":"%s","lyrics":"%s","audio_cover_strength":%s' "$5" "$6" "$7" "$8")"
}

first=1
for song in "$@"; do
    slug=$(basename "$song" .flac | sed 's/^[0-9]* - //; s/[^A-Za-z0-9]\+/-/g; s/-$//' | tr 'A-Z' 'a-z')
    dir="$OUT/$slug"
    mkdir -p "$dir"
    ffmpeg -loglevel error -y -ss "$CLIP_FROM_S" -t "$CLIP_S" -i "$song" -ar 48000 -ac 2 "$dir/00-original.wav"
    clip="$dir/00-original.wav"

    # closeness, keeping the voice: how far can it move and still be this song?
    for s in 0.8 0.5 0.3; do
        cover "$dir" "keepvoice__nofsq__s$s" "$clip" "$clip" cover-nofsq "$CAPTION_SAME" "[Instrumental]" "$s"
    done
    # imported songs have no lyrics: what does each treatment do to the vocal?
    cover "$dir" "keepvoice__nofsq__s0.5__nolyrics" "$clip" "$clip" cover-nofsq "$CAPTION_SAME" "" 0.5
    # a new voice and a new arrangement
    cover "$dir" "female__nofsq__s0.5" "$clip" "" cover-nofsq "$CAPTION_FEMALE" "" 0.5
    cover "$dir" "female__cover__s0.6" "$clip" "" cover "$CAPTION_FEMALE" "" 0.6
    # a different genre, close and free
    cover "$dir" "jazz__cover__s0.6" "$clip" "" cover "$CAPTION_JAZZ" "[Instrumental]" 0.6
    cover "$dir" "orchestral__cover__s0.35" "$clip" "" cover "$CAPTION_ORCH" "[Instrumental]" 0.35

    if [ "$first" = 1 ]; then
        first=0
        ffmpeg -loglevel error -y -i "$song" -ar 48000 -ac 2 "$dir/00-original-full.wav"
        full="$dir/00-original-full.wav"
        cover "$dir" "full__keepvoice__nofsq__s0.5" "$full" "$full" cover-nofsq "$CAPTION_SAME" "" 0.5
        cover "$dir" "full__female__cover__s0.6" "$full" "" cover "$CAPTION_FEMALE" "" 0.6
        end=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$clip")
        render "$dir" "longer__end__+${LONGER_S}s" "$clip" "" \
            "$(printf '"task_type":"repaint","caption":"%s","lyrics":"","repainting_start":%s,"repainting_end":%s' \
                "$CAPTION_SAME" "$end" "$(echo "$end + $LONGER_S" | bc)")"
    fi
done
echo "done"
