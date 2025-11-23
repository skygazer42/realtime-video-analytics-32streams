#!/usr/bin/env bash
# Spawn multiple RTSP streams from a local MP4 for quick dashboard testing (Linux/macOS).
# Usage: ./rtsp-multistream.sh [input_mp4] [stream_count]
# Requirements: ffmpeg installed; firewall allowing TCP 8554 (or change PORT).

set -euo pipefail

INPUT="${1:-data/samples/demo.mp4}"
STREAMS="${2:-4}"
HOST="${HOST:-0.0.0.0}"
PORT_START="${PORT_START:-8554}" # each stream uses PORT_START + i - 1

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg not found. Install ffmpeg first." >&2
  exit 1
fi

if [ ! -f "$INPUT" ]; then
  echo "Input file not found: $INPUT" >&2
  exit 1
fi

pids=()
cleanup() {
  for pid in "${pids[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT

echo "Starting $STREAMS RTSP streams from $INPUT ..."
for ((i=1; i<=STREAMS; i++)); do
  name=$(printf "stream%02d" "$i")
  port=$((PORT_START + i - 1))
  ffmpeg -hide_banner -loglevel warning -re -stream_loop -1 -i "$INPUT" \
    -c copy -f rtsp -rtsp_transport tcp -muxdelay 0.1 -listen 1 "rtsp://${HOST}:${port}/${name}" &
  pids+=("$!")
done

echo "RTSP endpoints:"
for ((i=1; i<=STREAMS; i++)); do
  name=$(printf "stream%02d" "$i")
  port=$((PORT_START + i - 1))
  echo "  rtsp://${HOST}:${port}/${name}"
done

echo "Press Ctrl+C to stop all streams."
wait
