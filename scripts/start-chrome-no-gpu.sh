#!/bin/bash

set -e

CHROME="${CHROME:-}"

if [ -z "$CHROME" ]; then
    for candidate in google-chrome google-chrome-stable chromium chromium-browser; do
        if command -v "$candidate" >/dev/null 2>&1; then
            CHROME="$candidate"
            break
        fi
    done
fi

if [ -z "$CHROME" ]; then
    echo "Chrome/Chromium was not found in PATH."
    echo "Set CHROME to your browser command or install google-chrome/chromium."
    exit 1
fi

if [ "$#" -eq 0 ]; then
    TARGET_ARGS=("http://localhost:7860/")
else
    TARGET_ARGS=("$@")
fi

"$CHROME" \
    --disable-gpu \
    --disable-gpu-compositing \
    --disable-accelerated-2d-canvas \
    --disable-accelerated-video-decode \
    --use-angle=swiftshader \
    --enable-unsafe-swiftshader \
    --disable-webgpu \
    "${TARGET_ARGS[@]}" &
