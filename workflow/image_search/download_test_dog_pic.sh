#!/bin/bash

BASE_URL="https://raw.githubusercontent.com/laxmimerit/dog-cat-full-dataset/master/data/train/dogs"
MIN=1
MAX=12498
INTERVAL=5
SAVE_DIR="dogs"

mkdir -p "$SAVE_DIR"

while true; do
    echo "==== New random cycle start ===="

    shuf -i ${MIN}-${MAX} | while read -r NUM; do
        FILE="dog.${NUM}.jpg"
        URL="${BASE_URL}/${FILE}"
        TARGET="${SAVE_DIR}/${FILE}"

        echo "$(date '+%F %T') downloading ${FILE}"

        wget -q \
            --tries=3 \
            --timeout=10 \
            -O "$TARGET" \
            "$URL"

        sleep "$INTERVAL"
    done
done
