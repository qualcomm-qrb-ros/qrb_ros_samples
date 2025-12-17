
#!/bin/sh
set -eu

SANDBOX="/usr/local/sandbox"
mkdir -p "$SANDBOX"

OUT="$SANDBOX/output.log"

: > "$OUT"

python3 -m pip install -r "$SANDBOX/requirement.txt"

python3 "$SANDBOX/runner.py" 2>&1 | tee -a "$OUT"