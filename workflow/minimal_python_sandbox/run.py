#!/bin/sh
set -eu

IMG="test_docker_exec:latest"
NAME="test_docker_exec"

mkdir -p _sandbox
chmod +x _sandbox/runner.sh

# Build image
docker build -t "$IMG" .

# Run container, mounting host ./_sandbox to container /usr/local/sandbox
# Container name fixed as requested.
docker run --rm   --name "$NAME"   -v "$(pwd)/_sandbox:/usr/local/sandbox"   "$IMG"