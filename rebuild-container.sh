#!/bin/bash
# Manually rebuild and restart NanoClaw container
# Run this from the project root when container code changes

set -e

echo "=== NanoClaw Container Rebuild ===="
echo ""

# 1. Build container image
echo "Step 1: Building container image..."
cd container
./build.sh
cd ..

echo ""
echo "Step 2: Finding and stopping running containers..."
# Stop all nanoclaw containers
docker ps -q --filter "name=nanoclaw" | xargs -r docker stop || true
podman ps -q --filter "name=nanoclaw" | xargs -r podman stop || true

echo ""
echo "Step 3: Container rebuild complete!"
echo "Next agent query will spawn a fresh container with new code."
echo ""
