#!/bin/bash
# Cleanup script for old NanoClaw data
# Removes old container logs and prunes unused containers

set -e

LOG_DIR="/workspace/logs"
LOG_RETENTION_DAYS=7

echo "🧹 NanoClaw Cleanup Script"
echo "=========================="

# Clean up old container logs
if [ -d "$LOG_DIR" ]; then
  echo "📋 Cleaning container logs older than ${LOG_RETENTION_DAYS} days..."
  OLD_LOGS=$(find "$LOG_DIR" -name "container-*.log" -mtime +${LOG_RETENTION_DAYS} 2>/dev/null | wc -l)
  if [ "$OLD_LOGS" -gt 0 ]; then
    find "$LOG_DIR" -name "container-*.log" -mtime +${LOG_RETENTION_DAYS} -delete
    echo "   Removed $OLD_LOGS old log files"
  else
    echo "   No old logs to remove"
  fi
fi

# Prune unused containers (stopped containers not used in 7 days)
echo "🐳 Pruning unused containers..."
if command -v podman &> /dev/null; then
  podman container prune -f --filter "until=168h" 2>/dev/null || echo "   No containers to prune"
elif command -v docker &> /dev/null; then
  docker container prune -f --filter "until=168h" 2>/dev/null || echo "   No containers to prune"
fi

# Prune dangling images (not tagged and not referenced)
echo "🖼️  Pruning dangling images..."
if command -v podman &> /dev/null; then
  podman image prune -f 2>/dev/null || echo "   No images to prune"
elif command -v docker &> /dev/null; then
  docker image prune -f 2>/dev/null || echo "   No images to prune"
fi

# Show disk usage
echo ""
echo "💾 Current disk usage:"
df -h /workspace | grep -v Filesystem

echo ""
echo "✅ Cleanup complete"
