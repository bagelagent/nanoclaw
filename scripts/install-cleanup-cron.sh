#!/bin/bash
# Install cleanup cron job
# Run this on the host to add the daily cleanup task

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLEANUP_SCRIPT="$SCRIPT_DIR/cleanup-old-data.sh"

echo "Installing NanoClaw cleanup cron job..."

# Check if cron job already exists
if crontab -l 2>/dev/null | grep -q "cleanup-old-data.sh"; then
  echo "⚠️  Cleanup cron job already exists"
  exit 0
fi

# Add cron job
(crontab -l 2>/dev/null; echo "# NanoClaw cleanup - runs daily at 3 AM"; echo "0 3 * * * $CLEANUP_SCRIPT >> /workspace/logs/cleanup.log 2>&1") | crontab -

echo "✅ Cleanup cron job installed (runs daily at 3 AM)"
echo "   Logs: /workspace/logs/cleanup.log"
