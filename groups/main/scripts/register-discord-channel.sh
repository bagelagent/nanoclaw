#!/bin/bash
# Register a Discord channel for Bagel
# Usage: ./register-discord-channel.sh <channel_id> <channel_name> [folder_name]

set -e

CHANNEL_ID="$1"
CHANNEL_NAME="$2"
FOLDER_NAME="${3:-}"

if [ -z "$CHANNEL_ID" ] || [ -z "$CHANNEL_NAME" ]; then
  echo "Usage: $0 <channel_id> <channel_name> [folder_name]"
  echo ""
  echo "Example: $0 1234567890 \"my-discord-channel\" discord-my-channel"
  echo ""
  echo "To find channel ID:"
  echo "1. Enable Developer Mode in Discord (Settings > Advanced)"
  echo "2. Right-click the channel and 'Copy Channel ID'"
  exit 1
fi

# Generate folder name if not provided
if [ -z "$FOLDER_NAME" ]; then
  # Convert channel name to folder-friendly format
  FOLDER_NAME=$(echo "$CHANNEL_NAME" | tr '[:upper:]' '[:lower:]' | tr ' ' '-' | sed 's/[^a-z0-9-]//g')
  FOLDER_NAME="discord-${FOLDER_NAME}"
fi

echo "Registering Discord channel:"
echo "  Channel ID: $CHANNEL_ID"
echo "  Name: $CHANNEL_NAME"
echo "  Folder: $FOLDER_NAME"
echo ""

# Create JID
JID="discord:${CHANNEL_ID}"

# Check if already registered
EXISTING=$(node -e "
const db = require('better-sqlite3')('/workspace/project/store/messages.db');
const row = db.prepare('SELECT * FROM registered_groups WHERE jid = ?').get('$JID');
if (row) {
  console.log('EXISTS');
  process.exit(1);
}
" 2>&1)

if [ "$EXISTING" = "EXISTS" ]; then
  echo "❌ Channel already registered!"
  exit 1
fi

# Create folder
FOLDER_PATH="/workspace/project/groups/${FOLDER_NAME}"
mkdir -p "$FOLDER_PATH"
mkdir -p "$FOLDER_PATH/conversations"
mkdir -p "$FOLDER_PATH/tmp"
mkdir -p "$FOLDER_PATH/.claude/skills"

# Create CLAUDE.md
cat > "$FOLDER_PATH/CLAUDE.md" << EOF
# ${CHANNEL_NAME}

This is a Discord channel workspace for Bagel.

## Channel Info
- Channel ID: ${CHANNEL_ID}
- Registered: $(date -u +"%Y-%m-%d %H:%M:%S UTC")

## Notes
Add channel-specific context, preferences, and memory here.
EOF

# Register in database
node -e "
const db = require('better-sqlite3')('/workspace/project/store/messages.db');
db.prepare(\`
  INSERT INTO registered_groups (jid, name, folder, trigger_pattern, requires_trigger, added_at)
  VALUES (?, ?, ?, ?, ?, datetime('now'))
\`).run('$JID', '$CHANNEL_NAME', '$FOLDER_NAME', '@Bagel', 0);
console.log('✅ Channel registered successfully!');
"

echo ""
echo "✅ Discord channel registered!"
echo ""
echo "Channel will now:"
echo "  • Respond to all messages (no @mention needed)"
echo "  • Store conversations in: $FOLDER_PATH/conversations/"
echo "  • Use memory from: $FOLDER_PATH/CLAUDE.md"
echo ""
echo "Restart NanoClaw to apply changes."
