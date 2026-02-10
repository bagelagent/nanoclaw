#!/bin/bash
# Add a research topic to the global queue
# Usage: add-research-topic.sh "Topic text" "source" "tag1,tag2" priority

TOPIC="$1"
SOURCE="${2:-manual}"
TAGS="${3:-general}"
PRIORITY="${4:-5}"

if [ -z "$TOPIC" ]; then
  echo "Usage: add-research-topic.sh \"Topic text\" \"source\" \"tag1,tag2\" priority"
  exit 1
fi

QUEUE_FILE="/workspace/project/groups/global/research-queue.json"
TOPIC_ID="rq-$(date +%s%3N)-$(openssl rand -hex 3)"
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

# Convert comma-separated tags to JSON array
TAG_ARRAY=$(echo "$TAGS" | python3 -c "import sys, json; print(json.dumps(sys.stdin.read().strip().split(',')))")

# Create new topic JSON
NEW_TOPIC=$(cat <<EOF
{
  "id": "$TOPIC_ID",
  "topic": "$TOPIC",
  "source": "$SOURCE",
  "addedBy": "main",
  "addedAt": "$TIMESTAMP",
  "priority": $PRIORITY,
  "status": "pending",
  "tags": $TAG_ARRAY
}
EOF
)

# Add to queue
python3 <<PYTHON
import json

with open('$QUEUE_FILE', 'r') as f:
    data = json.load(f)

new_topic = json.loads('''$NEW_TOPIC''')
data['queue'].append(new_topic)
data['metadata']['lastUpdated'] = '$TIMESTAMP'

with open('$QUEUE_FILE', 'w') as f:
    json.dump(data, f, indent=2)

print(f"Added topic: {new_topic['id']}")
print(f"Topic: {new_topic['topic']}")
print(f"Priority: {new_topic['priority']}")
PYTHON
