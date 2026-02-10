#!/bin/bash
# Research Heartbeat - Execute one research topic from the queue
# This runs as a scheduled task every N hours

QUEUE_FILE="/workspace/project/groups/global/research-queue.json"
RESEARCH_DIR="/workspace/project/groups/global/research"

# Create research directory if it doesn't exist
mkdir -p "$RESEARCH_DIR"

# Get the highest priority pending topic
TOPIC_JSON=$(jq -r '.queue[] | select(.status == "pending") | @json' "$QUEUE_FILE" | head -1)

if [ -z "$TOPIC_JSON" ]; then
  echo "No pending research topics in queue"
  exit 0
fi

# Extract topic details
TOPIC_ID=$(echo "$TOPIC_JSON" | jq -r '.id')
TOPIC=$(echo "$TOPIC_JSON" | jq -r '.topic')
TAGS=$(echo "$TOPIC_JSON" | jq -r '.tags | join(", ")')

echo "Research Topic: $TOPIC"
echo "Tags: $TAGS"
echo "ID: $TOPIC_ID"

# Output file for this research
OUTPUT_FILE="$RESEARCH_DIR/${TOPIC_ID}.md"

# Mark as in-progress
jq --arg id "$TOPIC_ID" \
   '(.queue[] | select(.id == $id) | .status) = "in-progress"' \
   "$QUEUE_FILE" > "${QUEUE_FILE}.tmp" && mv "${QUEUE_FILE}.tmp" "$QUEUE_FILE"

# The actual research will be done by the agent
# This script just sets up the context and marks the topic
echo "Topic ready for research: $TOPIC_ID"
