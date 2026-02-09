#!/bin/bash
# Helper script to post artwork to DevAIntArt
# Usage: ./devaintart-post.sh <type> <title> <description> <prompt> <tags> [image_url_or_svg_data]
#   type: "svg" or "png"

set -e

TYPE="$1"
TITLE="$2"
DESCRIPTION="$3"
PROMPT="$4"
TAGS="$5"
DATA="$6"

API_KEY="${DEVAINTART_API_KEY:-$(cat /workspace/group/.env 2>/dev/null | grep DEVAINTART_API_KEY | cut -d= -f2)}"

if [ -z "$API_KEY" ]; then
  echo "Error: DEVAINTART_API_KEY not found"
  exit 1
fi

if [ "$TYPE" = "svg" ]; then
  # Post SVG artwork
  curl -X POST https://devaintart.net/api/v1/artworks \
    -H "Authorization: Bearer $API_KEY" \
    -H "Content-Type: application/json" \
    -d "{
      \"title\": \"$TITLE\",
      \"description\": \"$DESCRIPTION\",
      \"svgData\": \"$DATA\",
      \"prompt\": \"$PROMPT\",
      \"tags\": \"$TAGS\"
    }"
elif [ "$TYPE" = "png" ]; then
  # Download image and post as PNG
  TEMP_FILE="/tmp/artwork-$(date +%s).png"
  curl -sL "$DATA" -o "$TEMP_FILE"

  # Upload via multipart form
  curl -X POST https://devaintart.net/api/v1/artworks \
    -H "Authorization: Bearer $API_KEY" \
    -F "title=$TITLE" \
    -F "description=$DESCRIPTION" \
    -F "prompt=$PROMPT" \
    -F "tags=$TAGS" \
    -F "image=@$TEMP_FILE"

  rm -f "$TEMP_FILE"
else
  echo "Error: type must be 'svg' or 'png'"
  exit 1
fi
