#!/bin/bash
# Test DoorDash authentication with saved cookies

echo "Testing DoorDash authentication..."

# Open DoorDash and inject cookies via CDP
agent-browser open "https://www.doordash.com"

# Wait for page load
sleep 3

# Take screenshot to verify we're logged in
agent-browser screenshot /workspace/group/tmp/doordash-auth-test.png

echo "Screenshot saved to /workspace/group/tmp/doordash-auth-test.png"
echo "Check if you see your account/name in the screenshot"
