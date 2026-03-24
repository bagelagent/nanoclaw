# NanoClaw Agent Container
# Runs Claude Code CLI in tmux with MCP tools for IPC

FROM node:22-slim

# Install system dependencies for Chromium, tmux, and GitHub CLI
RUN apt-get update && apt-get install -y \
    chromium \
    fonts-liberation \
    fonts-noto-cjk \
    fonts-noto-color-emoji \
    libgbm1 \
    libnss3 \
    libatk-bridge2.0-0 \
    libgtk-3-0 \
    libx11-xcb1 \
    libxcomposite1 \
    libxdamage1 \
    libxrandr2 \
    libasound2 \
    libpangocairo-1.0-0 \
    libcups2 \
    libdrm2 \
    libxshmfence1 \
    curl \
    git \
    postgresql \
    postgresql-client \
    tmux \
    jq \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Install GitHub CLI
RUN curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg \
    && chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
    && apt-get update \
    && apt-get install -y gh \
    && rm -rf /var/lib/apt/lists/*

# Set Chromium path for agent-browser
ENV AGENT_BROWSER_EXECUTABLE_PATH=/usr/bin/chromium
ENV PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH=/usr/bin/chromium

# Install agent-browser, claude-code, and MCP servers globally
RUN npm install -g agent-browser @anthropic-ai/claude-code @notionhq/notion-mcp-server figma-developer-mcp

# Create app directory for MCP server
WORKDIR /app

# Copy MCP server package files first for better caching
COPY agent-runner/package*.json ./

# Install dependencies (MCP server deps only)
RUN npm install --legacy-peer-deps

# Copy MCP server source code
COPY agent-runner/ ./

# Build TypeScript (MCP server)
RUN npm run build

# Install Playwright globally and download browsers (as root)
RUN npm install -g @playwright/test && \
    npx playwright install --with-deps chromium && \
    mkdir -p /home/node/.cache && \
    cp -r /root/.cache/ms-playwright /home/node/.cache/ && \
    chown -R node:node /home/node/.cache

# Enable Claude Code auto-updater: make global npm dirs writable by node user
RUN chown node:node /usr/local && \
    chown -R node:node /usr/local/lib/node_modules /usr/local/bin

# Create workspace directories
RUN mkdir -p /workspace/group /workspace/global /workspace/extra /workspace/ipc/messages /workspace/ipc/tasks /workspace/ipc/input /workspace/ipc/replies /workspace/ipc/progress

# Create entrypoint script that starts tmux with Claude Code
RUN cat > /app/entrypoint.sh << 'ENTRYPOINT_EOF'
#!/bin/bash
set -e

# Source environment variables
[ -f /workspace/env-dir/env ] && export $(cat /workspace/env-dir/env | xargs)

# Configure gh CLI if GITHUB_TOKEN is set
if [ -n "$GITHUB_TOKEN" ]; then
  export GH_TOKEN="$GITHUB_TOKEN"
fi

# Ensure onboarding is marked complete (host should write .claude.json, but belt-and-suspenders)
if [ ! -f "$HOME/.claude.json" ]; then
  echo '{"hasCompletedOnboarding":true,"hasAcknowledgedCostThreshold":true}' > "$HOME/.claude.json"
fi

# Pre-accept the bypass permissions dialog
# Use cp instead of mv since these may be bind-mounted files
if [ -f "$HOME/.claude.json" ]; then
  jq '. + {"hasDangerouslySkipPermissionsAccepted": true}' \
    "$HOME/.claude.json" > /tmp/claude.tmp && cp /tmp/claude.tmp "$HOME/.claude.json" && rm /tmp/claude.tmp
fi

# Pre-accept the workspace trust prompt and skip bypass permissions dialog
if [ -f "$HOME/.claude/settings.json" ]; then
  jq '. + {"trustedDirectories": ["/workspace/group", "/workspace/project"], "skipDangerousModePermissionPrompt": true}' \
    "$HOME/.claude/settings.json" > /tmp/settings.tmp && cp /tmp/settings.tmp "$HOME/.claude/settings.json" && rm /tmp/settings.tmp
fi

# Start PostgreSQL if needed
pg_ctl -D /home/node/pgdata -l /tmp/pg.log start 2>/dev/null || true

# Start tmux with Claude Code
# NOTE: First-run prompts (bypass permissions, trust folder) must be accepted
# once manually. The acceptance is saved to the mounted ~/.claude.json.
# After that, subsequent launches skip the prompts automatically.
tmux new-session -d -s claude -x 200 -y 50 \
  "claude --continue --dangerously-skip-permissions || claude --dangerously-skip-permissions"

# Keep container alive; exit when tmux session dies
while tmux has-session -t claude 2>/dev/null; do sleep 5; done
ENTRYPOINT_EOF
RUN chmod +x /app/entrypoint.sh

# Set ownership to node user (non-root) for writable directories
RUN chown -R node:node /workspace && chmod 777 /home/node

# Switch to non-root user (required for --dangerously-skip-permissions)
USER node

# Initialize PostgreSQL data directory for node user
ENV PATH="/usr/lib/postgresql/15/bin:$PATH"
ENV PGDATA=/home/node/pgdata
ENV PGHOST=localhost
RUN initdb -D /home/node/pgdata && \
    echo "host all all 127.0.0.1/32 trust" >> /home/node/pgdata/pg_hba.conf && \
    echo "local all all trust" >> /home/node/pgdata/pg_hba.conf && \
    echo "unix_socket_directories = '/tmp'" >> /home/node/pgdata/postgresql.conf && \
    pg_ctl -D /home/node/pgdata -l /tmp/pg.log start && \
    createdb node && \
    pg_ctl -D /home/node/pgdata stop

# Set working directory to group workspace
WORKDIR /workspace/group

# Entry point starts tmux with Claude Code
ENTRYPOINT ["/app/entrypoint.sh"]
