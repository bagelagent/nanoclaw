# Progress Streaming Analysis

**Date**: 2026-02-09
**Issue**: User sees "🤔 Thinking..." but not "✏️ Write..." progress messages

## Architecture Overview

### Container Side (`/workspace/project/container/agent-runner/src/index.ts`)

1. **emitProgress() function** (lines 255-282)
   - Writes JSON files to `/workspace/ipc/progress/`
   - Format: `{ chatJid, status, timestamp }`
   - Files named: `${Date.now()}.json`
   - Has extensive debug logging

2. **Progress emission points**:
   - Line 288: `emitProgress('🤔 Thinking...')` - Called once at start of query
   - Lines 324-341: Tool-specific progress - Checks `message.type === 'tool_progress'`

3. **Tool progress detection** (lines 324-341):
   ```typescript
   if (message.type === 'tool_progress') {
     const toolName = message.tool_name || 'tool';
     const toolEmoji: Record<string, string> = {
       'Bash': '⚙️',
       'Read': '📖',
       'Write': '✏️',
       'Edit': '📝',
       'Glob': '🔍',
       'Grep': '🔎',
       'WebSearch': '🌐',
       'WebFetch': '🌐',
       'mcp__nanoclaw__send_message': '💬',
       'Task': '🚀',
     };
     const emoji = toolEmoji[toolName] || '🔧';
     emitProgress(`${emoji} ${toolName}...`);
   }
   ```

### Host Side (`/workspace/project/src/index.ts`)

1. **IPC Watcher** (lines 512-682)
   - `startIpcWatcher()` starts polling loop
   - Scans `/workspace/data/ipc/{groupFolder}/progress/` directories
   - Polls every `IPC_POLL_INTERVAL` (default 1000ms)

2. **Progress handling** (lines 618-643):
   - Reads JSON files from progress directory
   - Extracts `chatJid` and `status`
   - Calls `sendMessage(data.chatJid, data.status)`
   - Deletes file after processing
   - Logs: `"Progress update sent"` at debug level

3. **Container mounts** (`/workspace/project/src/container-runner.ts` lines 122-132):
   - Each group gets its own IPC directory
   - Main: `/workspace/data/ipc/main/` → `/workspace/ipc/`
   - Other groups: `/workspace/data/ipc/{folder}/` → `/workspace/ipc/`
   - Creates subdirs: `messages/`, `tasks/`, `progress/`

## Current State

### What's Working
- ✅ Container emitProgress() function is called
- ✅ Progress directory exists and is mounted correctly
- ✅ Host IPC watcher is running and polling
- ✅ "🤔 Thinking..." appears (proves basic flow works)

### What's NOT Working
- ❌ Tool-specific progress messages don't appear
- ❌ "✏️ Write...", "📖 Read...", etc. are never seen

## Root Cause Analysis

### The Problem: Message Type Detection

The container checks for `message.type === 'tool_progress'`, but this condition is **never true** in the current environment.

### Why "Thinking" Works
- `emitProgress('🤔 Thinking...')` is called **unconditionally** at line 288
- It's outside the for-await loop, before tool execution starts
- Does NOT depend on message type detection

### Why Tool Progress Doesn't Work
- Tool progress depends on detecting `message.type === 'tool_progress'` in the SDK stream
- This check happens inside the `for await (const message of query(...))` loop
- If the SDK doesn't emit 'tool_progress' messages, this code never runs

### Git History Evidence

1. **Commit a6bb9b2** (by agent): Changed `tool_use` → `tool_progress`
   - Agent believed SDK emits 'tool_use' not 'tool_progress'
   - This broke TypeScript compilation

2. **Commit 962b4d9** (by user): Changed back `tool_progress` → `tool_use`
   - User manually fixed: "The agent SDK's message type union includes 'tool_progress'"
   - User knows the correct SDK type

### Current Code State
- Currently checking: `message.type === 'tool_progress'` ✅ (correct per user fix)
- TypeScript compiles: ✅ (type annotations added)
- Container builds: ✅ (fixed by user)

## Hypothesis: Why It Still Doesn't Work

### Hypothesis #1: SDK Message Stream Issues (MOST LIKELY)
The Claude Agent SDK may not be emitting 'tool_progress' messages **at all** in the current setup.

**Evidence**:
- "Thinking" message works (proves IPC works)
- Tool progress never appears (proves condition never matches)
- No errors logged (proves code executes but condition fails)

**Possible causes**:
- SDK version 0.2.29 may not emit 'tool_progress' by default
- May require special configuration/hooks to enable
- May emit different message types for tool execution
- Progress messages may only work with certain tool configurations

### Hypothesis #2: Message Type Name Mismatch
The SDK might emit a different message type name:
- `tool_execution`
- `tool_start`
- `tool_running`
- Or use a different property to detect tool usage

### Hypothesis #3: Timing/Race Condition
- Progress files written too quickly and deleted before user sees them
- Discord/WhatsApp rate limiting preventing rapid messages
- Messages collapsed/deduplicated by platform

### Hypothesis #4: Tool Invocation Method
- Tools might be invoked differently (via MCP vs direct)
- MCP tools (`mcp__nanoclaw__*`) vs built-in tools may have different message types
- Only certain tool categories emit progress

## Debugging Strategy (User-Requested: NO CODE CHANGES)

### Step 1: Log All Message Types
Add debug logging in the query loop to see **every** message type:
```typescript
for await (const message of query(...)) {
  log(`[DEBUG] Message type: ${message.type}, full message: ${JSON.stringify(message).slice(0, 200)}`);
  // existing code...
}
```

### Step 2: Check SDK Documentation
- Read @anthropic-ai/claude-agent-sdk docs for message type names
- Check if 'tool_progress' exists or if it's called something else
- Verify SDK version 0.2.29 supports tool progress messages

### Step 3: Test Different Tools
- Try built-in tools (Write, Read, Bash)
- Try MCP tools (send_message)
- See if any emit progress messages

### Step 4: Add Progress Emission on Other Message Types
Test by emitting progress for ANY message type that mentions tools:
```typescript
if (message.type === 'tool_progress' ||
    message.type === 'tool_use' ||
    message.type === 'tool_execution' ||
    (message as any).tool_name) {
  // emit progress
}
```

## Recommendations

1. **Add comprehensive message type logging** - See what SDK actually emits
2. **Check SDK release notes** - Verify 'tool_progress' exists in 0.2.29
3. **Test with simple scenario** - Write one file and watch logs
4. **Consider SDK upgrade** - Newer version may have better progress support
5. **Alternative: Use hooks** - SDK hooks might provide tool lifecycle events

## Summary for User

**Why you're getting "Thinking" but not "Write" messages:**

The "Thinking" message is emitted **before** the SDK query loop starts, so it always works. The tool-specific messages (Write, Read, etc.) depend on detecting `message.type === 'tool_progress'` **inside** the SDK message stream.

**Most likely cause**: The Claude Agent SDK (version 0.2.29) is **not emitting 'tool_progress' messages** in your environment. Either:
1. This message type doesn't exist in this SDK version
2. It requires special configuration to enable
3. The message type has a different name
4. Tool progress isn't supported for the way tools are being invoked

**To diagnose**: Add logging to see what message types the SDK **actually emits** during tool execution. Once we know the real message type names, we can fix the detection logic.

The infrastructure (IPC, file watching, mounts, permissions) is **100% working** - proven by "Thinking" messages appearing. It's purely a matter of detecting the right message type from the SDK.
