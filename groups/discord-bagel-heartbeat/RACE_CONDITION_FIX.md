# Race Condition Fix - Groups Responding to Wrong Messages

## Problem Description

**Symptom**: The bagel-heartbeat group responded with content that was clearly meant for the main group (game board reorganization, Railway deployment).

**Root Cause**: The container processes queries from multiple groups sequentially via a for-await loop, BUT the async nature of the SDK's `query()` function allows queries to overlap:

1. Query from MAIN group arrives → starts processing
2. While MAIN query is still running (waiting on tool execution, etc.), a query from BAGEL-HEARTBEAT group arrives
3. Both queries are now "in flight" even though the main loop is supposed to serialize them
4. The IpcMcp context (which contains `chatJid` determining where messages go) gets mixed up
5. MAIN's response gets sent to BAGEL-HEARTBEAT's chatJid

## The Fix

Add a **mutex/lock** to `processQuery()` to ensure ONLY ONE query can execute at a time, even if they're async operations:

```typescript
// CRITICAL: Mutex to prevent concurrent query processing
let queryInProgress = false;
const queryQueue: Array<() => void> = [];

async function processQuery(input: ContainerInput): Promise<ContainerOutput> {
  // Acquire lock - wait if another query is in progress
  if (queryInProgress) {
    await new Promise<void>(resolve => {
      queryQueue.push(resolve);
    });
  }
  queryInProgress = true;

  try {
    return await processQueryInternal(input);
  } finally {
    // Release lock and wake next query
    queryInProgress = false;
    const next = queryQueue.shift();
    if (next) next();
  }
}
```

This ensures queries are TRULY serialized - the second query will wait until the first query COMPLETELY finishes (including all tool executions and message sending) before it starts.

## Files to Modify

1. **`/workspace/project/container/agent-runner/src/index.ts`** (or wherever the source is on the host)
   - Rename current `processQuery` to `processQueryInternal`
   - Add the mutex wrapper as shown above
   - Add logging to track query start/end with group identification

2. Apply the patch in `FIX-race-condition.patch`

## How to Deploy

1. **From the host machine** (not from inside a container):

```bash
cd /workspace/project  # or wherever your NanoClaw project root is

# Apply the fix to container source
cd container/agent-runner
# Edit src/index.ts with the fix above

# Type-check
npx tsc --noEmit

# Rebuild container
cd ../..
./container/build.sh

# Restart the service
sudo systemctl restart nanoclaw
```

2. **OR** Use the deploy tool from main group:

From the main WhatsApp group, send:
```
@Bagel deploy the race condition fix. targets: container
```

Then the agent will:
- Commit changes
- Rebuild the container
- Restart the service

## Restart Service Manually

If you need to restart NanoClaw immediately:

```bash
# On the host machine
sudo systemctl restart nanoclaw

# Check status
sudo systemctl status nanoclaw

# View logs
sudo journalctl -u nanoclaw -f
```

## Verification

After deploying, the logs should show:
```
[QUERY_START] group=main chatJid=... isMain=true
...
[QUERY_END] group=main status=success
[QUERY_START] group=discord-bagel-heartbeat chatJid=... isMain=false
...
[QUERY_END] group=discord-bagel-heartbeat status=success
```

No query should start before the previous one ends.

## Additional Safeguards

Consider also adding:
1. **Request ID tracking** in IPC messages to detect misrouted messages
2. **Separate containers per group** instead of sharing one container
3. **Validation** in `writeIpcFile` to ensure chatJid matches the expected group

## Emergency Mitigation

If the issue persists after the fix:
1. Restart the nanoclaw service
2. Check if multiple containers are running: `docker ps | grep nanoclaw`
3. Check host logs: `sudo journalctl -u nanoclaw -n 100`
4. Temporarily disable non-main groups until root cause is confirmed fixed
