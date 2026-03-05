/**
 * Container Runner for NanoClaw
 * Manages persistent Docker containers per group and sends queries over stdin/stdout.
 */
import { ChildProcess, exec, spawn } from 'child_process';
import fs from 'fs';
import path from 'path';

import {
  AGENT_MODEL,
  CONTAINER_IMAGE,
  CONTAINER_MAX_OUTPUT_SIZE,
  CONTAINER_TIMEOUT,
  DATA_DIR,
  GROUPS_DIR,
  IDLE_TIMEOUT,
  TIMEZONE,
} from './config.js';
import { readEnvFile } from './env.js';
import { resolveGroupFolderPath, resolveGroupIpcPath } from './group-folder.js';
import { logger } from './logger.js';
import {
  CONTAINER_RUNTIME_BIN,
  readonlyMountArgs,
  stopContainer,
} from './container-runtime.js';
import { validateAdditionalMounts } from './mount-security.js';
import { RegisteredGroup } from './types.js';

// Sentinel markers for robust output parsing (must match agent-runner)
// These markers wrap the JSON output from the container to distinguish it from debug logs
const OUTPUT_START_MARKER = '---NANOCLAW_OUTPUT_START---';
const OUTPUT_END_MARKER = '---NANOCLAW_OUTPUT_END---';

const IDLE_TIMEOUT_MS = 5 * 60 * 1000; // 5 minutes
const MAX_QUERY_DURATION = 2 * 60 * 60 * 1000; // 2 hours hard cap

export interface ContainerInput {
  prompt: string;
  sessionId?: string;
  groupFolder: string;
  chatJid: string;
  isMain: boolean;
  model?: string;
  isScheduledTask?: boolean;
  assistantName?: string;
  secrets?: Record<string, string>;
}

export interface ContainerOutput {
  status: 'success' | 'error';
  result: string | null;
  newSessionId?: string;
  error?: string;
}

interface VolumeMount {
  hostPath: string;
  containerPath: string;
  readonly: boolean;
}

function buildVolumeMounts(
  group: RegisteredGroup,
  isMain: boolean,
): VolumeMount[] {
  const mounts: VolumeMount[] = [];
  const projectRoot = process.cwd();
  const groupDir = resolveGroupFolderPath(group.folder);

  if (isMain) {
    // Main gets the project root read-write so the trusted main agent can
    // self-modify (edit source, build, deploy). Non-main groups never get
    // the project mount at all.
    mounts.push({
      hostPath: projectRoot,
      containerPath: '/workspace/project',
      readonly: false,
    });

    // Shadow .env so the agent cannot read secrets from the mounted project root.
    // Secrets are passed via stdin instead (see readSecrets()).
    const envFile = path.join(projectRoot, '.env');
    if (fs.existsSync(envFile)) {
      mounts.push({
        hostPath: '/dev/null',
        containerPath: '/workspace/project/.env',
        readonly: true,
      });
    }

    // Main also gets its group folder as the working directory
    mounts.push({
      hostPath: groupDir,
      containerPath: '/workspace/group',
      readonly: false,
    });
  } else {
    // Other groups only get their own folder
    mounts.push({
      hostPath: groupDir,
      containerPath: '/workspace/group',
      readonly: false,
    });

    // Global memory directory (read-only for non-main)
    // Only directory mounts are supported, not file mounts
    const globalDir = path.join(GROUPS_DIR, 'global');
    if (fs.existsSync(globalDir)) {
      mounts.push({
        hostPath: globalDir,
        containerPath: '/workspace/global',
        readonly: true,
      });
    }
  }

  // Per-group Claude sessions directory (isolated from other groups)
  // Each group gets their own .claude/ to prevent cross-group session access
  const groupSessionsDir = path.join(
    DATA_DIR,
    'sessions',
    group.folder,
    '.claude',
  );
  fs.mkdirSync(groupSessionsDir, { recursive: true });
  const settingsFile = path.join(groupSessionsDir, 'settings.json');
  if (!fs.existsSync(settingsFile)) {
    fs.writeFileSync(
      settingsFile,
      JSON.stringify(
        {
          env: {
            // Enable agent swarms (subagent orchestration)
            // https://code.claude.com/docs/en/agent-teams#orchestrate-teams-of-claude-code-sessions
            CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS: '1',
            // Load CLAUDE.md from additional mounted directories
            // https://code.claude.com/docs/en/memory#load-memory-from-additional-directories
            CLAUDE_CODE_ADDITIONAL_DIRECTORIES_CLAUDE_MD: '1',
            // Enable Claude's memory feature (persists user preferences between sessions)
            // https://code.claude.com/docs/en/memory#manage-auto-memory
            CLAUDE_CODE_DISABLE_AUTO_MEMORY: '0',
          },
        },
        null,
        2,
      ) + '\n',
    );
  }

  // Sync skills from container/skills/ into each group's .claude/skills/
  const skillsSrc = path.join(process.cwd(), 'container', 'skills');
  const skillsDst = path.join(groupSessionsDir, 'skills');
  if (fs.existsSync(skillsSrc)) {
    for (const skillDir of fs.readdirSync(skillsSrc)) {
      const srcDir = path.join(skillsSrc, skillDir);
      if (!fs.statSync(srcDir).isDirectory()) continue;
      const dstDir = path.join(skillsDst, skillDir);
      fs.cpSync(srcDir, dstDir, { recursive: true });
    }
  }
  mounts.push({
    hostPath: groupSessionsDir,
    containerPath: '/home/node/.claude',
    readonly: false,
  });

  // Per-group IPC namespace: each group gets its own IPC directory
  // This prevents cross-group privilege escalation via IPC
  const groupIpcDir = resolveGroupIpcPath(group.folder);
  fs.mkdirSync(path.join(groupIpcDir, 'messages'), { recursive: true });
  fs.mkdirSync(path.join(groupIpcDir, 'tasks'), { recursive: true });
  fs.mkdirSync(path.join(groupIpcDir, 'progress'), { recursive: true });
  fs.mkdirSync(path.join(groupIpcDir, 'replies'), { recursive: true });
  fs.mkdirSync(path.join(groupIpcDir, 'input'), { recursive: true });
  mounts.push({
    hostPath: groupIpcDir,
    containerPath: '/workspace/ipc',
    readonly: false,
  });

  // Logs directory (read-only) - allows agent to debug by reading previous query logs
  const logsHostPath = path.join(GROUPS_DIR, group.folder, 'logs');
  fs.mkdirSync(logsHostPath, { recursive: true });
  mounts.push({
    hostPath: logsHostPath,
    containerPath: '/workspace/logs',
    readonly: true,
  });

  // Environment file directory
  // Only expose specific auth variables needed by Claude Code, not the entire .env
  const envDir = path.join(DATA_DIR, 'env');
  fs.mkdirSync(envDir, { recursive: true });
  const envFile = path.join(projectRoot, '.env');
  if (fs.existsSync(envFile)) {
    const envContent = fs.readFileSync(envFile, 'utf-8');
    const allowedVars = [
      'CLAUDE_CODE_OAUTH_TOKEN',
      'ANTHROPIC_API_KEY',
      'OPENAI_API_KEY',
    ];
    const filteredLines = envContent.split('\n').filter((line) => {
      const trimmed = line.trim();
      if (!trimmed || trimmed.startsWith('#')) return false;
      return allowedVars.some((v) => trimmed.startsWith(`${v}=`));
    });

    if (filteredLines.length > 0) {
      fs.writeFileSync(
        path.join(envDir, 'env'),
        filteredLines.join('\n') + '\n',
      );
      mounts.push({
        hostPath: envDir,
        containerPath: '/workspace/env-dir',
        readonly: true,
      });
    }
  }

  // Copy agent-runner source into a per-group writable location so agents
  // can customize it (add tools, change behavior) without affecting other
  // groups. Recompiled on container startup via entrypoint.sh.
  const agentRunnerSrc = path.join(
    projectRoot,
    'container',
    'agent-runner',
    'src',
  );
  const groupAgentRunnerDir = path.join(
    DATA_DIR,
    'sessions',
    group.folder,
    'agent-runner-src',
  );
  if (!fs.existsSync(groupAgentRunnerDir) && fs.existsSync(agentRunnerSrc)) {
    fs.cpSync(agentRunnerSrc, groupAgentRunnerDir, { recursive: true });
  }
  mounts.push({
    hostPath: groupAgentRunnerDir,
    containerPath: '/app/src',
    readonly: false,
  });

  // Embeddings DB for semantic memory (read-only, shared across all groups)
  const embeddingsDbPath = path.join(DATA_DIR, 'embeddings.db');
  if (fs.existsSync(embeddingsDbPath)) {
    mounts.push({
      hostPath: embeddingsDbPath,
      containerPath: '/workspace/embeddings.db',
      readonly: true,
    });
  }

  // Additional mounts validated against external allowlist (tamper-proof from containers)
  if (group.containerConfig?.additionalMounts) {
    const validatedMounts = validateAdditionalMounts(
      group.containerConfig.additionalMounts,
      group.name,
      isMain,
    );
    mounts.push(...validatedMounts);
  }

  return mounts;
}

/**
 * Read allowed secrets from .env for passing to the container via stdin.
 * Secrets are never written to disk or mounted as files.
 */
function readSecrets(): Record<string, string> {
  return readEnvFile([
    'CLAUDE_CODE_OAUTH_TOKEN',
    'ANTHROPIC_API_KEY',
    'ANTHROPIC_BASE_URL',
    'ANTHROPIC_AUTH_TOKEN',
  ]);
}

function buildContainerArgs(
  mounts: VolumeMount[],
  containerName: string,
): string[] {
  const args: string[] = ['run', '-i', '--rm', '--name', containerName];

  // Pass host timezone so container's local time matches the user's
  args.push('-e', `TZ=${TIMEZONE}`);

  // Run as host user so bind-mounted files are accessible.
  // Skip when running as root (uid 0), as the container's node user (uid 1000),
  // or when getuid is unavailable (native Windows without WSL).
  const hostUid = process.getuid?.();
  const hostGid = process.getgid?.();
  if (hostUid != null && hostUid !== 0 && hostUid !== 1000) {
    args.push('--user', `${hostUid}:${hostGid}`);
    args.push('-e', 'HOME=/home/node');
  }

  for (const mount of mounts) {
    if (mount.readonly) {
      args.push(...readonlyMountArgs(mount.hostPath, mount.containerPath));
    } else {
      args.push('-v', `${mount.hostPath}:${mount.containerPath}`);
    }
  }

  args.push(CONTAINER_IMAGE);

  return args;
}

// ─── Container Pool ──────────────────────────────────────────────────────────

interface QueuedQuery {
  input: ContainerInput;
  resolve: (output: ContainerOutput) => void;
}

interface PoolEntry {
  process: ChildProcess;
  containerName: string;
  group: RegisteredGroup;
  isMain: boolean;
  lastUsed: number;
  idleTimer: NodeJS.Timeout;
  /** Stderr buffer for diagnostics */
  stderr: string;
  stderrTruncated: boolean;
  /**
   * Stdout accumulator between sentinel markers.
   * Data arrives in chunks; we buffer until we see both markers.
   */
  stdoutBuf: string;
  /** Whether the container process has exited */
  exited: boolean;
  exitCode: number | null;
  /**
   * If a query is in flight, this holds the resolve callback.
   * Only one query can be active per container at a time.
   */
  pendingResolve: ((output: ContainerOutput) => void) | null;
  pendingTimeout: NodeJS.Timeout | null;
  hardDeadlineTimeout: NodeJS.Timeout | null;
  /** Queries waiting for the current one to finish */
  queryQueue: QueuedQuery[];
}

class ContainerPool {
  private pool = new Map<string, PoolEntry>();

  /**
   * Get or spawn a persistent container for a group.
   */
  getOrSpawn(group: RegisteredGroup, isMain: boolean): PoolEntry {
    const key = group.folder;
    const existing = this.pool.get(key);
    if (existing && !existing.exited) {
      return existing;
    }

    // Clean up dead entry
    if (existing) {
      this.pool.delete(key);
    }

    const groupDir = resolveGroupFolderPath(group.folder);
    fs.mkdirSync(groupDir, { recursive: true });

    const mounts = buildVolumeMounts(group, isMain);
    const safeName = group.folder.replace(/[^a-zA-Z0-9-]/g, '-');
    const containerName = `nanoclaw-${safeName}-${Date.now()}`;
    const containerArgs = buildContainerArgs(mounts, containerName);

    logger.debug(
      {
        group: group.name,
        containerName,
        mounts: mounts.map(
          (m) =>
            `${m.hostPath} -> ${m.containerPath}${m.readonly ? ' (ro)' : ''}`,
        ),
      },
      'Container mount configuration',
    );

    logger.info(
      {
        group: group.name,
        containerName,
        mountCount: mounts.length,
        isMain,
      },
      'Spawning persistent container',
    );

    const container = spawn(CONTAINER_RUNTIME_BIN, containerArgs, {
      stdio: ['pipe', 'pipe', 'pipe'],
    });

    const entry: PoolEntry = {
      process: container,
      containerName,
      group,
      isMain,
      lastUsed: Date.now(),
      idleTimer: setTimeout(() => this.shutdownEntry(key), IDLE_TIMEOUT_MS),
      stderr: '',
      stderrTruncated: false,
      stdoutBuf: '',
      exited: false,
      exitCode: null,
      pendingResolve: null,
      pendingTimeout: null,
      hardDeadlineTimeout: null,
      queryQueue: [],
    };

    // Stream stderr for logging/diagnostics
    container.stderr.on('data', (data) => {
      const chunk = data.toString();

      // Reset activity timeout on any stderr output (SDK messages flow here)
      if (entry.pendingResolve && entry.pendingTimeout) {
        clearTimeout(entry.pendingTimeout);
        entry.pendingTimeout = this.createActivityTimeout(entry);
      }

      const lines = chunk.trim().split('\n');
      for (const line of lines) {
        if (line) logger.debug({ container: group.folder }, line);
      }
      if (entry.stderrTruncated) return;
      const remaining = CONTAINER_MAX_OUTPUT_SIZE - entry.stderr.length;
      if (chunk.length > remaining) {
        entry.stderr += chunk.slice(0, remaining);
        entry.stderrTruncated = true;
      } else {
        entry.stderr += chunk;
      }
    });

    // Stream stdout — look for sentinel-delimited output per query
    container.stdout.on('data', (data) => {
      entry.stdoutBuf += data.toString();
      this.checkForOutput(entry);
    });

    container.on('close', (code) => {
      entry.exited = true;
      entry.exitCode = code;
      clearTimeout(entry.idleTimer);

      logger.info(
        { group: group.name, containerName, code },
        'Persistent container exited',
      );

      // If a query was pending, resolve with error
      if (entry.pendingResolve) {
        if (entry.pendingTimeout) clearTimeout(entry.pendingTimeout);
        if (entry.hardDeadlineTimeout) clearTimeout(entry.hardDeadlineTimeout);
        entry.pendingResolve({
          status: 'error',
          result: null,
          error: `Container exited unexpectedly with code ${code}: ${entry.stderr.slice(-200)}`,
        });
        entry.pendingResolve = null;
        entry.pendingTimeout = null;
        entry.hardDeadlineTimeout = null;
      }

      // Reject any queued queries
      this.drainQueue(entry);

      this.pool.delete(key);
    });

    container.on('error', (err) => {
      entry.exited = true;
      clearTimeout(entry.idleTimer);
      logger.error(
        { group: group.name, containerName, error: err },
        'Container spawn error',
      );

      if (entry.pendingResolve) {
        if (entry.pendingTimeout) clearTimeout(entry.pendingTimeout);
        if (entry.hardDeadlineTimeout) clearTimeout(entry.hardDeadlineTimeout);
        entry.pendingResolve({
          status: 'error',
          result: null,
          error: `Container spawn error: ${err.message}`,
        });
        entry.pendingResolve = null;
        entry.pendingTimeout = null;
        entry.hardDeadlineTimeout = null;
      }

      // Reject any queued queries
      this.drainQueue(entry);

      this.pool.delete(key);
    });

    this.pool.set(key, entry);
    return entry;
  }

  /**
   * Check if stdout buffer contains a complete sentinel-delimited response.
   */
  private checkForOutput(entry: PoolEntry): void {
    if (!entry.pendingResolve) return;

    const startIdx = entry.stdoutBuf.indexOf(OUTPUT_START_MARKER);
    const endIdx = entry.stdoutBuf.indexOf(OUTPUT_END_MARKER);

    if (startIdx === -1 || endIdx === -1 || endIdx <= startIdx) return;

    // Extract the JSON between markers
    const jsonStr = entry.stdoutBuf
      .slice(startIdx + OUTPUT_START_MARKER.length, endIdx)
      .trim();

    // Remove the consumed output from buffer (keep anything after end marker)
    entry.stdoutBuf = entry.stdoutBuf.slice(endIdx + OUTPUT_END_MARKER.length);

    const resolve = entry.pendingResolve;
    entry.pendingResolve = null;
    if (entry.pendingTimeout) {
      clearTimeout(entry.pendingTimeout);
      entry.pendingTimeout = null;
    }
    if (entry.hardDeadlineTimeout) {
      clearTimeout(entry.hardDeadlineTimeout);
      entry.hardDeadlineTimeout = null;
    }

    try {
      const output: ContainerOutput = JSON.parse(jsonStr);
      resolve(output);
    } catch (err) {
      logger.error(
        { group: entry.group.name, jsonStr: jsonStr.slice(0, 200), error: err },
        'Failed to parse container output from persistent container',
      );
      resolve({
        status: 'error',
        result: null,
        error: `Failed to parse container output: ${err instanceof Error ? err.message : String(err)}`,
      });
    }

    // Dispatch next queued query if any
    this.drainQueue(entry);
  }

  /**
   * Dispatch the next queued query to the container, if any.
   */
  private drainQueue(entry: PoolEntry): void {
    if (entry.pendingResolve || entry.queryQueue.length === 0) return;
    if (entry.exited) {
      // Container is gone — reject all queued queries
      for (const q of entry.queryQueue) {
        q.resolve({
          status: 'error',
          result: null,
          error: `Container exited before query could be dispatched`,
        });
      }
      entry.queryQueue = [];
      return;
    }

    const next = entry.queryQueue.shift()!;
    logger.debug(
      { group: entry.group.name, remaining: entry.queryQueue.length },
      'Dispatching queued query',
    );
    this.dispatchQuery(entry, next.input, next.resolve);
  }

  /**
   * Create an activity-based timeout that fires after CONTAINER_TIMEOUT ms
   * of inactivity. Each stderr chunk resets this timer.
   */
  private createActivityTimeout(entry: PoolEntry): NodeJS.Timeout {
    return setTimeout(() => {
      if (entry.pendingResolve) {
        logger.error(
          { group: entry.group.name, containerName: entry.containerName },
          'Query activity timeout — no container output',
        );
        entry.pendingResolve({
          status: 'error',
          result: null,
          error: `Query timed out after ${CONTAINER_TIMEOUT}ms of inactivity`,
        });
        entry.pendingResolve = null;
        entry.pendingTimeout = null;
        if (entry.hardDeadlineTimeout) {
          clearTimeout(entry.hardDeadlineTimeout);
          entry.hardDeadlineTimeout = null;
        }
        this.shutdownEntry(entry.group.folder);
      }
    }, CONTAINER_TIMEOUT);
  }

  /**
   * Send a query to a persistent container and wait for the response.
   */
  sendQuery(entry: PoolEntry, input: ContainerInput): Promise<ContainerOutput> {
    return new Promise((resolve) => {
      if (entry.exited) {
        resolve({
          status: 'error',
          result: null,
          error: `Container already exited with code ${entry.exitCode}`,
        });
        return;
      }

      // If a query is already in flight, queue this one instead of overwriting
      if (entry.pendingResolve) {
        logger.debug(
          { group: entry.group.name, containerName: entry.containerName },
          'Query already in flight, queueing',
        );
        entry.queryQueue.push({ input, resolve });
        return;
      }

      this.dispatchQuery(entry, input, resolve);
    });
  }

  /**
   * Actually dispatch a query to the container's stdin.
   */
  private dispatchQuery(
    entry: PoolEntry,
    input: ContainerInput,
    resolve: (output: ContainerOutput) => void,
  ): void {
    // Reset per-query stderr buffer for logging
    entry.stderr = '';
    entry.stderrTruncated = false;

    entry.pendingResolve = resolve;
    entry.lastUsed = Date.now();

    // Reset idle timer
    clearTimeout(entry.idleTimer);
    entry.idleTimer = setTimeout(
      () => this.shutdownEntry(entry.group.folder),
      IDLE_TIMEOUT_MS,
    );

    // Activity-based timeout: resets on each stderr chunk from container
    entry.pendingTimeout = this.createActivityTimeout(entry);

    // Hard wall-clock deadline as safety cap
    entry.hardDeadlineTimeout = setTimeout(() => {
      if (entry.pendingResolve) {
        logger.error(
          { group: entry.group.name, containerName: entry.containerName },
          'Query hard deadline reached',
        );
        entry.pendingResolve({
          status: 'error',
          result: null,
          error: `Query exceeded maximum duration of ${MAX_QUERY_DURATION}ms`,
        });
        entry.pendingResolve = null;
        if (entry.pendingTimeout) {
          clearTimeout(entry.pendingTimeout);
          entry.pendingTimeout = null;
        }
        entry.hardDeadlineTimeout = null;
        this.shutdownEntry(entry.group.folder);
      }
    }, MAX_QUERY_DURATION);

    // Write JSON line to stdin
    // Attach a one-time error handler to prevent unhandled 'error' event crashes
    const onStdinError = (err: Error) => {
      if (entry.pendingResolve) {
        if (entry.pendingTimeout) clearTimeout(entry.pendingTimeout);
        if (entry.hardDeadlineTimeout) clearTimeout(entry.hardDeadlineTimeout);
        entry.pendingResolve({
          status: 'error',
          result: null,
          error: `Container stdin error: ${err.message}`,
        });
        entry.pendingResolve = null;
        entry.pendingTimeout = null;
        entry.hardDeadlineTimeout = null;
      }
    };
    entry.process.stdin!.once('error', onStdinError);

    // Pass secrets via stdin (never written to disk or mounted as files)
    const fullInput = { ...input, secrets: readSecrets() };
    const line = JSON.stringify(fullInput) + '\n';
    entry.process.stdin!.write(line, (err) => {
      if (err && entry.pendingResolve) {
        entry.process.stdin!.removeListener('error', onStdinError);
        if (entry.pendingTimeout) clearTimeout(entry.pendingTimeout);
        if (entry.hardDeadlineTimeout) clearTimeout(entry.hardDeadlineTimeout);
        entry.pendingResolve({
          status: 'error',
          result: null,
          error: `Failed to write to container stdin: ${err.message}`,
        });
        entry.pendingResolve = null;
        entry.pendingTimeout = null;
        entry.hardDeadlineTimeout = null;
      }
    });
  }

  /**
   * Gracefully shut down a single container by closing its stdin.
   * @param force If true, shut down even if a query is in flight
   */
  private shutdownEntry(key: string, force: boolean = false): void {
    const entry = this.pool.get(key);
    if (!entry || entry.exited) {
      this.pool.delete(key);
      return;
    }

    // Don't shut down if a query is in flight — reschedule the idle timer
    // Unless force=true (for explicit restarts)
    if (entry.pendingResolve && !force) {
      logger.debug(
        { group: entry.group.name, containerName: entry.containerName },
        'Skipping idle shutdown, query still in flight',
      );
      clearTimeout(entry.idleTimer);
      entry.idleTimer = setTimeout(
        () => this.shutdownEntry(key),
        IDLE_TIMEOUT_MS,
      );
      return;
    }

    logger.info(
      { group: entry.group.name, containerName: entry.containerName },
      'Shutting down idle persistent container',
    );

    clearTimeout(entry.idleTimer);
    if (entry.pendingTimeout) clearTimeout(entry.pendingTimeout);
    if (entry.hardDeadlineTimeout) clearTimeout(entry.hardDeadlineTimeout);

    // Remove from pool immediately so getOrSpawn() won't return
    // this entry while it's draining (prevents write-after-end crashes)
    this.pool.delete(key);

    // Send shutdown signal and close stdin
    try {
      entry.process.stdin!.write(JSON.stringify({ type: 'shutdown' }) + '\n');
    } catch {
      // stdin may already be closed
    }
    try {
      entry.process.stdin!.end();
    } catch {
      // ignore
    }

    // Give it a few seconds, then force kill the container directly
    setTimeout(() => {
      if (!entry.exited) {
        logger.warn(
          { group: entry.group.name, containerName: entry.containerName },
          'Container did not exit after shutdown, force killing',
        );
        exec(stopContainer(entry.containerName), { timeout: 10000 }, (err) => {
          if (err) {
            logger.error(
              { containerName: entry.containerName, error: err.message },
              'Container stop failed, trying SIGKILL on client',
            );
            entry.process.kill('SIGKILL');
          }
        });
      }
    }, 5000);
  }

  /**
   * Shut down all persistent containers. Called during process shutdown.
   */
  async shutdownAll(): Promise<void> {
    const entries = Array.from(this.pool.keys());
    for (const key of entries) {
      this.shutdownEntry(key);
    }
  }

  /**
   * Get the ChildProcess and container name for a group (if active).
   * Used by GroupQueue for shutdown tracking.
   */
  getEntry(
    groupFolder: string,
  ): { process: ChildProcess; containerName: string } | null {
    const entry = this.pool.get(groupFolder);
    if (entry && !entry.exited) {
      return { process: entry.process, containerName: entry.containerName };
    }
    return null;
  }

  /**
   * Restart a specific container by shutting it down.
   * The next query will spawn a fresh container with new code.
   * Forces shutdown even if a query is currently in flight.
   */
  restartContainer(groupFolder: string): boolean {
    const entry = this.pool.get(groupFolder);
    if (entry && !entry.exited) {
      logger.info(
        { group: entry.group.name, containerName: entry.containerName },
        'Restarting container (requested via IPC) - forcing shutdown',
      );
      this.shutdownEntry(groupFolder, true); // force=true to shutdown during query
      return true;
    }
    return false;
  }
}

// Module-level pool instance
const containerPool = new ContainerPool();

export async function runContainerAgent(
  group: RegisteredGroup,
  input: ContainerInput,
  onProcess: (proc: ChildProcess, containerName: string) => void,
  onOutput?: (output: ContainerOutput) => Promise<void>,
): Promise<ContainerOutput> {
  const startTime = Date.now();

  const logsDir = path.join(GROUPS_DIR, group.folder, 'logs');
  fs.mkdirSync(logsDir, { recursive: true });

  const entry = containerPool.getOrSpawn(group, input.isMain);

  // Always register the process with the queue so shutdown can find it
  onProcess(entry.process, entry.containerName);

  logger.info(
    {
      group: group.name,
      containerName: entry.containerName,
      isMain: input.isMain,
    },
    'Sending query to persistent container',
  );

  const fullInput = { ...input, model: AGENT_MODEL };
  const output = await containerPool.sendQuery(entry, fullInput);

  // Notify caller of output (used for streaming results to channels)
  if (onOutput && output.status === 'success' && output.result) {
    await onOutput(output);
  }

  const duration = Date.now() - startTime;

  // Write log file
  const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
  const logFile = path.join(logsDir, `container-${timestamp}.log`);
  const isVerbose =
    process.env.LOG_LEVEL === 'debug' || process.env.LOG_LEVEL === 'trace';

  const logLines = [
    `=== Container Query Log ===`,
    `Timestamp: ${new Date().toISOString()}`,
    `Group: ${group.name}`,
    `Container: ${entry.containerName}`,
    `IsMain: ${input.isMain}`,
    `Duration: ${duration}ms`,
    `Status: ${output.status}`,
    ``,
  ];

  if (isVerbose || output.status === 'error') {
    logLines.push(
      `=== Input ===`,
      JSON.stringify(input, null, 2),
      ``,
      `=== Stderr${entry.stderrTruncated ? ' (TRUNCATED)' : ''} ===`,
      entry.stderr,
      ``,
      `=== Output ===`,
      JSON.stringify(output, null, 2),
    );
  } else {
    logLines.push(
      `=== Input Summary ===`,
      `Prompt length: ${input.prompt.length} chars`,
      `Session ID: ${input.sessionId || 'new'}`,
      ``,
    );
  }

  fs.writeFileSync(logFile, logLines.join('\n'));
  logger.debug({ logFile, verbose: isVerbose }, 'Container log written');

  logger.info(
    {
      group: group.name,
      duration,
      status: output.status,
      hasResult: !!output.result,
    },
    'Container query completed',
  );

  return output;
}

/**
 * Shut down all persistent containers. Called during process shutdown.
 */
export async function shutdownPool(): Promise<void> {
  await containerPool.shutdownAll();
}

/**
 * Restart a specific container to pick up new code.
 * Returns true if a container was restarted, false if none was running.
 */
export function restartContainer(groupFolder: string): boolean {
  return containerPool.restartContainer(groupFolder);
}

export function writeTasksSnapshot(
  groupFolder: string,
  isMain: boolean,
  tasks: Array<{
    id: string;
    groupFolder: string;
    prompt: string;
    schedule_type: string;
    schedule_value: string;
    status: string;
    next_run: string | null;
  }>,
): void {
  // Write filtered tasks to the group's IPC directory
  const groupIpcDir = resolveGroupIpcPath(groupFolder);
  fs.mkdirSync(groupIpcDir, { recursive: true });

  // Main sees all tasks, others only see their own
  const filteredTasks = isMain
    ? tasks
    : tasks.filter((t) => t.groupFolder === groupFolder);

  const tasksFile = path.join(groupIpcDir, 'current_tasks.json');
  fs.writeFileSync(tasksFile, JSON.stringify(filteredTasks, null, 2));
}

export interface AvailableGroup {
  jid: string;
  name: string;
  lastActivity: string;
  isRegistered: boolean;
}

/**
 * Write available groups snapshot for the container to read.
 * Only main group can see all available groups (for activation).
 * Non-main groups only see their own registration status.
 */
export function writeGroupsSnapshot(
  groupFolder: string,
  isMain: boolean,
  groups: AvailableGroup[],
  registeredJids: Set<string>,
): void {
  const groupIpcDir = resolveGroupIpcPath(groupFolder);
  fs.mkdirSync(groupIpcDir, { recursive: true });

  // Main sees all groups; others see nothing (they can't activate groups)
  const visibleGroups = isMain ? groups : [];

  const groupsFile = path.join(groupIpcDir, 'available_groups.json');
  fs.writeFileSync(
    groupsFile,
    JSON.stringify(
      {
        groups: visibleGroups,
        lastSync: new Date().toISOString(),
      },
      null,
      2,
    ),
  );
}
