/**
 * Container Runner for NanoClaw
 * Manages persistent Docker containers per group.
 * Communication: docker exec into tmux + file-based IPC (no stdin/stdout piping).
 */
import { exec, execSync } from 'child_process';
import fs from 'fs';
import os from 'os';
import path from 'path';
import { promisify } from 'util';

import { CONTAINER_IMAGE, DATA_DIR, GROUPS_DIR, TIMEZONE } from './config.js';
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

const execAsync = promisify(exec);

const NON_MAIN_GRACE_PERIOD = 2 * 60 * 1000; // 2min grace after query before shutdown
const MAX_QUERY_DURATION = 2 * 60 * 60 * 1000; // 2 hours hard cap
// Adaptive idle polling: fast at first, ramps up over time
const IDLE_POLL_FAST = 500; // First 30s
const IDLE_POLL_MEDIUM = 1500; // 30s–2min
const IDLE_POLL_SLOW = 3000; // 2min+
const IDLE_SNAP_GAP = 1000; // Gap between snap1→snap2 (was 3000)
const TMUX_READY_TIMEOUT = 120000; // 2 minutes to wait for Claude Code to start

export interface ContainerInput {
  prompt: string;
  groupFolder: string;
  chatJid: string;
  isMain: boolean;
  model?: string;
  isScheduledTask?: boolean;
  assistantName?: string;
}

export interface ContainerOutput {
  status: 'success' | 'error';
  result: null; // Responses go via MCP tools → IPC, not stdout
  error?: string;
}

interface VolumeMount {
  hostPath: string;
  containerPath: string;
  readonly: boolean;
}

// ─── Docker exec helper ──────────────────────────────────────────────────────

async function dockerExec(
  containerName: string,
  cmd: string,
  timeoutMs: number = 15000,
): Promise<string> {
  const { stdout } = await execAsync(
    `${CONTAINER_RUNTIME_BIN} exec ${containerName} ${cmd}`,
    { timeout: timeoutMs },
  );
  return stdout;
}

// ─── tmux communication ─────────────────────────────────────────────────────

/**
 * Send text to the Claude Code tmux session via load-buffer + paste-buffer.
 * Uses base64 encoding to avoid shell escaping issues with complex prompts.
 * After pasting, sends C-m (Return) to submit — matching the pmux pattern.
 */
async function sendToTmux(containerName: string, text: string): Promise<void> {
  const b64 = Buffer.from(text).toString('base64');
  await dockerExec(
    containerName,
    `bash -c 'echo ${b64} | base64 -d > /tmp/nc_input.txt'`,
  );
  await dockerExec(containerName, 'tmux load-buffer /tmp/nc_input.txt');
  await dockerExec(containerName, 'tmux paste-buffer -t claude');
  // Small delay before submitting (ink TUI needs time to process paste)
  await new Promise((r) => setTimeout(r, 300));
  await dockerExec(containerName, 'tmux send-keys -t claude C-m');
}

/**
 * Check if Claude Code is idle by examining the tmux pane content.
 * Claude is idle when:
 * - Pane does NOT contain "Esc to interrupt" (which means it's working)
 * - Pane does NOT contain "Enter to confirm" (which means it's showing a prompt)
 * - Two consecutive snapshots are identical (output has stabilized)
 */
function isClaudeIdle(paneContent: string): boolean {
  const trimmed = paneContent.trimEnd();
  // Claude is working if it shows interrupt instructions
  if (trimmed.includes('Esc to interrupt')) return false;
  if (trimmed.includes('ctrl+c to interrupt')) return false;
  // Claude is showing an interactive prompt
  if (trimmed.includes('Enter to confirm')) return false;
  // Must have some content (not just blank)
  if (trimmed.length === 0) return false;
  return true;
}

/**
 * Wait for Claude Code to become idle (two consecutive snapshots match and show idle state).
 * Uses the proven two-snapshot approach from Codeman/recon projects.
 */
async function waitForIdle(
  containerName: string,
  timeoutMs: number,
): Promise<void> {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    try {
      const snap1 = await dockerExec(
        containerName,
        'tmux capture-pane -t claude -p',
        10000,
      );
      await new Promise((r) => setTimeout(r, IDLE_SNAP_GAP));
      const snap2 = await dockerExec(
        containerName,
        'tmux capture-pane -t claude -p',
        10000,
      );

      if (snap1 === snap2 && isClaudeIdle(snap2)) {
        return;
      }
    } catch {
      // Docker exec might fail transiently, keep trying
    }
    // Adaptive backoff: fast polling early, slower over time
    const elapsed = Date.now() - start;
    const delay =
      elapsed < 30000
        ? IDLE_POLL_FAST
        : elapsed < 120000
          ? IDLE_POLL_MEDIUM
          : IDLE_POLL_SLOW;
    await new Promise((r) => setTimeout(r, delay));
  }
  throw new Error(`Timeout waiting for Claude Code idle after ${timeoutMs}ms`);
}

const IDLE_CHECK_DELAY = 120000; // Don't check for idle until 2min after sending prompt

/**
 * Write per-query context file for the MCP server to read.
 */
async function writeQueryContext(
  containerName: string,
  input: ContainerInput,
): Promise<void> {
  const context = JSON.stringify({
    chatJid: input.chatJid,
    groupFolder: input.groupFolder,
    isMain: input.isMain,
  });
  const b64 = Buffer.from(context).toString('base64');
  await dockerExec(
    containerName,
    `bash -c 'echo ${b64} | base64 -d > /workspace/ipc/context.json'`,
  );
}

// ─── Volume mounts ───────────────────────────────────────────────────────────

function buildVolumeMounts(
  group: RegisteredGroup,
  isMain: boolean,
): VolumeMount[] {
  const mounts: VolumeMount[] = [];
  const projectRoot = process.cwd();
  const groupDir = resolveGroupFolderPath(group.folder);

  if (isMain) {
    // Main gets the project root read-write so the trusted main agent can
    // self-modify (edit source, build, deploy).
    mounts.push({
      hostPath: projectRoot,
      containerPath: '/workspace/project',
      readonly: false,
    });

    // Shadow .env so the agent cannot read secrets from the mounted project root.
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
  const groupSessionsDir = path.join(
    DATA_DIR,
    'sessions',
    group.folder,
    '.claude',
  );
  fs.mkdirSync(groupSessionsDir, { recursive: true });

  // Write or update settings.json with MCP server config
  const settingsFile = path.join(groupSessionsDir, 'settings.json');
  const existingSettings = fs.existsSync(settingsFile)
    ? JSON.parse(fs.readFileSync(settingsFile, 'utf-8'))
    : {};

  const settings = {
    ...existingSettings,
    env: {
      ...(existingSettings.env || {}),
      CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS: '1',
      CLAUDE_CODE_ADDITIONAL_DIRECTORIES_CLAUDE_MD: '1',
      CLAUDE_CODE_DISABLE_AUTO_MEMORY: '1',
    },
    // Skip the bypass-permissions confirmation dialog
    skipDangerousModePermissionPrompt: true,
  };

  fs.writeFileSync(settingsFile, JSON.stringify(settings, null, 2) + '\n');

  // Write ~/.claude.json to skip ALL first-run onboarding prompts
  // (theme picker, trust folder, API key confirmation, bypass permissions)
  const onboardingFile = path.join(
    DATA_DIR,
    'sessions',
    group.folder,
    '.claude.json',
  );
  const existingOnboarding = fs.existsSync(onboardingFile)
    ? (() => {
        try {
          return JSON.parse(fs.readFileSync(onboardingFile, 'utf-8'));
        } catch {
          return {};
        }
      })()
    : {};

  // Extract API key suffix for pre-approval
  const envVars = readEnvFile(['ANTHROPIC_API_KEY']);
  const apiKey = envVars.ANTHROPIC_API_KEY || '';
  const apiKeySuffix = apiKey ? apiKey.slice(-20) : '';

  const onboardingData = {
    ...existingOnboarding,
    hasCompletedOnboarding: true,
    hasAcknowledgedCostThreshold: true,
    // Pre-approve the API key so Claude Code doesn't prompt
    customApiKeyResponses: {
      approved: apiKeySuffix ? [apiKeySuffix] : [],
      rejected: [],
    },
    // MCP server config — Claude Code reads mcpServers from .claude.json (user scope)
    mcpServers: {
      ...(existingOnboarding.mcpServers || {}),
      nanoclaw: {
        type: 'stdio',
        command: 'node',
        args: ['/app/dist/ipc-mcp-stdio.js'],
        env: {},
      },
    },
    // Pre-accept workspace trust for known directories
    projects: {
      ...(existingOnboarding.projects || {}),
      '/workspace/group': {
        ...(existingOnboarding.projects?.['/workspace/group'] || {}),
        allowedTools: [],
        hasTrustDialogAccepted: true,
      },
    },
  };
  fs.writeFileSync(
    onboardingFile,
    JSON.stringify(onboardingData, null, 2) + '\n',
  );
  mounts.push({
    hostPath: onboardingFile,
    containerPath: '/home/node/.claude.json',
    readonly: false,
  });

  // Symlink OAuth credentials so Claude Code CLI can authenticate.
  // We symlink rather than copy because OAuth tokens get refreshed —
  // a copy would go stale and invalidate the host's token.
  // The symlink target (/workspace/credentials/.credentials.json) is
  // where we mount the shared credentials file.
  const credSymlink = path.join(groupSessionsDir, '.credentials.json');
  try {
    // Remove existing file/symlink before creating
    if (fs.existsSync(credSymlink) || fs.lstatSync(credSymlink).isSymbolicLink()) {
      fs.unlinkSync(credSymlink);
    }
  } catch {
    // lstatSync throws if path doesn't exist at all — that's fine
  }
  const hostCredentials = path.join(
    os.homedir(),
    '.claude',
    '.credentials.json',
  );
  if (fs.existsSync(hostCredentials)) {
    // Mount the host credentials file into a dedicated path in the container
    mounts.push({
      hostPath: hostCredentials,
      containerPath: '/workspace/credentials/.credentials.json',
      readonly: false,
    });
    // Symlink from where Claude Code expects it to the mounted file
    fs.symlinkSync(
      '/workspace/credentials/.credentials.json',
      credSymlink,
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

  // Per-group IPC namespace
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

  // Logs directory (read-only)
  const logsHostPath = path.join(GROUPS_DIR, group.folder, 'logs');
  fs.mkdirSync(logsHostPath, { recursive: true });
  mounts.push({
    hostPath: logsHostPath,
    containerPath: '/workspace/logs',
    readonly: true,
  });

  // Environment file directory
  const envDir = path.join(DATA_DIR, 'env');
  fs.mkdirSync(envDir, { recursive: true });
  const envFile = path.join(projectRoot, '.env');
  if (fs.existsSync(envFile)) {
    const envContent = fs.readFileSync(envFile, 'utf-8');
    const allowedVars = [
      'ANTHROPIC_API_KEY',
      'OPENAI_API_KEY',
      'ELEVENLABS_API_KEY',
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

  // Embeddings DB for semantic memory (read-only, shared across all groups)
  const embeddingsDbPath = path.join(DATA_DIR, 'embeddings.db');
  if (fs.existsSync(embeddingsDbPath)) {
    mounts.push({
      hostPath: embeddingsDbPath,
      containerPath: '/workspace/embeddings.db',
      readonly: true,
    });
  }

  // Mount code-review plugin (read-only)
  const codeReviewPlugin = path.join(
    os.homedir(),
    '.claude',
    'plugins',
    'marketplaces',
    'claude-plugins-official',
    'plugins',
    'code-review',
  );
  if (fs.existsSync(codeReviewPlugin)) {
    mounts.push({
      hostPath: codeReviewPlugin,
      containerPath: '/app/plugins/code-review',
      readonly: true,
    });
  }

  // Additional mounts validated against external allowlist
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

function buildContainerArgs(
  mounts: VolumeMount[],
  containerName: string,
): string[] {
  // Use -dt (detached + TTY) instead of -i (interactive stdin)
  const args: string[] = ['run', '-dt', '--rm', '--name', containerName];

  // Pass host timezone
  args.push('-e', `TZ=${TIMEZONE}`);

  // Run as host user so bind-mounted files are accessible.
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
  containerName: string;
  group: RegisteredGroup;
  isMain: boolean;
  lastUsed: number;
  shutdownTimer: NodeJS.Timeout | null;
  busy: boolean;
  exited: boolean;
  confirmedIdle: boolean;
  queryQueue: QueuedQuery[];
}

class ContainerPool {
  private pool = new Map<string, PoolEntry>();

  /**
   * Get or spawn a persistent container for a group.
   */
  async getOrSpawn(
    group: RegisteredGroup,
    isMain: boolean,
  ): Promise<PoolEntry> {
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
      'Spawning persistent container (tmux)',
    );

    // Spawn container in detached mode with TTY
    execSync(`${CONTAINER_RUNTIME_BIN} ${containerArgs.join(' ')}`, {
      timeout: 30000,
    });

    const entry: PoolEntry = {
      containerName,
      group,
      isMain,
      lastUsed: Date.now(),
      shutdownTimer: null,
      busy: false,
      exited: false,
      confirmedIdle: false,
      queryQueue: [],
    };

    this.pool.set(key, entry);

    // Wait for Claude Code to be ready (shows `>` prompt)
    try {
      await waitForIdle(containerName, TMUX_READY_TIMEOUT);
      entry.confirmedIdle = true;
      logger.info(
        { group: group.name, containerName },
        'Claude Code ready in container',
      );
    } catch (err) {
      logger.error(
        { group: group.name, containerName, error: err },
        'Claude Code failed to start in container',
      );
      entry.exited = true;
      this.pool.delete(key);
      // Try to stop the container
      exec(stopContainer(containerName));
      throw err;
    }

    return entry;
  }

  /**
   * Send a query to a persistent container and wait for completion.
   * Response is delivered via MCP tools → IPC files, not stdout.
   */
  async sendQuery(
    entry: PoolEntry,
    input: ContainerInput,
  ): Promise<ContainerOutput> {
    if (entry.exited) {
      return {
        status: 'error',
        result: null,
        error: 'Container already exited',
      };
    }

    // If a query is already in flight, queue this one
    if (entry.busy) {
      return new Promise((resolve) => {
        logger.debug(
          { group: entry.group.name, containerName: entry.containerName },
          'Query already in flight, queueing',
        );
        entry.queryQueue.push({ input, resolve });
      });
    }

    return this.dispatchQuery(entry, input);
  }

  private async dispatchQuery(
    entry: PoolEntry,
    input: ContainerInput,
  ): Promise<ContainerOutput> {
    entry.busy = true;
    entry.lastUsed = Date.now();

    // Cancel any pending shutdown — new work arrived
    if (entry.shutdownTimer) {
      clearTimeout(entry.shutdownTimer);
      entry.shutdownTimer = null;
    }

    try {
      // Update per-query context for MCP server
      await writeQueryContext(entry.containerName, input);

      // Skip pre-query idle check if we already confirmed idle after last query
      if (entry.confirmedIdle) {
        logger.info(
          { group: entry.group.name, containerName: entry.containerName },
          'Skipping pre-query idle check (confirmed idle)',
        );
      } else {
        await waitForIdle(entry.containerName, 30000);
      }
      entry.confirmedIdle = false;

      // Paste the prompt into tmux
      await sendToTmux(entry.containerName, input.prompt);

      // Wait before checking for idle — gives Claude Code time to start
      // processing. Without this, the idle detector can see the pre-processing
      // state and immediately declare the query complete.
      await new Promise((r) => setTimeout(r, IDLE_CHECK_DELAY));

      // Wait for processing to complete
      await waitForIdle(entry.containerName, MAX_QUERY_DURATION);
      entry.confirmedIdle = true;

      // Response delivered via MCP tools → IPC files (not stdout)
      return { status: 'success', result: null };
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : String(err);
      logger.error(
        {
          group: entry.group.name,
          containerName: entry.containerName,
          error: errorMsg,
        },
        'Query error',
      );

      // Check if container is still alive
      try {
        await dockerExec(
          entry.containerName,
          'tmux has-session -t claude',
          5000,
        );
      } catch {
        entry.exited = true;
        this.pool.delete(entry.group.folder);
      }

      return { status: 'error', result: null, error: errorMsg };
    } finally {
      entry.busy = false;

      // Schedule shutdown for non-main containers after query completes
      if (!entry.isMain && !entry.exited) {
        this.scheduleShutdown(entry);
      }

      // Drain next queued query
      if (entry.queryQueue.length > 0 && !entry.exited) {
        const next = entry.queryQueue.shift()!;
        logger.debug(
          { group: entry.group.name, remaining: entry.queryQueue.length },
          'Dispatching queued query',
        );
        this.dispatchQuery(entry, next.input).then(next.resolve);
      }
    }
  }

  /**
   * Schedule a shutdown check for a non-main container.
   * After a grace period, verify the container is truly idle via tmux before killing.
   */
  private scheduleShutdown(entry: PoolEntry): void {
    if (entry.shutdownTimer) {
      clearTimeout(entry.shutdownTimer);
    }
    entry.shutdownTimer = setTimeout(async () => {
      entry.shutdownTimer = null;
      if (entry.exited || entry.busy) return;

      // Real tmux idle check before shutting down
      try {
        const snap1 = await dockerExec(
          entry.containerName,
          'tmux capture-pane -t claude -p',
          10000,
        );
        await new Promise((r) => setTimeout(r, IDLE_SNAP_GAP));
        const snap2 = await dockerExec(
          entry.containerName,
          'tmux capture-pane -t claude -p',
          10000,
        );
        if (snap1 !== snap2 || !isClaudeIdle(snap2)) {
          logger.debug(
            { group: entry.group.name, containerName: entry.containerName },
            'Container still active in tmux, rescheduling shutdown',
          );
          this.scheduleShutdown(entry);
          return;
        }
      } catch {
        // If we can't check, assume it's dead
      }

      this.shutdownEntry(entry.group.folder);
    }, NON_MAIN_GRACE_PERIOD);
  }

  /**
   * Gracefully shut down a single container.
   */
  private async shutdownEntry(
    key: string,
    force: boolean = false,
  ): Promise<void> {
    const entry = this.pool.get(key);
    if (!entry || entry.exited) {
      this.pool.delete(key);
      return;
    }

    // Don't shut down if a query is in flight
    if (entry.busy && !force) {
      logger.debug(
        { group: entry.group.name, containerName: entry.containerName },
        'Skipping shutdown, query still in flight',
      );
      return;
    }

    logger.info(
      { group: entry.group.name, containerName: entry.containerName },
      'Shutting down container',
    );

    if (entry.shutdownTimer) {
      clearTimeout(entry.shutdownTimer);
      entry.shutdownTimer = null;
    }

    // Remove from pool immediately
    this.pool.delete(key);
    entry.exited = true;

    // Reject any queued queries
    for (const q of entry.queryQueue) {
      q.resolve({
        status: 'error',
        result: null,
        error: 'Container shutting down',
      });
    }
    entry.queryQueue = [];

    // Send /exit to Claude Code via tmux
    try {
      await dockerExec(
        entry.containerName,
        'tmux send-keys -t claude "/exit" Enter',
        5000,
      );
    } catch {
      // tmux might already be gone
    }

    // Fallback: docker stop after 5s
    setTimeout(() => {
      exec(stopContainer(entry.containerName), { timeout: 10000 }, (err) => {
        if (err) {
          logger.warn(
            { containerName: entry.containerName, error: err.message },
            'Container stop failed (may already be stopped)',
          );
        }
      });
    }, 5000);
  }

  /**
   * Shut down all persistent containers.
   */
  async shutdownAll(): Promise<void> {
    const entries = Array.from(this.pool.keys());
    for (const key of entries) {
      await this.shutdownEntry(key);
    }
  }

  /**
   * Get info about a group's container (if active).
   */
  getEntry(groupFolder: string): { containerName: string } | null {
    const entry = this.pool.get(groupFolder);
    if (entry && !entry.exited) {
      return { containerName: entry.containerName };
    }
    return null;
  }

  /**
   * Restart a specific container by shutting it down.
   * The next query will spawn a fresh container.
   */
  restartContainer(groupFolder: string): boolean {
    const entry = this.pool.get(groupFolder);
    if (entry && !entry.exited) {
      logger.info(
        { group: entry.group.name, containerName: entry.containerName },
        'Restarting container (requested via IPC) - forcing shutdown',
      );
      this.shutdownEntry(groupFolder, true);
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
  onContainerReady: (containerName: string) => void,
): Promise<ContainerOutput> {
  const startTime = Date.now();

  const logsDir = path.join(GROUPS_DIR, group.folder, 'logs');
  fs.mkdirSync(logsDir, { recursive: true });

  const entry = await containerPool.getOrSpawn(group, input.isMain);

  // Notify caller of container name (for queue tracking)
  onContainerReady(entry.containerName);

  logger.info(
    {
      group: group.name,
      containerName: entry.containerName,
      isMain: input.isMain,
    },
    'Sending query to container (tmux)',
  );

  const output = await containerPool.sendQuery(entry, input);

  const duration = Date.now() - startTime;

  // Write log file
  const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
  const logFile = path.join(logsDir, `container-${timestamp}.log`);

  const logLines = [
    `=== Container Query Log ===`,
    `Timestamp: ${new Date().toISOString()}`,
    `Group: ${group.name}`,
    `Container: ${entry.containerName}`,
    `IsMain: ${input.isMain}`,
    `Duration: ${duration}ms`,
    `Status: ${output.status}`,
    ``,
    `=== Input Summary ===`,
    `Prompt length: ${input.prompt.length} chars`,
    ``,
  ];

  if (output.status === 'error') {
    logLines.push(`=== Error ===`, output.error || 'Unknown error');
  }

  fs.writeFileSync(logFile, logLines.join('\n'));
  logger.debug({ logFile }, 'Container log written');

  logger.info(
    {
      group: group.name,
      duration,
      status: output.status,
    },
    'Container query completed',
  );

  return output;
}

/**
 * Run a headless query using `claude -p` inside an existing container.
 * Used for scheduled tasks.
 */
export async function runHeadlessQuery(
  group: RegisteredGroup,
  input: ContainerInput,
  onContainerReady: (containerName: string) => void,
): Promise<ContainerOutput> {
  const entry = await containerPool.getOrSpawn(group, input.isMain);
  onContainerReady(entry.containerName);

  try {
    // Write context file for MCP server
    await writeQueryContext(entry.containerName, input);

    // Run headless claude -p inside the existing container
    // Escape single quotes in prompt for bash
    const escapedPrompt = input.prompt.replace(/'/g, "'\\''");
    const b64 = Buffer.from(input.prompt).toString('base64');
    await dockerExec(
      entry.containerName,
      `bash -c 'echo ${b64} | base64 -d | claude -p --dangerously-skip-permissions'`,
      MAX_QUERY_DURATION,
    );

    return { status: 'success', result: null };
  } catch (err) {
    return {
      status: 'error',
      result: null,
      error: err instanceof Error ? err.message : String(err),
    };
  }
}

/**
 * Pre-spawn the main container so it's warm for the first query.
 */
export async function warmUpMain(group: RegisteredGroup): Promise<void> {
  logger.info({ group: group.name }, 'Pre-spawning main container');
  await containerPool.getOrSpawn(group, true);
}

/**
 * Shut down all persistent containers. Called during process shutdown.
 */
export async function shutdownPool(): Promise<void> {
  await containerPool.shutdownAll();
}

/**
 * Restart a specific container to pick up new code.
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
  const groupIpcDir = resolveGroupIpcPath(groupFolder);
  fs.mkdirSync(groupIpcDir, { recursive: true });

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

export function writeGroupsSnapshot(
  groupFolder: string,
  isMain: boolean,
  groups: AvailableGroup[],
  registeredJids: Set<string>,
): void {
  const groupIpcDir = resolveGroupIpcPath(groupFolder);
  fs.mkdirSync(groupIpcDir, { recursive: true });

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
