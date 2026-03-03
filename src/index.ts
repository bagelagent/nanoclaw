import { exec, execSync } from 'child_process';
import fs from 'fs';
import path from 'path';

import makeWASocket, {
  DisconnectReason,
  WASocket,
  makeCacheableSignalKeyStore,
  useMultiFileAuthState,
} from '@whiskeysockets/baileys';
import { CronExpressionParser } from 'cron-parser';

import { handleGitHubIpc } from './github-handler.js';
import { initGitHubClient } from './github-api.js';
import { startWebhookServer, sweepClosedIssueGroups } from './webhook-server.js';

import {
  ASSISTANT_NAME,
  DATA_DIR,
  DISCORD_ENABLED,
  GROUPS_DIR,
  IDLE_TIMEOUT,
  IPC_POLL_INTERVAL,
  MEMORY_ENABLED,
  POLL_INTERVAL,
  STORE_DIR,
  TIMEZONE,
  TRIGGER_PATTERN,
  WHATSAPP_ENABLED,
} from './config.js';
import './channels/index.js';
import {
  getChannelFactory,
  getRegisteredChannelNames,
} from './channels/registry.js';
import {
  ContainerOutput,
  runContainerAgent,
  shutdownPool,
  writeGroupsSnapshot,
  writeTasksSnapshot,
} from './container-runner.js';
import {
  cleanupOrphans,
  ensureContainerRuntimeRunning,
} from './container-runtime.js';
import {
  closeDatabase,
  createTask,
  deleteTask,
  getAllChats,
  getAllRegisteredGroups,
  getAllSessions,
  getAllTasks,
  getLastGroupSync,
  getMessagesSince,
  getNewMessages,
  getChatMetadata,
  getRouterState,
  getTaskById,
  initDatabase,
  setLastGroupSync,
  setRegisteredGroup,
  setRouterState,
  setSession,
  storeChatMetadata,
  storeMessage,
  storeWhatsAppMessage,
  updateChatName,
  updateTask,
} from './db.js';
import { GroupQueue } from './group-queue.js';
import { resolveGroupFolderPath } from './group-folder.js';
import { closeEmbeddingsDb, searchMemory, startMemoryIndexer } from './memory-indexer.js';
import { findChannel, formatMessages, formatOutbound } from './router.js';
import { startSchedulerLoop } from './task-scheduler.js';
import { connectDiscord, sendDiscordMessage, setDiscordTyping } from './discord.js';
import { Channel, NewMessage, RegisteredGroup } from './types.js';
import { logger } from './logger.js';
import { initOpenAI, isAudioMessage, transcribeAudio } from './audio.js';
import { initGemini, generateImageGemini, isGeminiEnabled } from './image-gen.js';

// Re-export for backwards compatibility during refactor
export { escapeXml, formatMessages } from './router.js';

const GROUP_SYNC_INTERVAL_MS = 24 * 60 * 60 * 1000; // 24 hours

let sock: WASocket;
let lastTimestamp = '';
let sessions: Record<string, string> = {};
let registeredGroups: Record<string, RegisteredGroup> = {};
let lastAgentTimestamp: Record<string, string> = {};
// LID to phone number mapping (WhatsApp now sends LID JIDs for self-chats)
let lidToPhoneMap: Record<string, string> = {};
// Guards to prevent duplicate loops on WhatsApp reconnect
let messageLoopRunning = false;
let ipcWatcherRunning = false;
let groupSyncTimerStarted = false;

const channels: Channel[] = [];
const queue = new GroupQueue();

/**
 * Translate a JID from LID format to phone format if we have a mapping.
 * Returns the original JID if no mapping exists.
 */
function translateJid(jid: string): string {
  if (!jid.endsWith('@lid')) return jid;
  const lidUser = jid.split('@')[0].split(':')[0];
  const phoneJid = lidToPhoneMap[lidUser];
  if (phoneJid) {
    logger.debug({ lidJid: jid, phoneJid }, 'Translated LID to phone JID');
    return phoneJid;
  }
  return jid;
}

async function setTyping(jid: string, isTyping: boolean): Promise<void> {
  if (jid.startsWith('discord:')) {
    await setDiscordTyping(jid, isTyping);
    return;
  }
  try {
    await sock.sendPresenceUpdate(isTyping ? 'composing' : 'paused', jid);
  } catch (err) {
    logger.debug({ jid, err }, 'Failed to update typing status');
  }
}

function loadState(): void {
  // Load from SQLite (migration from JSON happens in initDatabase)
  lastTimestamp = getRouterState('last_timestamp') || '';
  const agentTs = getRouterState('last_agent_timestamp');
  try {
    lastAgentTimestamp = agentTs ? JSON.parse(agentTs) : {};
  } catch {
    logger.warn('Corrupted last_agent_timestamp in DB, resetting');
    lastAgentTimestamp = {};
  }
  sessions = getAllSessions();
  registeredGroups = getAllRegisteredGroups();
  logger.info(
    { groupCount: Object.keys(registeredGroups).length },
    'State loaded',
  );
}

function saveState(): void {
  setRouterState('last_timestamp', lastTimestamp);
  setRouterState(
    'last_agent_timestamp',
    JSON.stringify(lastAgentTimestamp),
  );
}

function registerGroup(jid: string, group: RegisteredGroup): void {
  let groupDir: string;
  try {
    groupDir = resolveGroupFolderPath(group.folder);
  } catch (err) {
    logger.warn(
      { jid, folder: group.folder, err },
      'Rejecting group registration with invalid folder',
    );
    return;
  }

  registeredGroups[jid] = group;
  setRegisteredGroup(jid, group);

  // Create group folder
  fs.mkdirSync(path.join(groupDir, 'logs'), { recursive: true });

  logger.info(
    { jid, name: group.name, folder: group.folder },
    'Group registered',
  );
}

/**
 * Sync group metadata from WhatsApp.
 * Fetches all participating groups and stores their names in the database.
 * Called on startup, daily, and on-demand via IPC.
 */
async function syncGroupMetadata(force = false): Promise<void> {
  // Check if we need to sync (skip if synced recently, unless forced)
  if (!force) {
    const lastSync = getLastGroupSync();
    if (lastSync) {
      const lastSyncTime = new Date(lastSync).getTime();
      const now = Date.now();
      if (now - lastSyncTime < GROUP_SYNC_INTERVAL_MS) {
        logger.debug({ lastSync }, 'Skipping group sync - synced recently');
        return;
      }
    }
  }

  try {
    logger.info('Syncing group metadata from WhatsApp...');
    const groups = await sock.groupFetchAllParticipating();

    let count = 0;
    for (const [jid, metadata] of Object.entries(groups)) {
      if (metadata.subject) {
        updateChatName(jid, metadata.subject);
        count++;
      }
    }

    setLastGroupSync();
    logger.info({ count }, 'Group metadata synced');
  } catch (err) {
    logger.error({ err }, 'Failed to sync group metadata');
  }
}

/**
 * Get available groups list for the agent.
 * Returns groups ordered by most recent activity.
 */
export function getAvailableGroups(): import('./container-runner.js').AvailableGroup[] {
  const chats = getAllChats();
  const registeredJids = new Set(Object.keys(registeredGroups));

  return chats
    .filter(
      (c) =>
        c.jid !== '__group_sync__' &&
        (c.jid.endsWith('@g.us') || c.jid.startsWith('discord:')),
    )
    .map((c) => ({
      jid: c.jid,
      name: c.name,
      lastActivity: c.last_message_time,
      isRegistered: registeredJids.has(c.jid),
    }));
}

/** @internal - exported for testing */
export function _setRegisteredGroups(
  groups: Record<string, RegisteredGroup>,
): void {
  registeredGroups = groups;
}

const ATTACHMENT_RE = /\[Attached image: ([^\]]+)\] (https?:\/\/\S+)/g;

/**
 * Download image attachments from message content into the group workspace.
 * Rewrites content to reference local paths the agent can Read directly.
 */
async function downloadAttachments(
  content: string,
  groupFolder: string,
): Promise<string> {
  const matches = [...content.matchAll(ATTACHMENT_RE)];
  if (matches.length === 0) return content;

  const tmpDir = path.join(GROUPS_DIR, groupFolder, 'tmp');
  fs.mkdirSync(tmpDir, { recursive: true });

  let result = content;
  for (const match of matches) {
    const [fullMatch, filename, url] = match;
    const safeName = `${Date.now()}-${filename.replace(/[^a-zA-Z0-9._-]/g, '_')}`;
    const localPath = path.join(tmpDir, safeName);
    const containerPath = `/workspace/group/tmp/${safeName}`;

    const tmpPath = `${localPath}.tmp`;
    try {
      const response = await fetch(url, { signal: AbortSignal.timeout(15000) });
      if (response.ok) {
        const buffer = Buffer.from(await response.arrayBuffer());
        fs.writeFileSync(tmpPath, buffer);
        fs.renameSync(tmpPath, localPath);
        result = result.replace(
          fullMatch,
          `[Attached image: ${filename}] (saved to ${containerPath} — use the Read tool to view it)`,
        );
      } else {
        logger.warn({ url, status: response.status }, 'Failed to download attachment');
      }
    } catch (err) {
      logger.warn({ url, err }, 'Failed to download attachment');
      try { fs.unlinkSync(tmpPath); } catch {}
    }
  }

  return result;
}

/**
 * Process all pending messages for a group.
 * Called by the GroupQueue when it's this group's turn.
 */
async function processGroupMessages(chatJid: string): Promise<boolean> {
  let group = registeredGroups[chatJid];

  // Auto-register Discord channels on first message
  if (!group && chatJid.startsWith('discord:') && !chatJid.startsWith('discord:dm:')) {
    const chatInfo = getChatMetadata(chatJid);
    if (chatInfo) {
      // Extract channel name from chat info (format: "#channel-name (Server Name)")
      const match = chatInfo.match(/^#([^\s]+)/);
      const channelName = match ? match[1] : chatJid.split(':')[1];
      const folderName = `discord-${channelName}`;

      // Auto-register the channel
      group = {
        name: chatInfo,
        folder: folderName,
        trigger: `@${ASSISTANT_NAME}`,
        requiresTrigger: false,
        added_at: new Date().toISOString(),
      };

      registeredGroups[chatJid] = group;
      setRegisteredGroup(chatJid, group);

      // Create group folder and initial CLAUDE.md
      const groupDir = path.join(DATA_DIR, '..', 'groups', folderName);
      fs.mkdirSync(groupDir, { recursive: true });

      const claudeMdPath = path.join(groupDir, 'CLAUDE.md');
      if (!fs.existsSync(claudeMdPath)) {
        fs.writeFileSync(
          claudeMdPath,
          `# ${chatInfo}\n\nThis is a Discord channel workspace with persistent memory.\n`,
        );
      }

      logger.info(
        { jid: chatJid, name: group.name, folder: group.folder },
        'Auto-registered Discord channel',
      );
    } else {
      return true;
    }
  }

  if (!group) return true;

  const channel = findChannel(channels, chatJid);
  if (!channel) {
    logger.warn({ chatJid }, 'No channel owns JID, skipping messages');
    return true;
  }

  const isMainGroup = group.isMain === true;

  const sinceTimestamp = lastAgentTimestamp[chatJid] || '';
  const missedMessages = getMessagesSince(
    chatJid,
    sinceTimestamp,
    ASSISTANT_NAME,
  );

  if (missedMessages.length === 0) return true;

  // For non-main groups, check if trigger is required and present
  if (!isMainGroup && group.requiresTrigger !== false) {
    const hasTrigger = missedMessages.some((m) =>
      TRIGGER_PATTERN.test(m.content.trim()),
    );
    if (!hasTrigger) return true;
  }

  // Download any image attachments into the group workspace
  for (const m of missedMessages) {
    if (ATTACHMENT_RE.test(m.content)) {
      ATTACHMENT_RE.lastIndex = 0;
      m.content = await downloadAttachments(m.content, group.folder);
    }
  }

  let prompt = formatMessages(missedMessages);

  // Automatic memory context injection
  if (MEMORY_ENABLED) {
    try {
      const searchText = missedMessages
        .map((m) => m.content)
        .join(' ')
        .slice(0, 500);
      const memoryResults = await searchMemory(searchText, 'hybrid', 5);
      if (memoryResults.length > 0) {
        const context = memoryResults
          .map((r) => `[${r.source}] ${r.content.slice(0, 300)}`)
          .join('\n\n');
        prompt = `<memory-context>\nRelevant context from your memory:\n\n${context}\n</memory-context>\n\n${prompt}`;
      }
    } catch (err) {
      logger.error({ err }, 'Memory context injection failed');
    }
  }

  // Advance cursor so the piping path in startMessageLoop won't re-fetch
  // these messages. Save the old cursor so we can roll back on error.
  const previousCursor = lastAgentTimestamp[chatJid] || '';
  lastAgentTimestamp[chatJid] =
    missedMessages[missedMessages.length - 1].timestamp;
  saveState();

  logger.info(
    { group: group.name, messageCount: missedMessages.length },
    'Processing messages',
  );

  // Track idle timer for closing stdin when agent is idle
  let idleTimer: ReturnType<typeof setTimeout> | null = null;

  const resetIdleTimer = () => {
    if (idleTimer) clearTimeout(idleTimer);
    idleTimer = setTimeout(() => {
      logger.debug(
        { group: group.name },
        'Idle timeout, closing container stdin',
      );
      queue.closeStdin(chatJid);
    }, IDLE_TIMEOUT);
  };

  await channel.setTyping?.(chatJid, true);
  let hadError = false;
  let outputSentToUser = false;

  const output = await runAgent(group, prompt, chatJid, async (result) => {
    // Streaming output callback — called for each agent result
    if (result.result) {
      const raw =
        typeof result.result === 'string'
          ? result.result
          : JSON.stringify(result.result);
      // Strip <internal>...</internal> blocks — agent uses these for internal reasoning
      const text = raw.replace(/<internal>[\s\S]*?<\/internal>/g, '').trim();
      logger.info({ group: group.name }, `Agent output: ${raw.slice(0, 200)}`);
      if (text) {
        await channel.sendMessage(chatJid, text);
        outputSentToUser = true;
      }
      // Only reset idle timer on actual results, not session-update markers (result: null)
      resetIdleTimer();
    }

    if (result.status === 'success') {
      queue.notifyIdle(chatJid);
    }

    if (result.status === 'error') {
      hadError = true;
    }
  });

  await channel.setTyping?.(chatJid, false);
  if (idleTimer) clearTimeout(idleTimer);

  if (output === 'error' || hadError) {
    // If we already sent output to the user, don't roll back the cursor —
    // the user got their response and re-processing would send duplicates.
    if (outputSentToUser) {
      logger.warn(
        { group: group.name },
        'Agent error after output was sent, skipping cursor rollback to prevent duplicates',
      );
      return true;
    }
    // Roll back cursor so retries can re-process these messages
    lastAgentTimestamp[chatJid] = previousCursor;
    saveState();
    logger.warn(
      { group: group.name },
      'Agent error, rolled back message cursor for retry',
    );
    return false;
  }

  return true;
}

async function runAgent(
  group: RegisteredGroup,
  prompt: string,
  chatJid: string,
  onOutput?: (output: ContainerOutput) => Promise<void>,
): Promise<'success' | 'error'> {
  const isMain = group.isMain === true;
  const sessionId = sessions[group.folder];

  // Update tasks snapshot for container to read (filtered by group)
  const tasks = getAllTasks();
  writeTasksSnapshot(
    group.folder,
    isMain,
    tasks.map((t) => ({
      id: t.id,
      groupFolder: t.group_folder,
      prompt: t.prompt,
      schedule_type: t.schedule_type,
      schedule_value: t.schedule_value,
      status: t.status,
      next_run: t.next_run,
    })),
  );

  // Update available groups snapshot (main group only can see all groups)
  const availableGroups = getAvailableGroups();
  writeGroupsSnapshot(
    group.folder,
    isMain,
    availableGroups,
    new Set(Object.keys(registeredGroups)),
  );

  // Wrap onOutput to track session ID from streamed results
  const wrappedOnOutput = onOutput
    ? async (output: ContainerOutput) => {
        if (output.newSessionId) {
          sessions[group.folder] = output.newSessionId;
          setSession(group.folder, output.newSessionId);
        }
        await onOutput(output);
      }
    : undefined;

  try {
    const output = await runContainerAgent(
      group,
      {
        prompt,
        sessionId,
        groupFolder: group.folder,
        chatJid,
        isMain,
        assistantName: ASSISTANT_NAME,
      },
      (proc, containerName) =>
        queue.registerProcess(chatJid, proc, containerName, group.folder),
      wrappedOnOutput,
    );

    if (output.newSessionId) {
      sessions[group.folder] = output.newSessionId;
      setSession(group.folder, output.newSessionId);
    }

    if (output.status === 'error') {
      logger.error(
        { group: group.name, error: output.error },
        'Container agent error',
      );
      return 'error';
    }

    return 'success';
  } catch (err) {
    logger.error({ group: group.name, err }, 'Agent error');
    return 'error';
  }
}

async function sendMessage(jid: string, text: string): Promise<void> {
  // GitHub groups don't have a real chat JID - skip sending
  if (jid.startsWith('github-')) {
    logger.debug({ jid }, 'Skipping sendMessage for GitHub group (no chat)');
    return;
  }

  if (jid.startsWith('discord:')) {
    await sendDiscordMessage(jid, text);
  } else {
    try {
      await sock.sendMessage(jid, { text });
      logger.info({ jid, length: text.length }, 'Message sent');
    } catch (err) {
      logger.error({ jid, err }, 'Failed to send message');
    }
  }
}

async function sendVoiceMessage(
  jid: string,
  text: string,
  voice: 'alloy' | 'echo' | 'fable' | 'onyx' | 'nova' | 'shimmer',
): Promise<void> {
  try {
    const { generateSpeech } = await import('./audio.js');
    const audioBuffer = await generateSpeech(text, voice);

    if (!audioBuffer) {
      logger.error('Failed to generate speech, sending text instead');
      await sendMessage(jid, text);
      return;
    }

    if (jid.startsWith('discord:')) {
      // Discord: send as audio attachment
      const { sendDiscordVoiceMessage } = await import('./discord.js');
      await sendDiscordVoiceMessage(jid, audioBuffer);
      logger.info({ jid, voice, length: audioBuffer.length }, 'Discord voice message sent');
    } else {
      // WhatsApp: send as PTT (push-to-talk) audio message
      await sock.sendMessage(jid, {
        audio: audioBuffer,
        mimetype: 'audio/ogg; codecs=opus',
        ptt: true,
      });
      logger.info({ jid, voice, length: audioBuffer.length }, 'WhatsApp voice message sent');
    }
  } catch (err) {
    logger.error({ jid, err }, 'Failed to send voice message, sending text instead');
    await sendMessage(jid, text);
  }
}

function cleanupStaleIpcFiles(): void {
  const ipcBaseDir = path.join(DATA_DIR, 'ipc');
  const oneHourAgo = Date.now() - 60 * 60 * 1000;
  const subDirs = ['errors', 'replies', 'progress'];

  // Clean global error dir
  const globalErrorDir = path.join(ipcBaseDir, 'errors');
  if (fs.existsSync(globalErrorDir)) {
    try {
      for (const file of fs.readdirSync(globalErrorDir)) {
        const filePath = path.join(globalErrorDir, file);
        try {
          const stat = fs.statSync(filePath);
          if (stat.mtimeMs < oneHourAgo) {
            fs.unlinkSync(filePath);
          }
        } catch {}
      }
    } catch {}
  }

  // Clean per-group IPC subdirectories
  try {
    const groupFolders = fs.readdirSync(ipcBaseDir).filter((f) => {
      try { return fs.statSync(path.join(ipcBaseDir, f)).isDirectory() && f !== 'errors'; } catch { return false; }
    });

    for (const group of groupFolders) {
      for (const sub of subDirs) {
        const dir = path.join(ipcBaseDir, group, sub);
        if (!fs.existsSync(dir)) continue;
        try {
          for (const file of fs.readdirSync(dir)) {
            const filePath = path.join(dir, file);
            try {
              const stat = fs.statSync(filePath);
              if (stat.mtimeMs < oneHourAgo) {
                fs.unlinkSync(filePath);
              }
            } catch {}
          }
        } catch {}
      }
    }
  } catch {}

  logger.info('Cleaned up stale IPC files');
}

function startIpcWatcher(): void {
  if (ipcWatcherRunning) {
    logger.debug('IPC watcher already running, skipping duplicate start');
    return;
  }
  ipcWatcherRunning = true;

  const ipcBaseDir = path.join(DATA_DIR, 'ipc');
  fs.mkdirSync(ipcBaseDir, { recursive: true });

  // Clean up stale IPC files from previous runs
  cleanupStaleIpcFiles();

  const processIpcFiles = async () => {
    // Scan all group IPC directories (identity determined by directory)
    let groupFolders: string[];
    try {
      groupFolders = fs.readdirSync(ipcBaseDir).filter((f) => {
        const stat = fs.statSync(path.join(ipcBaseDir, f));
        return stat.isDirectory() && f !== 'errors';
      });
    } catch (err) {
      logger.error({ err }, 'Error reading IPC base directory');
      setTimeout(processIpcFiles, IPC_POLL_INTERVAL);
      return;
    }

    // Build folder→isMain lookup from registered groups
    const folderIsMain = new Map<string, boolean>();
    for (const group of Object.values(registeredGroups)) {
      if (group.isMain) folderIsMain.set(group.folder, true);
    }

    for (const sourceGroup of groupFolders) {
      const isMain = folderIsMain.get(sourceGroup) === true;
      const messagesDir = path.join(ipcBaseDir, sourceGroup, 'messages');
      const tasksDir = path.join(ipcBaseDir, sourceGroup, 'tasks');

      // Process messages from this group's IPC directory
      try {
        if (fs.existsSync(messagesDir)) {
          const messageFiles = fs
            .readdirSync(messagesDir)
            .filter((f) => f.endsWith('.json'));
          for (const file of messageFiles) {
            const filePath = path.join(messagesDir, file);
            try {
              const data = JSON.parse(fs.readFileSync(filePath, 'utf-8'));
              if (data.type === 'message' && data.chatJid && data.text) {
                // Authorization: verify this group can send to this chatJid
                const targetGroup = registeredGroups[data.chatJid];
                if (
                  isMain ||
                  (targetGroup && targetGroup.folder === sourceGroup)
                ) {
                  // For Discord, don't prefix with assistant name
                  const message = data.chatJid.startsWith('discord:')
                    ? data.text
                    : `${ASSISTANT_NAME}: ${data.text}`;
                  await sendMessage(data.chatJid, message);
                  logger.info(
                    { chatJid: data.chatJid, sourceGroup },
                    'IPC message sent',
                  );
                } else {
                  logger.warn(
                    { chatJid: data.chatJid, sourceGroup },
                    'Unauthorized IPC message attempt blocked',
                  );
                }
              } else if (data.type === 'voice_message' && data.chatJid && data.text) {
                // Handle voice message
                const targetGroup = registeredGroups[data.chatJid];
                if (
                  isMain ||
                  (targetGroup && targetGroup.folder === sourceGroup)
                ) {
                  await sendVoiceMessage(
                    data.chatJid,
                    data.text,
                    data.voice || 'nova',
                  );
                  logger.info(
                    { chatJid: data.chatJid, sourceGroup, voice: data.voice },
                    'IPC voice message sent',
                  );
                } else {
                  logger.warn(
                    { chatJid: data.chatJid, sourceGroup },
                    'Unauthorized IPC voice message attempt blocked',
                  );
                }
              } else if (data.type === 'image' && data.chatJid) {
                // Handle image message
                const targetGroup = registeredGroups[data.chatJid];
                if (
                  isMain ||
                  (targetGroup && targetGroup.folder === sourceGroup)
                ) {
                  // Read image from path or decode base64
                  let imageBuffer: Buffer;
                  let filename: string;

                  if (data.imagePath) {
                    // Translate container paths to host paths
                    let hostPath = data.imagePath;
                    if (hostPath.startsWith('/workspace/project/')) {
                      hostPath = path.join(process.cwd(), hostPath.slice('/workspace/project/'.length));
                    } else if (hostPath.startsWith('/workspace/group/')) {
                      hostPath = path.join(GROUPS_DIR, sourceGroup, hostPath.slice('/workspace/group/'.length));
                    } else if (hostPath.startsWith('/workspace/extra/')) {
                      // Additional mounts — resolve via group's containerConfig
                      const srcGroup = registeredGroups[Object.keys(registeredGroups).find(jid => registeredGroups[jid].folder === sourceGroup) || ''];
                      if (srcGroup?.containerConfig?.additionalMounts) {
                        const rest = hostPath.slice('/workspace/extra/'.length);
                        const mountName = rest.split('/')[0];
                        const subPath = rest.slice(mountName.length + 1);
                        const mount = srcGroup.containerConfig.additionalMounts.find((m: any) => m.containerPath === mountName);
                        if (mount) {
                          hostPath = path.join(mount.hostPath.replace(/^~/, process.env.HOME || ''), subPath);
                        }
                      }
                    }
                    imageBuffer = fs.readFileSync(hostPath);
                    filename = path.basename(hostPath);
                  } else if (data.imageBase64) {
                    imageBuffer = Buffer.from(data.imageBase64, 'base64');
                    filename = data.filename || 'image.png';
                  } else {
                    throw new Error('No image source provided');
                  }

                  // Send to appropriate platform
                  if (data.chatJid.startsWith('discord:')) {
                    const { sendDiscordImage } = await import('./discord.js');
                    await sendDiscordImage(
                      data.chatJid,
                      imageBuffer,
                      filename,
                      data.caption,
                    );
                  } else {
                    // WhatsApp - not yet implemented
                    logger.warn(
                      { chatJid: data.chatJid },
                      'WhatsApp image sending not yet implemented',
                    );
                  }

                  logger.info(
                    { chatJid: data.chatJid, sourceGroup, filename },
                    'IPC image sent',
                  );
                } else {
                  logger.warn(
                    { chatJid: data.chatJid, sourceGroup },
                    'Unauthorized IPC image attempt blocked',
                  );
                }
              }
              fs.unlinkSync(filePath);
            } catch (err) {
              logger.error(
                { file, sourceGroup, err },
                'Error processing IPC message',
              );
              const errorDir = path.join(ipcBaseDir, 'errors');
              fs.mkdirSync(errorDir, { recursive: true });
              fs.renameSync(
                filePath,
                path.join(errorDir, `${sourceGroup}-${file}`),
              );
            }
          }
        }
      } catch (err) {
        logger.error(
          { err, sourceGroup },
          'Error reading IPC messages directory',
        );
      }

      // Process progress updates from this group's IPC directory
      const progressDir = path.join(ipcBaseDir, sourceGroup, 'progress');
      try {
        if (fs.existsSync(progressDir)) {
          const progressFiles = fs
            .readdirSync(progressDir)
            .filter((f) => f.endsWith('.json'));
          for (const file of progressFiles) {
            const filePath = path.join(progressDir, file);
            try {
              const data = JSON.parse(fs.readFileSync(filePath, 'utf-8'));
              if (data.chatJid && data.status) {
                // Send progress update
                await sendMessage(data.chatJid, data.status);
                logger.debug({ chatJid: data.chatJid, status: data.status }, 'Progress update sent');
              }
              fs.unlinkSync(filePath);
            } catch (err) {
              logger.error({ file, sourceGroup, err }, 'Error processing progress update');
              try { fs.unlinkSync(filePath); } catch {}
            }
          }
        }
      } catch (err) {
        logger.error({ err, sourceGroup }, 'Error reading progress directory');
      }

      // Process tasks from this group's IPC directory
      try {
        if (fs.existsSync(tasksDir)) {
          const taskFiles = fs
            .readdirSync(tasksDir)
            .filter((f) => f.endsWith('.json'));
          for (const file of taskFiles) {
            const filePath = path.join(tasksDir, file);
            try {
              const data = JSON.parse(fs.readFileSync(filePath, 'utf-8'));
              // Pass source group identity to processTaskIpc for authorization
              await processTaskIpc(data, sourceGroup, isMain);
              fs.unlinkSync(filePath);
            } catch (err) {
              logger.error(
                { file, sourceGroup, err },
                'Error processing IPC task',
              );
              const errorDir = path.join(ipcBaseDir, 'errors');
              fs.mkdirSync(errorDir, { recursive: true });
              fs.renameSync(
                filePath,
                path.join(errorDir, `${sourceGroup}-${file}`),
              );
            }
          }
        }
      } catch (err) {
        logger.error({ err, sourceGroup }, 'Error reading IPC tasks directory');
      }
    }

    setTimeout(processIpcFiles, IPC_POLL_INTERVAL);
  };

  processIpcFiles();
  logger.info('IPC watcher started (per-group namespaces)');
}

async function sendIpcReply(
  groupFolder: string,
  requestId: string,
  status: 'success' | 'error',
  message?: string,
  error?: string
): Promise<void> {
  const repliesDir = path.join(DATA_DIR, 'ipc', groupFolder, 'replies');
  fs.mkdirSync(repliesDir, { recursive: true });

  const replyPath = path.join(repliesDir, `${requestId}.json`);
  const reply = { status, message, error, timestamp: new Date().toISOString() };

  // Atomic write
  const tempPath = `${replyPath}.tmp`;
  fs.writeFileSync(tempPath, JSON.stringify(reply, null, 2));
  fs.renameSync(tempPath, replyPath);

  logger.debug({ requestId, status }, 'IPC reply sent');

  // Auto-delete after 60s to prevent unbounded growth
  setTimeout(() => {
    try { fs.unlinkSync(replyPath); } catch {}
  }, 60000);
}

async function processTaskIpc(
  data: {
    type: string;
    requestId?: string; // For reply matching
    taskId?: string;
    prompt?: string;
    schedule_type?: string;
    schedule_value?: string;
    context_mode?: string;
    groupFolder?: string;
    chatJid?: string;
    targetJid?: string;
    // For register_group
    jid?: string;
    name?: string;
    folder?: string;
    trigger?: string;
    containerConfig?: RegisteredGroup['containerConfig'];
    // For deploy
    targets?: string[];
    commitMessage?: string;
  },
  sourceGroup: string, // Verified identity from IPC directory
  isMain: boolean, // Verified from directory path
): Promise<void> {
  switch (data.type) {
    case 'test_container_build':
      // Only main group can test builds
      if (!isMain) {
        logger.warn({ sourceGroup }, 'Non-main group attempted to test build');
        return;
      }
      try {
        logger.info('Testing container build...');
        const buildOutput = execSync('./container/build.sh', {
          cwd: process.cwd(),
          encoding: 'utf-8',
          timeout: 600000
        });
        logger.info({ output: buildOutput }, 'Container build test succeeded');

        // Send success message back to agent
        const successMsg = `Container build test SUCCEEDED:\n\n${buildOutput}`;
        const successJid = data.chatJid || Object.keys(registeredGroups).find(jid => registeredGroups[jid].folder === sourceGroup);
        if (successJid) {
          await sendMessage(successJid, successMsg);
        }
      } catch (err) {
        logger.error({ err }, 'Container build test failed');

        // Send failure message with full error details
        let errorMsg = 'Container build test FAILED:\n\n';
        if (err && typeof err === 'object') {
          if ('stdout' in err) errorMsg += `STDOUT:\n${err.stdout}\n\n`;
          if ('stderr' in err) errorMsg += `STDERR:\n${err.stderr}\n\n`;
          if ('status' in err) errorMsg += `Exit code: ${err.status}\n`;
        } else {
          errorMsg += String(err);
        }

        // Find the chatJid for the source group
        const chatJid = data.chatJid || Object.keys(registeredGroups).find(jid => registeredGroups[jid].folder === sourceGroup);
        if (chatJid) {
          await sendMessage(chatJid, errorMsg);
        }
      }
      return;

    case 'restart_container': {
      // Only main group can restart containers
      if (!isMain) {
        logger.warn({ sourceGroup }, 'Non-main group attempted to restart container');
        return;
      }

      const requestId = data.requestId as string | undefined;
      const chatJid = data.chatJid as string | undefined;

      if (data.groupFolder) {
        try {
          const { restartContainer } = await import('./container-runner.js');
          const restarted = restartContainer(data.groupFolder);
          logger.info(
            { groupFolder: data.groupFolder, restarted },
            'Container restart requested via IPC',
          );

          if (restarted) {
            if (chatJid) {
              await sendMessage(chatJid, `${ASSISTANT_NAME}: Container restarted successfully`);
            }
            if (requestId) {
              await sendIpcReply(sourceGroup, requestId, 'success', 'Container restarted');
            }
          } else {
            if (chatJid) {
              await sendMessage(chatJid, `${ASSISTANT_NAME}: Container not found or already stopped`);
            }
            if (requestId) {
              await sendIpcReply(sourceGroup, requestId, 'error', undefined, 'Container not found or already stopped');
            }
          }
        } catch (err: any) {
          logger.error({ err }, 'Container restart failed');
          if (chatJid) {
            await sendMessage(chatJid, `${ASSISTANT_NAME}: Container restart failed: ${err.message}`);
          }
          if (requestId) {
            await sendIpcReply(sourceGroup, requestId, 'error', undefined, err.message);
          }
        }
      }
      return;
    }
    case 'schedule_task':
      if (
        data.prompt &&
        data.schedule_type &&
        data.schedule_value &&
        data.targetJid
      ) {
        // Resolve the target group from JID
        const targetJid = data.targetJid as string;
        const targetGroupEntry = registeredGroups[targetJid];

        if (!targetGroupEntry) {
          logger.warn(
            { targetJid },
            'Cannot schedule task: target group not registered',
          );
          break;
        }

        const targetFolder = targetGroupEntry.folder;

        // Authorization: non-main groups can only schedule for themselves
        if (!isMain && targetFolder !== sourceGroup) {
          logger.warn(
            { sourceGroup, targetFolder },
            'Unauthorized schedule_task attempt blocked',
          );
          break;
        }

        const scheduleType = data.schedule_type as 'cron' | 'interval' | 'once';

        let nextRun: string | null = null;
        if (scheduleType === 'cron') {
          try {
            const interval = CronExpressionParser.parse(data.schedule_value, {
              tz: TIMEZONE,
            });
            nextRun = interval.next().toISOString();
          } catch {
            logger.warn(
              { scheduleValue: data.schedule_value },
              'Invalid cron expression',
            );
            break;
          }
        } else if (scheduleType === 'interval') {
          const ms = parseInt(data.schedule_value, 10);
          if (isNaN(ms) || ms <= 0) {
            logger.warn(
              { scheduleValue: data.schedule_value },
              'Invalid interval',
            );
            break;
          }
          nextRun = new Date(Date.now() + ms).toISOString();
        } else if (scheduleType === 'once') {
          const scheduled = new Date(data.schedule_value);
          if (isNaN(scheduled.getTime())) {
            logger.warn(
              { scheduleValue: data.schedule_value },
              'Invalid timestamp',
            );
            break;
          }
          nextRun = scheduled.toISOString();
        }

        const taskId = `task-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
        const contextMode =
          data.context_mode === 'group' || data.context_mode === 'isolated'
            ? data.context_mode
            : 'isolated';
        createTask({
          id: taskId,
          group_folder: targetFolder,
          chat_jid: targetJid,
          prompt: data.prompt,
          schedule_type: scheduleType,
          schedule_value: data.schedule_value,
          context_mode: contextMode,
          next_run: nextRun,
          status: 'active',
          created_at: new Date().toISOString(),
        });
        logger.info(
          { taskId, sourceGroup, targetFolder, contextMode },
          'Task created via IPC',
        );
      }
      break;

    case 'pause_task':
      if (data.taskId) {
        const task = getTaskById(data.taskId);
        if (task && (isMain || task.group_folder === sourceGroup)) {
          updateTask(data.taskId, { status: 'paused' });
          logger.info(
            { taskId: data.taskId, sourceGroup },
            'Task paused via IPC',
          );
        } else {
          logger.warn(
            { taskId: data.taskId, sourceGroup },
            'Unauthorized task pause attempt',
          );
        }
      }
      break;

    case 'resume_task':
      if (data.taskId) {
        const task = getTaskById(data.taskId);
        if (task && (isMain || task.group_folder === sourceGroup)) {
          updateTask(data.taskId, { status: 'active' });
          logger.info(
            { taskId: data.taskId, sourceGroup },
            'Task resumed via IPC',
          );
        } else {
          logger.warn(
            { taskId: data.taskId, sourceGroup },
            'Unauthorized task resume attempt',
          );
        }
      }
      break;

    case 'cancel_task':
      if (data.taskId) {
        const task = getTaskById(data.taskId);
        if (task && (isMain || task.group_folder === sourceGroup)) {
          deleteTask(data.taskId);
          logger.info(
            { taskId: data.taskId, sourceGroup },
            'Task cancelled via IPC',
          );
        } else {
          logger.warn(
            { taskId: data.taskId, sourceGroup },
            'Unauthorized task cancel attempt',
          );
        }
      }
      break;

    case 'refresh_groups':
      // Only main group can request a refresh
      if (isMain) {
        logger.info(
          { sourceGroup },
          'Group metadata refresh requested via IPC',
        );
        if (WHATSAPP_ENABLED && sock) {
          await syncGroupMetadata(true);
        }
        // Write updated snapshot immediately
        const availableGroups = getAvailableGroups();
        writeGroupsSnapshot(
          sourceGroup,
          true,
          availableGroups,
          new Set(Object.keys(registeredGroups)),
        );
      } else {
        logger.warn(
          { sourceGroup },
          'Unauthorized refresh_groups attempt blocked',
        );
      }
      break;

    case 'register_group':
      // Only main group can register new groups
      if (!isMain) {
        logger.warn(
          { sourceGroup },
          'Unauthorized register_group attempt blocked',
        );
        break;
      }
      if (data.jid && data.name && data.folder && data.trigger) {
        registerGroup(data.jid, {
          name: data.name,
          folder: data.folder,
          trigger: data.trigger,
          added_at: new Date().toISOString(),
          containerConfig: data.containerConfig,
        });
      } else {
        logger.warn(
          { data },
          'Invalid register_group request - missing required fields',
        );
      }
      break;

    case 'deploy': {
      if (!isMain) {
        logger.warn({ sourceGroup }, 'Unauthorized deploy attempt blocked');
        break;
      }

      const targets = (data.targets || []) as string[];
      const commitMsg = data.commitMessage || 'chore: agent-initiated deploy';
      const requestId = data.requestId as string | undefined;
      const chatJid = data.chatJid as string | undefined;

      logger.info({ targets, commitMsg, requestId }, 'Deploy requested via IPC');

      // 1. Git commit all changes (audit trail)
      try {
        execSync('git add -A', { cwd: process.cwd(), stdio: 'pipe' });
        // Check if there's anything to commit
        try {
          execSync('git diff --cached --quiet', { cwd: process.cwd(), stdio: 'pipe' });
          logger.info('No changes to commit, proceeding with build');
        } catch {
          // Non-zero exit means there are staged changes
          execSync(
            `git commit -m ${JSON.stringify(commitMsg + '\n\nDeployed by agent via IPC')}`,
            { cwd: process.cwd(), stdio: 'pipe' },
          );
          logger.info({ commitMsg }, 'Changes committed');
        }
      } catch (err: any) {
        logger.error({ err }, 'Git commit failed, aborting deploy');
        if (chatJid) {
          await sendMessage(chatJid, `${ASSISTANT_NAME}: Deploy failed during git commit:\n\n\`\`\`\n${err.stderr?.toString() || err.message}\n\`\`\``);
        }
        if (requestId) {
          await sendIpcReply(sourceGroup, requestId, 'error', undefined, 'Git commit failed');
        }
        break;
      }

      // 2. Build host if requested
      if (targets.includes('host')) {
        try {
          logger.info('Building host...');
          execSync('npm run build', { cwd: process.cwd(), stdio: 'pipe', timeout: 60000 });
          logger.info('Host build succeeded');
        } catch (err: any) {
          logger.error({ err }, 'Host build failed, aborting deploy');
          if (chatJid) {
            await sendMessage(chatJid, `${ASSISTANT_NAME}: Host build failed:\n\n\`\`\`\n${err.stderr?.toString().slice(0, 1000) || err.message}\n\`\`\``);
          }
          if (requestId) {
            await sendIpcReply(sourceGroup, requestId, 'error', undefined, 'Host build failed');
          }
          break;
        }
      }

      // 3. Build container if requested
      if (targets.includes('container')) {
        try {
          logger.info('Building container...');
          const buildOutput = execSync('./container/build.sh', {
            cwd: process.cwd(),
            stdio: 'pipe',
            timeout: 600000,
            encoding: 'utf-8'
          });
          logger.info({ output: buildOutput }, 'Container build succeeded');
        } catch (err: any) {
          logger.error({ err }, 'Container build failed, aborting deploy');
          if (chatJid) {
            await sendMessage(chatJid, `${ASSISTANT_NAME}: Container build failed:\n\n\`\`\`\n${err.stderr?.toString().slice(0, 1000) || err.message}\n\`\`\``);
          }
          if (requestId) {
            await sendIpcReply(sourceGroup, requestId, 'error', undefined, 'Container build failed');
          }
          break;
        }
      }

      // 4. Send success reply and notification
      if (chatJid) {
        await sendMessage(chatJid, `${ASSISTANT_NAME}: Deploy completed successfully!\n\nTargets: ${targets.join(', ') || 'none (commit only)'}`);
      }
      if (requestId) {
        await sendIpcReply(sourceGroup, requestId, 'success', `Deploy completed. Targets: ${targets.join(', ')}`);
      }

      // 5. Graceful shutdown after delay — systemd will restart with new code
      // Wait long enough for the agent query to complete and send final message
      logger.info('Deploy complete, restarting in 30 seconds...');
      setTimeout(async () => {
        await shutdownPool();
        await queue.shutdown(10000);
        process.exit(0);
      }, 30000);
      break;
    }

    case 'generate_image': {
      const requestId = data.requestId;
      if (!requestId || !(data as any).prompt) {
        logger.warn({ sourceGroup }, 'generate_image missing requestId or prompt');
        return;
      }
      if (!isGeminiEnabled()) {
        const repliesDir = path.join(DATA_DIR, 'ipc', sourceGroup, 'replies');
        fs.mkdirSync(repliesDir, { recursive: true });
        const replyPath = path.join(repliesDir, `${requestId}.json`);
        const replyPayload = { status: 'error', error: 'Image generation disabled (missing GOOGLE_API_KEY)', timestamp: new Date().toISOString() };
        const tempPath = `${replyPath}.tmp`;
        fs.writeFileSync(tempPath, JSON.stringify(replyPayload, null, 2));
        fs.renameSync(tempPath, replyPath);
        setTimeout(() => { try { fs.unlinkSync(replyPath); } catch {} }, 60000);
        return;
      }

      try {
        const imgData = data as any;
        const fullPrompt = (imgData.aspectRatio && imgData.aspectRatio !== '1:1')
          ? `Generate an image with ${imgData.aspectRatio} aspect ratio. ${imgData.prompt}`
          : imgData.prompt;
        const groupDir = path.join(GROUPS_DIR, sourceGroup);
        const result = await generateImageGemini(fullPrompt, imgData.aspectRatio || '1:1', groupDir);

        const repliesDir = path.join(DATA_DIR, 'ipc', sourceGroup, 'replies');
        fs.mkdirSync(repliesDir, { recursive: true });
        const replyPath = path.join(repliesDir, `${requestId}.json`);
        const replyPayload = { status: 'success', data: { containerPath: result.containerPath, filename: result.filename }, timestamp: new Date().toISOString() };
        const tempPath = `${replyPath}.tmp`;
        fs.writeFileSync(tempPath, JSON.stringify(replyPayload, null, 2));
        fs.renameSync(tempPath, replyPath);
        logger.debug({ requestId, status: 'success' }, 'generate_image reply sent');
        setTimeout(() => { try { fs.unlinkSync(replyPath); } catch {} }, 60000);
      } catch (err) {
        logger.error({ err, sourceGroup }, 'generate_image failed');
        const repliesDir = path.join(DATA_DIR, 'ipc', sourceGroup, 'replies');
        fs.mkdirSync(repliesDir, { recursive: true });
        const replyPath = path.join(repliesDir, `${requestId}.json`);
        const replyPayload = { status: 'error', error: err instanceof Error ? err.message : String(err), timestamp: new Date().toISOString() };
        const tempPath = `${replyPath}.tmp`;
        fs.writeFileSync(tempPath, JSON.stringify(replyPayload, null, 2));
        fs.renameSync(tempPath, replyPath);
        setTimeout(() => { try { fs.unlinkSync(replyPath); } catch {} }, 60000);
      }
      return;
    }

    default:
      // Check if it's a GitHub IPC request
      if (data.type.startsWith('github_')) {
        // Allow GitHub operations from main OR GitHub groups
        const isGitHubGroup = sourceGroup.startsWith('github-');
        if (!isMain && !isGitHubGroup) {
          logger.warn({ sourceGroup, type: data.type }, 'Non-GitHub group attempted GitHub operation');
          return;
        }

        try {
          const reply = await handleGitHubIpc(data as any, sourceGroup);

          // Write reply directly with { status, data, error } structure
          // (sendIpcReply writes { status, message, error } which doesn't match
          // what the container MCP tools expect — they read reply.data)
          if (data.requestId) {
            const repliesDir = path.join(DATA_DIR, 'ipc', sourceGroup, 'replies');
            fs.mkdirSync(repliesDir, { recursive: true });
            const replyPath = path.join(repliesDir, `${data.requestId}.json`);
            const replyPayload = { status: reply.status, data: reply.data, error: reply.error, timestamp: new Date().toISOString() };
            const tempPath = `${replyPath}.tmp`;
            fs.writeFileSync(tempPath, JSON.stringify(replyPayload, null, 2));
            fs.renameSync(tempPath, replyPath);
            logger.debug({ requestId: data.requestId, status: reply.status }, 'GitHub IPC reply sent');
            setTimeout(() => { try { fs.unlinkSync(replyPath); } catch {} }, 60000);
          }
        } catch (err) {
          logger.error({ err, type: data.type }, 'GitHub IPC handler error');
          if (data.requestId) {
            const repliesDir = path.join(DATA_DIR, 'ipc', sourceGroup, 'replies');
            fs.mkdirSync(repliesDir, { recursive: true });
            const replyPath = path.join(repliesDir, `${data.requestId}.json`);
            const replyPayload = { status: 'error', error: err instanceof Error ? err.message : String(err), timestamp: new Date().toISOString() };
            const tempPath = `${replyPath}.tmp`;
            fs.writeFileSync(tempPath, JSON.stringify(replyPayload, null, 2));
            fs.renameSync(tempPath, replyPath);
            setTimeout(() => { try { fs.unlinkSync(replyPath); } catch {} }, 60000);
          }
        }
      } else {
        logger.warn({ type: data.type }, 'Unknown IPC task type');
      }
  }
}

async function connectWhatsApp(): Promise<void> {
  const authDir = path.join(STORE_DIR, 'auth');
  fs.mkdirSync(authDir, { recursive: true });

  const { state, saveCreds } = await useMultiFileAuthState(authDir);

  sock = makeWASocket({
    auth: {
      creds: state.creds,
      keys: makeCacheableSignalKeyStore(state.keys, logger),
    },
    printQRInTerminal: false,
    logger,
    browser: ['NanoClaw', 'Chrome', '1.0.0'],
  });

  sock.ev.on('connection.update', (update) => {
    const { connection, lastDisconnect, qr } = update;

    if (qr) {
      const msg =
        'WhatsApp authentication required. Run /setup in Claude Code.';
      logger.error(msg);
      exec(
        `osascript -e 'display notification "${msg}" with title "NanoClaw" sound name "Basso"'`,
      );
      setTimeout(() => process.exit(1), 1000);
    }

    if (connection === 'close') {
      const reason = (lastDisconnect?.error as any)?.output?.statusCode;
      const shouldReconnect = reason !== DisconnectReason.loggedOut;
      logger.info({ reason, shouldReconnect }, 'Connection closed');

      if (shouldReconnect) {
        logger.info('Reconnecting...');
        connectWhatsApp();
      } else {
        logger.info('Logged out. Run /setup to re-authenticate.');
        process.exit(0);
      }
    } else if (connection === 'open') {
      logger.info('Connected to WhatsApp');

      // Build LID to phone mapping from auth state for self-chat translation
      if (sock.user) {
        const phoneUser = sock.user.id.split(':')[0];
        const lidUser = sock.user.lid?.split(':')[0];
        if (lidUser && phoneUser) {
          lidToPhoneMap[lidUser] = `${phoneUser}@s.whatsapp.net`;
          logger.debug({ lidUser, phoneUser }, 'LID to phone mapping set');
        }
      }

      // Sync group metadata on startup (respects 24h cache)
      syncGroupMetadata().catch((err) =>
        logger.error({ err }, 'Initial group sync failed'),
      );
      // Set up daily sync timer (only once)
      if (!groupSyncTimerStarted) {
        groupSyncTimerStarted = true;
        setInterval(() => {
          syncGroupMetadata().catch((err) =>
            logger.error({ err }, 'Periodic group sync failed'),
          );
        }, GROUP_SYNC_INTERVAL_MS);
      }
      startSharedServices();
    }
  });

  sock.ev.on('creds.update', saveCreds);

  sock.ev.on('messages.upsert', ({ messages }) => {
    for (const msg of messages) {
      if (!msg.message) continue;
      const rawJid = msg.key.remoteJid;
      if (!rawJid || rawJid === 'status@broadcast') continue;

      // Translate LID JID to phone JID if applicable
      const chatJid = translateJid(rawJid);

      const timestamp = new Date(
        Number(msg.messageTimestamp) * 1000,
      ).toISOString();

      // Always store chat metadata for group discovery
      storeChatMetadata(chatJid, timestamp);

      // Only store full message content for registered groups
      if (registeredGroups[chatJid]) {
        // Handle audio transcription asynchronously
        if (isAudioMessage(msg)) {
          const groupFolder = registeredGroups[chatJid]?.folder || 'unknown';
          const tmpDir = path.join(GROUPS_DIR, groupFolder, 'tmp');
          transcribeAudio(msg, tmpDir)
            .then((transcribedText) => {
              storeWhatsAppMessage(
                msg,
                chatJid,
                msg.key.fromMe || false,
                msg.pushName || undefined,
                transcribedText || undefined,
              );
            })
            .catch((err) => {
              logger.error({ err }, 'Audio transcription failed');
              // Store message without transcription
              storeWhatsAppMessage(
                msg,
                chatJid,
                msg.key.fromMe || false,
                msg.pushName || undefined,
              );
            });
        } else {
          storeWhatsAppMessage(
            msg,
            chatJid,
            msg.key.fromMe || false,
            msg.pushName || undefined,
          );
        }
      }
    }
  });
}

async function startMessageLoop(): Promise<void> {
  if (messageLoopRunning) {
    logger.debug('Message loop already running, skipping duplicate start');
    return;
  }
  messageLoopRunning = true;

  logger.info(`NanoClaw running (trigger: @${ASSISTANT_NAME})`);

  while (true) {
    try {
      const jids = Object.keys(registeredGroups);
      const { messages, newTimestamp } = getNewMessages(
        jids,
        lastTimestamp,
        ASSISTANT_NAME,
      );

      if (messages.length > 0) {
        logger.info({ count: messages.length }, 'New messages');

        // Advance the "seen" cursor for all messages immediately
        lastTimestamp = newTimestamp;
        saveState();

        // Deduplicate by group
        const messagesByGroup = new Map<string, NewMessage[]>();
        for (const msg of messages) {
          const existing = messagesByGroup.get(msg.chat_jid);
          if (existing) {
            existing.push(msg);
          } else {
            messagesByGroup.set(msg.chat_jid, [msg]);
          }
        }

        for (const [chatJid, groupMessages] of messagesByGroup) {
          const group = registeredGroups[chatJid];
          if (!group) continue;

          const channel = findChannel(channels, chatJid);
          if (!channel) {
            logger.warn({ chatJid }, 'No channel owns JID, skipping messages');
            continue;
          }

          const isMainGroup = group.isMain === true;
          const needsTrigger = !isMainGroup && group.requiresTrigger !== false;

          // For non-main groups, only act on trigger messages.
          // Non-trigger messages accumulate in DB and get pulled as
          // context when a trigger eventually arrives.
          if (needsTrigger) {
            const hasTrigger = groupMessages.some((m) =>
              TRIGGER_PATTERN.test(m.content.trim()),
            );
            if (!hasTrigger) continue;
          }

          // Pull all messages since lastAgentTimestamp so non-trigger
          // context that accumulated between triggers is included.
          const allPending = getMessagesSince(
            chatJid,
            lastAgentTimestamp[chatJid] || '',
            ASSISTANT_NAME,
          );
          const messagesToSend =
            allPending.length > 0 ? allPending : groupMessages;
          const formatted = formatMessages(messagesToSend);

          if (queue.sendMessage(chatJid, formatted)) {
            logger.debug(
              { chatJid, count: messagesToSend.length },
              'Piped messages to active container',
            );
            lastAgentTimestamp[chatJid] =
              messagesToSend[messagesToSend.length - 1].timestamp;
            saveState();
            // Show typing indicator while the container processes the piped message
            channel
              .setTyping?.(chatJid, true)
              ?.catch((err) =>
                logger.warn({ chatJid, err }, 'Failed to set typing indicator'),
              );
          } else {
            // No active container — enqueue for a new one
            queue.enqueueMessageCheck(chatJid);
          }
        }
      }
    } catch (err) {
      logger.error({ err }, 'Error in message loop');
    }
    await new Promise((resolve) => setTimeout(resolve, POLL_INTERVAL));
  }
}

/**
 * Startup recovery: check for unprocessed messages in registered groups.
 * Handles crash between advancing lastTimestamp and processing messages.
 */
function recoverPendingMessages(): void {
  for (const [chatJid, group] of Object.entries(registeredGroups)) {
    const sinceTimestamp = lastAgentTimestamp[chatJid] || '';
    const pending = getMessagesSince(chatJid, sinceTimestamp, ASSISTANT_NAME);
    if (pending.length > 0) {
      logger.info(
        { group: group.name, pendingCount: pending.length },
        'Recovery: found unprocessed messages',
      );
      queue.enqueueMessageCheck(chatJid);
    }
  }
}

function ensureContainerSystemRunning(): void {
  ensureContainerRuntimeRunning();
  cleanupOrphans();
}

/**
 * Start shared services (scheduler, IPC watcher, message loop).
 * Called once when the first channel connects.
 */
function startSharedServices(): void {
  if (MEMORY_ENABLED) {
    startMemoryIndexer(() => {
      const folders = new Set<string>();
      for (const group of Object.values(registeredGroups)) {
        folders.add(group.folder);
      }
      return Array.from(folders);
    });
  }

  startSchedulerLoop({
    sendMessage,
    registeredGroups: () => registeredGroups,
    getSessions: () => sessions,
    queue,
    onProcess: (groupJid, proc, containerName, groupFolder) =>
      queue.registerProcess(groupJid, proc, containerName, groupFolder),
  });
  startIpcWatcher();
  queue.setProcessMessagesFn(processGroupMessages);
  recoverPendingMessages();
  startMessageLoop();
}

async function main(): Promise<void> {
  ensureContainerSystemRunning();
  initDatabase();
  logger.info('Database initialized');

  // Initialize OpenAI for audio features
  const openaiKey = process.env.OPENAI_API_KEY;
  if (openaiKey) {
    initOpenAI(openaiKey);
  } else {
    logger.warn('OPENAI_API_KEY not set - audio transcription and TTS disabled');
  }

  // Initialize Gemini for image generation
  const googleApiKey = process.env.GOOGLE_API_KEY;
  if (googleApiKey) {
    initGemini(googleApiKey);
  } else {
    logger.warn('GOOGLE_API_KEY not set - image generation disabled');
  }

  // Initialize GitHub API client
  const githubToken = process.env.GITHUB_TOKEN;
  if (githubToken) {
    initGitHubClient(githubToken);
    logger.info('GitHub integration enabled');

    // Start webhook server for GitHub events
    const webhookPort = parseInt(process.env.WEBHOOK_PORT || '3000', 10);
    startWebhookServer(webhookPort, queue);

    // Sweep closed-issue groups every hour
    setInterval(() => {
      sweepClosedIssueGroups().catch((err) =>
        logger.error({ err }, 'Issue group sweeper failed'),
      );
    }, 60 * 60 * 1000);
  } else {
    logger.warn('GITHUB_TOKEN not set - GitHub integration disabled');
  }

  loadState();

  // Graceful shutdown handlers
  const shutdown = async (signal: string) => {
    logger.info({ signal }, 'Shutdown signal received');
    await shutdownPool();
    await queue.shutdown(10000);
    closeDatabase();
    closeEmbeddingsDb();
    for (const ch of channels) await ch.disconnect();
    process.exit(0);
  };
  process.on('SIGTERM', () => shutdown('SIGTERM'));
  process.on('SIGINT', () => shutdown('SIGINT'));

  // Connect channel-registry channels (upstream pattern)
  // Each channel self-registers via the barrel import above.
  // Factories return null when credentials are missing, so unconfigured channels are skipped.
  for (const channelName of getRegisteredChannelNames()) {
    const factory = getChannelFactory(channelName)!;
    const channelOpts = {
      onMessage: (_chatJid: string, msg: NewMessage) => storeMessage(msg),
      onChatMetadata: (
        chatJid: string,
        timestamp: string,
        name?: string,
        channel?: string,
        isGroup?: boolean,
      ) => storeChatMetadata(chatJid, timestamp, name, channel, isGroup),
      registeredGroups: () => registeredGroups,
    };
    const channel = factory(channelOpts);
    if (!channel) {
      logger.warn(
        { channel: channelName },
        'Channel installed but credentials missing — skipping. Check .env or re-run the channel skill.',
      );
      continue;
    }
    channels.push(channel);
    await channel.connect();
  }

  // Also connect our direct integrations (WhatsApp + Discord) which aren't
  // in the channel registry yet
  if (!WHATSAPP_ENABLED && !DISCORD_ENABLED && channels.length === 0) {
    logger.error(
      'No channels configured. Set DISCORD_BOT_TOKEN env var or authenticate WhatsApp.',
    );
    process.exit(1);
  }

  if (DISCORD_ENABLED) {
    await connectDiscord({
      onMessage: (chatJid, isRegistered) => {
        // Always process Discord messages - auto-register channels on first use
        queue.enqueueMessageCheck(chatJid);
      },
    });
  }

  if (WHATSAPP_ENABLED) {
    await connectWhatsApp();
  } else {
    // No WhatsApp — start shared services directly
    // (when WhatsApp is enabled, these are started in the connection.open handler)
    logger.info('WhatsApp not configured, starting services directly');
    startSharedServices();
  }
}

// Guard: only run when executed directly, not when imported by tests
const isDirectRun =
  process.argv[1] &&
  new URL(import.meta.url).pathname ===
    new URL(`file://${process.argv[1]}`).pathname;

if (isDirectRun) {
  main().catch((err) => {
    logger.error({ err }, 'Failed to start NanoClaw');
    process.exit(1);
  });
}
