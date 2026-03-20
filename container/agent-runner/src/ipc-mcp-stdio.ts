/**
 * Stdio MCP Server for NanoClaw
 * Standalone process that Claude Code connects to as a stdio MCP server.
 * Reads per-query context from /workspace/ipc/context.json (updated by host before each query).
 * Writes IPC files for the host process to pick up.
 */

import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { z } from 'zod';
import crypto from 'crypto';
import fs from 'fs';
import path from 'path';
import { CronExpressionParser } from 'cron-parser';
import { search as memorySearch } from './memory-search.js';
import { generateTts, generateSoundEffect, generateMusic } from './elevenlabs.js';

const IPC_DIR = '/workspace/ipc';
const MESSAGES_DIR = path.join(IPC_DIR, 'messages');
const TASKS_DIR = path.join(IPC_DIR, 'tasks');
const CONTEXT_FILE = path.join(IPC_DIR, 'context.json');

// ─── Dynamic per-query context ──────────────────────────────────────────────

interface IpcMcpContext {
  chatJid: string;
  groupFolder: string;
  isMain: boolean;
}

/**
 * Read context from /workspace/ipc/context.json (written by host before each query).
 * Falls back to environment variables for backward compatibility / initial startup.
 */
function getContext(): IpcMcpContext {
  try {
    if (fs.existsSync(CONTEXT_FILE)) {
      const ctx = JSON.parse(fs.readFileSync(CONTEXT_FILE, 'utf-8'));
      return {
        chatJid: ctx.chatJid || process.env.NANOCLAW_CHAT_JID || '',
        groupFolder: ctx.groupFolder || process.env.NANOCLAW_GROUP_FOLDER || '',
        isMain: ctx.isMain ?? process.env.NANOCLAW_IS_MAIN === '1',
      };
    }
  } catch {
    // Fall through to env vars
  }
  return {
    chatJid: process.env.NANOCLAW_CHAT_JID || '',
    groupFolder: process.env.NANOCLAW_GROUP_FOLDER || '',
    isMain: process.env.NANOCLAW_IS_MAIN === '1',
  };
}

// ─── IPC helpers ─────────────────────────────────────────────────────────────

function writeIpcFile(dir: string, data: Record<string, unknown>): string {
  fs.mkdirSync(dir, { recursive: true });

  const requestId = crypto.randomUUID();
  const filename = `${requestId}.json`;
  const filepath = path.join(dir, filename);

  // Add requestId to payload for reply matching
  data.requestId = requestId;

  // Atomic write: temp file then rename
  const tempPath = `${filepath}.tmp`;
  fs.writeFileSync(tempPath, JSON.stringify(data, null, 2));
  fs.renameSync(tempPath, filepath);

  return requestId;
}

async function waitForReply(requestId: string, timeoutMs: number): Promise<any> {
  const REPLIES_DIR = path.join(IPC_DIR, 'replies');
  const replyPath = path.join(REPLIES_DIR, `${requestId}.json`);
  const startTime = Date.now();

  while (Date.now() - startTime < timeoutMs) {
    try {
      if (fs.existsSync(replyPath)) {
        const reply = JSON.parse(fs.readFileSync(replyPath, 'utf-8'));
        fs.unlinkSync(replyPath); // Clean up
        return reply;
      }
    } catch {
      // File might be mid-write, continue polling
    }
    await new Promise(resolve => setTimeout(resolve, 500));
  }

  throw new Error(`Timeout waiting for reply after ${timeoutMs}ms`);
}

// ─── MCP Server ──────────────────────────────────────────────────────────────

const server = new McpServer({
  name: 'nanoclaw',
  version: '1.0.0',
});

// ─── send_message ────────────────────────────────────────────────────────────

server.tool(
  'send_message',
  "Send a message to the user or group immediately while you're still running. Use this for progress updates or to send multiple messages. You can call this multiple times. Note: when running as a scheduled task, your final output is NOT sent to the user — use this tool if you need to communicate with the user or group.",
  {
    text: z.string().describe('The message text to send'),
    sender: z.string().optional().describe('Your role/identity name (e.g. "Researcher"). When set, messages appear from a dedicated bot in Telegram.'),
  },
  async (args) => {
    const ctx = getContext();
    const data: Record<string, unknown> = {
      type: 'message',
      chatJid: ctx.chatJid,
      text: args.text,
      sender: args.sender || undefined,
      groupFolder: ctx.groupFolder,
      timestamp: new Date().toISOString(),
    };

    writeIpcFile(MESSAGES_DIR, data);

    return { content: [{ type: 'text' as const, text: 'Message sent.' }] };
  },
);

// ─── send_email ─────────────────────────────────────────────────────────────

server.tool(
  'send_email',
  'Send an email via Yahoo (bagel.agent@yahoo.com). Supports text body and optional file attachments. Main group only.',
  {
    to: z.string().describe('Recipient email address'),
    subject: z.string().describe('Email subject line'),
    body: z.string().describe('Email body text'),
    attachments: z.array(z.object({
      filename: z.string().describe('Display filename (e.g. "chart.png")'),
      path: z.string().describe('Absolute path to file (must be under /workspace/group/ or /workspace/project/)'),
      inline: z.boolean().optional().describe('If true, embed as inline image in HTML body instead of attachment'),
    })).optional().describe('Optional file attachments'),
  },
  async (args) => {
    const ctx = getContext();
    const data: Record<string, unknown> = {
      type: 'email',
      to: args.to,
      subject: args.subject,
      body: args.body,
      attachments: args.attachments || [],
      groupFolder: ctx.groupFolder,
      timestamp: new Date().toISOString(),
    };

    writeIpcFile(MESSAGES_DIR, data);

    return { content: [{ type: 'text' as const, text: `Email queued for delivery to ${args.to}.` }] };
  },
);

// ─── send_voice_message ──────────────────────────────────────────────────────

server.tool(
  'send_voice_message',
  'Send a voice message (audio) to the user or group. The text will be converted to speech using OpenAI TTS. Use this when the user explicitly asks for a voice response.',
  {
    text: z.string().describe('The text to convert to speech and send as a voice message'),
    voice: z.enum(['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer']).default('nova').describe('Voice to use for TTS. nova is warm and friendly (default), alloy is neutral, echo is deep, fable is expressive, onyx is authoritative, shimmer is bright'),
  },
  async (args) => {
    const ctx = getContext();
    const data: Record<string, unknown> = {
      type: 'voice_message',
      chatJid: ctx.chatJid,
      text: args.text,
      voice: args.voice,
      groupFolder: ctx.groupFolder,
      timestamp: new Date().toISOString(),
    };

    writeIpcFile(MESSAGES_DIR, data);

    return { content: [{ type: 'text' as const, text: 'Voice message queued for delivery.' }] };
  },
);

// ─── send_image ──────────────────────────────────────────────────────────────

server.tool(
  'send_image',
  'Send an image to the user or group. Provide either a local file path or a base64-encoded image string. IMPORTANT: image_path must be under /workspace/group/ or /workspace/project/ — these are the only paths accessible to the host. Save images to /workspace/group/ first if needed.',
  {
    caption: z.string().optional().describe('Optional caption/text to send with the image'),
    image_path: z.string().optional().describe('Absolute path under /workspace/group/ or /workspace/project/ (e.g., "/workspace/group/output.png"). Paths outside these directories are NOT accessible to the host and will fail.'),
    image_base64: z.string().optional().describe('Base64-encoded image data (without data:image/png;base64, prefix)'),
    filename: z.string().optional().describe('Filename for the image (e.g., "chart.png"). Required if using image_base64.'),
  },
  async (args) => {
    if (!args.image_path && !args.image_base64) {
      return {
        content: [{ type: 'text' as const, text: 'Error: Must provide either image_path or image_base64' }],
        isError: true,
      };
    }

    if (args.image_path && args.image_base64) {
      return {
        content: [{ type: 'text' as const, text: 'Error: Cannot provide both image_path and image_base64' }],
        isError: true,
      };
    }

    if (args.image_base64 && !args.filename) {
      return {
        content: [{ type: 'text' as const, text: 'Error: filename is required when using image_base64' }],
        isError: true,
      };
    }

    if (args.image_path) {
      const resolved = path.resolve(args.image_path);
      const allowedPrefixes = ['/workspace/group/', '/workspace/project/'];
      if (!allowedPrefixes.some(prefix => resolved.startsWith(prefix))) {
        return {
          content: [{ type: 'text' as const, text: 'Error: image_path must be under /workspace/group/ or /workspace/project/' }],
          isError: true,
        };
      }
    }

    const ctx = getContext();
    const data: Record<string, unknown> = {
      type: 'image',
      chatJid: ctx.chatJid,
      caption: args.caption,
      imagePath: args.image_path,
      imageBase64: args.image_base64,
      filename: args.filename,
      groupFolder: ctx.groupFolder,
      timestamp: new Date().toISOString(),
    };

    writeIpcFile(MESSAGES_DIR, data);

    return { content: [{ type: 'text' as const, text: 'Image queued for delivery.' }] };
  },
);

// ─── send_audio ──────────────────────────────────────────────────────────────

server.tool(
  'send_audio',
  'Send an audio file to the user or group. The file must be under /workspace/group/ or /workspace/project/.',
  {
    audio_path: z.string().describe('Absolute path to the audio file (must be under /workspace/group/ or /workspace/project/)'),
    caption: z.string().optional().describe('Optional caption to send with the audio'),
    as_voice: z.boolean().default(false).describe('If true, send as a push-to-talk voice message (WhatsApp PTT). If false, send as an audio file attachment.'),
  },
  async (args) => {
    const resolved = path.resolve(args.audio_path);
    const allowedPrefixes = ['/workspace/group/', '/workspace/project/'];
    if (!allowedPrefixes.some(prefix => resolved.startsWith(prefix))) {
      return {
        content: [{ type: 'text' as const, text: 'Error: audio_path must be under /workspace/group/ or /workspace/project/' }],
        isError: true,
      };
    }

    const ctx = getContext();
    const data: Record<string, unknown> = {
      type: 'audio_file',
      chatJid: ctx.chatJid,
      audioPath: args.audio_path,
      caption: args.caption,
      asVoice: args.as_voice || false,
      groupFolder: ctx.groupFolder,
      timestamp: new Date().toISOString(),
    };

    writeIpcFile(MESSAGES_DIR, data);

    return { content: [{ type: 'text' as const, text: 'Audio queued for delivery.' }] };
  },
);

// ─── generate_image ──────────────────────────────────────────────────────────

server.tool(
  'generate_image',
  `Generate an image from a text description using AI (Gemini). Returns a file path you can send with send_image.

Tips: Be specific and descriptive. Include style (photorealistic, watercolor, pixel art), composition (close-up, wide angle), and mood.`,
  {
    prompt: z.string().describe('Detailed description of the image to generate'),
    aspect_ratio: z.enum(['1:1', '3:4', '4:3', '9:16', '16:9'])
      .default('1:1')
      .describe('1:1=square, 16:9=landscape, 9:16=portrait, 4:3/3:4=standard'),
  },
  async (args) => {
    const ctx = getContext();
    const requestId = writeIpcFile(TASKS_DIR, {
      type: 'generate_image',
      prompt: args.prompt,
      aspectRatio: args.aspect_ratio || '1:1',
      groupFolder: ctx.groupFolder,
      chatJid: ctx.chatJid,
      timestamp: new Date().toISOString(),
    });

    try {
      const reply = await waitForReply(requestId, 60000);

      if (reply.status === 'success') {
        return {
          content: [{
            type: 'text' as const,
            text: `Image generated! Saved to: ${reply.data.containerPath}\n\nUse send_image with image_path="${reply.data.containerPath}" to send it.`,
          }],
        };
      }
      return {
        content: [{ type: 'text' as const, text: `Image generation failed: ${reply.error}` }],
        isError: true,
      };
    } catch (err) {
      return {
        content: [{ type: 'text' as const, text: `Image generation error: ${err instanceof Error ? err.message : String(err)}` }],
        isError: true,
      };
    }
  },
);

// ─── schedule_task ───────────────────────────────────────────────────────────

server.tool(
  'schedule_task',
  `Schedule a recurring or one-time task. The task will run as a full agent with access to all tools.

CONTEXT MODE - Choose based on task type:
\u2022 "group": Task runs in the group's conversation context, with access to chat history. Use for tasks that need context about ongoing discussions, user preferences, or recent interactions.
\u2022 "isolated": Task runs in a fresh session with no conversation history. Use for independent tasks that don't need prior context. When using isolated mode, include all necessary context in the prompt itself.

If unsure which mode to use, you can ask the user. Examples:
- "Remind me about our discussion" \u2192 group (needs conversation context)
- "Check the weather every morning" \u2192 isolated (self-contained task)
- "Follow up on my request" \u2192 group (needs to know what was requested)
- "Generate a daily report" \u2192 isolated (just needs instructions in prompt)

MESSAGING BEHAVIOR - The task agent's output is sent to the user or group. It can also use send_message for immediate delivery, or wrap output in <internal> tags to suppress it. Include guidance in the prompt about whether the agent should:
\u2022 Always send a message (e.g., reminders, daily briefings)
\u2022 Only send a message when there's something to report (e.g., "notify me if...")
\u2022 Never send a message (background maintenance tasks)

SCHEDULE VALUE FORMAT (all times are LOCAL timezone):
\u2022 cron: Standard cron expression (e.g., "*/5 * * * *" for every 5 minutes, "0 9 * * *" for daily at 9am LOCAL time)
\u2022 interval: Milliseconds between runs (e.g., "300000" for 5 minutes, "3600000" for 1 hour)
\u2022 once: Local time WITHOUT "Z" suffix (e.g., "2026-02-01T15:30:00"). Do NOT use UTC/Z suffix.`,
  {
    prompt: z.string().describe('What the agent should do when the task runs. For isolated mode, include all necessary context here.'),
    schedule_type: z.enum(['cron', 'interval', 'once']).describe('cron=recurring at specific times, interval=recurring every N ms, once=run once at specific time'),
    schedule_value: z.string().describe('cron: "*/5 * * * *" | interval: milliseconds like "300000" | once: local timestamp like "2026-02-01T15:30:00" (no Z suffix!)'),
    context_mode: z.enum(['group', 'isolated']).default('group').describe('group=runs with chat history and memory, isolated=fresh session (include context in prompt)'),
    target_group_jid: z.string().optional().describe('(Main group only) JID of the group to schedule the task for. Defaults to the current group.'),
  },
  async (args) => {
    const ctx = getContext();

    // Validate schedule_value before writing IPC
    if (args.schedule_type === 'cron') {
      try {
        CronExpressionParser.parse(args.schedule_value);
      } catch {
        return {
          content: [{ type: 'text' as const, text: `Invalid cron: "${args.schedule_value}". Use format like "0 9 * * *" (daily 9am) or "*/5 * * * *" (every 5 min).` }],
          isError: true,
        };
      }
    } else if (args.schedule_type === 'interval') {
      const ms = parseInt(args.schedule_value, 10);
      if (isNaN(ms) || ms <= 0) {
        return {
          content: [{ type: 'text' as const, text: `Invalid interval: "${args.schedule_value}". Must be positive milliseconds (e.g., "300000" for 5 min).` }],
          isError: true,
        };
      }
    } else if (args.schedule_type === 'once') {
      if (/[Zz]$/.test(args.schedule_value) || /[+-]\d{2}:\d{2}$/.test(args.schedule_value)) {
        return {
          content: [{ type: 'text' as const, text: `Timestamp must be local time without timezone suffix. Got "${args.schedule_value}" — use format like "2026-02-01T15:30:00".` }],
          isError: true,
        };
      }
      const date = new Date(args.schedule_value);
      if (isNaN(date.getTime())) {
        return {
          content: [{ type: 'text' as const, text: `Invalid timestamp: "${args.schedule_value}". Use local time format like "2026-02-01T15:30:00".` }],
          isError: true,
        };
      }
    }

    // Non-main groups can only schedule for themselves
    const targetJid = ctx.isMain && args.target_group_jid ? args.target_group_jid : ctx.chatJid;

    const data: Record<string, unknown> = {
      type: 'schedule_task',
      prompt: args.prompt,
      schedule_type: args.schedule_type,
      schedule_value: args.schedule_value,
      context_mode: args.context_mode || 'group',
      targetJid,
      createdBy: ctx.groupFolder,
      timestamp: new Date().toISOString(),
    };

    const requestId = writeIpcFile(TASKS_DIR, data);

    return {
      content: [{ type: 'text' as const, text: `Task scheduled (${requestId}): ${args.schedule_type} - ${args.schedule_value}` }],
    };
  },
);

// ─── list_tasks ──────────────────────────────────────────────────────────────

server.tool(
  'list_tasks',
  "List all scheduled tasks. From main: shows all tasks. From other groups: shows only that group's tasks.",
  {},
  async () => {
    const ctx = getContext();
    const tasksFile = path.join(IPC_DIR, 'current_tasks.json');

    try {
      if (!fs.existsSync(tasksFile)) {
        return { content: [{ type: 'text' as const, text: 'No scheduled tasks found.' }] };
      }

      const allTasks = JSON.parse(fs.readFileSync(tasksFile, 'utf-8'));

      const tasks = ctx.isMain
        ? allTasks
        : allTasks.filter((t: { groupFolder: string }) => t.groupFolder === ctx.groupFolder);

      if (tasks.length === 0) {
        return { content: [{ type: 'text' as const, text: 'No scheduled tasks found.' }] };
      }

      const formatted = tasks
        .map(
          (t: { id: string; prompt: string; schedule_type: string; schedule_value: string; status: string; next_run: string }) =>
            `- [${t.id}] ${t.prompt.slice(0, 50)}... (${t.schedule_type}: ${t.schedule_value}) - ${t.status}, next: ${t.next_run || 'N/A'}`,
        )
        .join('\n');

      return { content: [{ type: 'text' as const, text: `Scheduled tasks:\n${formatted}` }] };
    } catch (err) {
      return {
        content: [{ type: 'text' as const, text: `Error reading tasks: ${err instanceof Error ? err.message : String(err)}` }],
      };
    }
  },
);

// ─── pause_task ──────────────────────────────────────────────────────────────

server.tool(
  'pause_task',
  'Pause a scheduled task. It will not run until resumed.',
  { task_id: z.string().describe('The task ID to pause') },
  async (args) => {
    const ctx = getContext();
    writeIpcFile(TASKS_DIR, {
      type: 'pause_task',
      taskId: args.task_id,
      groupFolder: ctx.groupFolder,
      isMain: ctx.isMain,
      timestamp: new Date().toISOString(),
    });

    return { content: [{ type: 'text' as const, text: `Task ${args.task_id} pause requested.` }] };
  },
);

// ─── resume_task ─────────────────────────────────────────────────────────────

server.tool(
  'resume_task',
  'Resume a paused task.',
  { task_id: z.string().describe('The task ID to resume') },
  async (args) => {
    const ctx = getContext();
    writeIpcFile(TASKS_DIR, {
      type: 'resume_task',
      taskId: args.task_id,
      groupFolder: ctx.groupFolder,
      isMain: ctx.isMain,
      timestamp: new Date().toISOString(),
    });

    return { content: [{ type: 'text' as const, text: `Task ${args.task_id} resume requested.` }] };
  },
);

// ─── cancel_task ─────────────────────────────────────────────────────────────

server.tool(
  'cancel_task',
  'Cancel and delete a scheduled task.',
  { task_id: z.string().describe('The task ID to cancel') },
  async (args) => {
    const ctx = getContext();
    writeIpcFile(TASKS_DIR, {
      type: 'cancel_task',
      taskId: args.task_id,
      groupFolder: ctx.groupFolder,
      isMain: ctx.isMain,
      timestamp: new Date().toISOString(),
    });

    return { content: [{ type: 'text' as const, text: `Task ${args.task_id} cancellation requested.` }] };
  },
);

// ─── register_group ──────────────────────────────────────────────────────────

server.tool(
  'register_group',
  `Register a new chat/group so the agent can respond to messages there. Main group only.

Use available_groups.json to find the JID for a group. The folder name must be channel-prefixed: "{channel}_{group-name}" (e.g., "whatsapp_family-chat", "telegram_dev-team", "discord_general"). Use lowercase with hyphens for the group name part.`,
  {
    jid: z.string().describe('The chat JID (e.g., "120363336345536173@g.us", "tg:-1001234567890", "dc:1234567890123456")'),
    name: z.string().describe('Display name for the group'),
    folder: z.string().describe('Channel-prefixed folder name (e.g., "whatsapp_family-chat", "telegram_dev-team")'),
    trigger: z.string().describe('Trigger word (e.g., "@Andy")'),
  },
  async (args) => {
    const ctx = getContext();
    if (!ctx.isMain) {
      return {
        content: [{ type: 'text' as const, text: 'Only the main group can register new groups.' }],
        isError: true,
      };
    }

    writeIpcFile(TASKS_DIR, {
      type: 'register_group',
      jid: args.jid,
      name: args.name,
      folder: args.folder,
      trigger: args.trigger,
      timestamp: new Date().toISOString(),
    });

    return {
      content: [{ type: 'text' as const, text: `Group "${args.name}" registered. It will start receiving messages immediately.` }],
    };
  },
);

// ─── deploy (main only) ─────────────────────────────────────────────────────

server.tool(
  'deploy',
  `Deploy code changes to NanoClaw. Main group only.

This tool commits your changes, builds the specified targets, and restarts the service. The current session will end after calling this.

BEFORE calling deploy:
1. Make your changes to files under /workspace/project/
2. Type-check: cd /workspace/project && npx tsc --noEmit (for host changes in src/)
3. For container changes: cd /workspace/project/container/agent-runner && npx tsc --noEmit
4. Review changes: cd /workspace/project && git diff
5. Only then call deploy

TARGETS — what to rebuild:
\u2022 "host" — changes to src/*.ts (host process code)
\u2022 "container" — changes to container/agent-runner/ (agent code, MCP tools, Dockerfile)
\u2022 Both — changes spanning host and container code
\u2022 Neither — changes to CLAUDE.md, docs, or non-code files (just commits, no rebuild/restart)

After deploy, the service restarts automatically. Your session ends — the user will see the service come back online.`,
  {
    targets: z.array(z.enum(['host', 'container'])).describe('What to rebuild. Empty array = commit only (no rebuild/restart).'),
    commit_message: z.string().optional().describe('Git commit message describing your changes. Defaults to "chore: agent-initiated deploy".'),
  },
  async (args) => {
    const ctx = getContext();
    if (!ctx.isMain) {
      return {
        content: [{ type: 'text' as const, text: 'Deploy is only available from the main group.' }],
        isError: true,
      };
    }

    const requestId = writeIpcFile(TASKS_DIR, {
      type: 'deploy',
      targets: args.targets,
      commitMessage: args.commit_message,
      chatJid: ctx.chatJid,
      groupFolder: ctx.groupFolder,
      timestamp: new Date().toISOString(),
    });

    try {
      const reply = await waitForReply(requestId, 300000);

      if (reply.status === 'success') {
        // If container was rebuilt, schedule container restart
        if (args.targets.includes('container')) {
          setTimeout(() => {
            writeIpcFile(TASKS_DIR, {
              type: 'restart_container',
              groupFolder: 'main',
              chatJid: ctx.chatJid,
              timestamp: new Date().toISOString(),
            });
          }, 2000);
        }

        return {
          content: [{
            type: 'text' as const,
            text: `Deploy succeeded!\n\n${reply.message || ''}\n\nTargets: ${args.targets.join(', ') || 'none (commit only)'}${args.targets.includes('container') ? '\n\nContainer will restart in 2 seconds to load new code.' : ''}`,
          }],
        };
      } else {
        return {
          content: [{
            type: 'text' as const,
            text: `Deploy failed: ${reply.error}\n\nThe host has already notified the user with details. Check your changes and try again.`,
          }],
          isError: true,
        };
      }
    } catch (err) {
      return {
        content: [{
          type: 'text' as const,
          text: `Deploy timeout or error: ${err instanceof Error ? err.message : String(err)}\n\nThe host may still be processing (builds take time). Check host logs or ask the user.`,
        }],
        isError: true,
      };
    }
  },
);

// ─── test_container_build (main only) ────────────────────────────────────────

server.tool(
  'test_container_build',
  'Test the container build script to verify it works. Runs ./container/build.sh on the host and returns the output. Use this after modifying build.sh or Dockerfile to ensure builds work before deploying.',
  {},
  async () => {
    const ctx = getContext();
    if (!ctx.isMain) {
      return {
        content: [{ type: 'text' as const, text: 'Only the main group can test container builds.' }],
        isError: true,
      };
    }

    writeIpcFile(TASKS_DIR, {
      type: 'test_container_build',
      timestamp: new Date().toISOString(),
    });

    return {
      content: [{ type: 'text' as const, text: 'Container build test initiated. The host will run ./container/build.sh and return the results via a message.' }],
    };
  },
);

// ─── restart_container (main only) ───────────────────────────────────────────

server.tool(
  'restart_container',
  'Restart the persistent container to pick up new code changes. Use this after deploying container changes to ensure the new code is loaded immediately instead of waiting for idle timeout.',
  {
    groupFolder: z.string().describe('The group folder name to restart (typically "main")'),
  },
  async (args) => {
    const ctx = getContext();
    if (!ctx.isMain) {
      return {
        content: [{ type: 'text' as const, text: 'Only the main group can restart containers.' }],
        isError: true,
      };
    }

    writeIpcFile(TASKS_DIR, {
      type: 'restart_container',
      groupFolder: args.groupFolder,
      timestamp: new Date().toISOString(),
    });

    return {
      content: [{ type: 'text' as const, text: `Container restart requested for group: ${args.groupFolder}. The container will shut down and the next query will spawn a fresh container with new code.` }],
    };
  },
);

// ─── ElevenLabs audio tools ─────────────────────────────────────────────────

server.tool(
  'elevenlabs_tts',
  'Generate speech audio from text using ElevenLabs TTS. Returns a file path. Use send_audio to deliver it.',
  {
    text: z.string().describe('The text to convert to speech'),
    voice_id: z.string().optional().describe('ElevenLabs voice ID (default: Rachel). Common voices: JBFqnCBsd6RMkjVDRZzb (Rachel), 21m00Tcm4TlvDq8ikWAM (Adam), EXAVITQu4vr4xnSDxMaL (Bella)'),
    model_id: z.string().optional().describe('Model ID (default: eleven_multilingual_v2)'),
  },
  async (args) => {
    try {
      const outputDir = '/workspace/group';
      const filePath = await generateTts(args.text, args.voice_id, args.model_id, outputDir);
      return {
        content: [{ type: 'text' as const, text: `Speech generated: ${filePath}\n\nUse send_audio with audio_path="${filePath}" to send it (set as_voice=true for push-to-talk style).` }],
      };
    } catch (err) {
      return {
        content: [{ type: 'text' as const, text: `ElevenLabs TTS error: ${err instanceof Error ? err.message : String(err)}` }],
        isError: true,
      };
    }
  },
);

server.tool(
  'elevenlabs_sound_effect',
  'Generate a sound effect from a text description using ElevenLabs. Returns a file path.',
  {
    prompt: z.string().describe('Description of the sound effect (e.g., "thunderstorm with heavy rain", "cat purring")'),
    duration_seconds: z.number().min(0.5).max(30).optional().describe('Duration in seconds (0.5-30). Omit for auto duration.'),
  },
  async (args) => {
    try {
      const outputDir = '/workspace/group';
      const filePath = await generateSoundEffect(args.prompt, args.duration_seconds, outputDir);
      return {
        content: [{ type: 'text' as const, text: `Sound effect generated: ${filePath}\n\nUse send_audio with audio_path="${filePath}" to send it.` }],
      };
    } catch (err) {
      return {
        content: [{ type: 'text' as const, text: `ElevenLabs sound effect error: ${err instanceof Error ? err.message : String(err)}` }],
        isError: true,
      };
    }
  },
);

server.tool(
  'elevenlabs_music',
  'Generate music from a text prompt using ElevenLabs. Returns a file path.',
  {
    prompt: z.string().describe('Description of the music (e.g., "chill lo-fi beats with soft piano", "epic orchestral battle music")'),
    duration_seconds: z.number().min(3).max(300).optional().describe('Duration in seconds (3-300). Omit for default.'),
    instrumental: z.boolean().optional().describe('Force instrumental only (no vocals). Default: false.'),
  },
  async (args) => {
    try {
      const outputDir = '/workspace/group';
      const durationMs = args.duration_seconds ? args.duration_seconds * 1000 : undefined;
      const filePath = await generateMusic(args.prompt, durationMs, args.instrumental, outputDir);
      return {
        content: [{ type: 'text' as const, text: `Music generated: ${filePath}\n\nUse send_audio with audio_path="${filePath}" to send it.` }],
      };
    } catch (err) {
      return {
        content: [{ type: 'text' as const, text: `ElevenLabs music error: ${err instanceof Error ? err.message : String(err)}` }],
        isError: true,
      };
    }
  },
);

// ─── semantic_search ─────────────────────────────────────────────────────────

server.tool(
  'semantic_search',
  `Search your memory using natural language. Finds relevant content by meaning, not just keywords.

Use this tool proactively whenever context would help:
- A name comes up — pull everything you know about that person
- A topic surfaces — find related notes, conversations, past work
- Before making commitments — check what you said before
- When you need more context about something — search for it
- When someone references a past event or discussion — look it up

Returns the most relevant chunks from your memory files, ranked by relevance.`,
  {
    query: z.string().describe("Natural language search query — describe what you're looking for"),
    mode: z.enum(['hybrid', 'semantic', 'keyword']).default('hybrid').describe('hybrid (recommended): combines meaning + keywords. semantic: meaning only. keyword: exact word matches only.'),
    limit: z.number().int().min(1).max(20).default(5).describe('Number of results to return (1-20)'),
  },
  async (args) => {
    const ctx = getContext();
    try {
      // Main group searches all memories; non-main groups only search their own
      const results = await memorySearch(args.query, args.mode, args.limit, ctx.isMain ? undefined : ctx.groupFolder);

      if (results.length === 0) {
        return { content: [{ type: 'text' as const, text: 'No matching memories found.' }] };
      }

      const formatted = results.map((r, i) =>
        `[${i + 1}] Source: ${r.source} (${r.type})\n${r.content}`,
      ).join('\n\n---\n\n');

      return {
        content: [{ type: 'text' as const, text: `Found ${results.length} relevant memories:\n\n${formatted}` }],
      };
    } catch (err) {
      return {
        content: [{ type: 'text' as const, text: `Memory search error: ${err instanceof Error ? err.message : String(err)}` }],
        isError: true,
      };
    }
  },
);

// ─── GitHub tools ────────────────────────────────────────────────────────────

function registerGitHubTool(
  name: string,
  description: string,
  schema: Record<string, z.ZodTypeAny>,
  ipcType: string,
  timeoutMs: number = 30000,
) {
  server.tool(name, description, schema, async (args) => {
    const ctx = getContext();
    const isGitHubGroup = ctx.groupFolder.startsWith('github-');
    if (!ctx.isMain && !isGitHubGroup) {
      return {
        content: [{ type: 'text' as const, text: 'GitHub tools are only available from the main group or GitHub groups.' }],
        isError: true,
      };
    }

    const requestId = writeIpcFile(TASKS_DIR, {
      type: ipcType,
      ...args,
      timestamp: new Date().toISOString(),
    });

    try {
      const reply = await waitForReply(requestId, timeoutMs);

      if (reply.status === 'success') {
        return {
          content: [{
            type: 'text' as const,
            text: reply.data ? JSON.stringify(reply.data, null, 2) : 'Success',
          }],
        };
      } else {
        return {
          content: [{ type: 'text' as const, text: `Failed: ${reply.error}` }],
          isError: true,
        };
      }
    } catch (err) {
      return {
        content: [{ type: 'text' as const, text: `Timeout or error: ${err instanceof Error ? err.message : String(err)}` }],
        isError: true,
      };
    }
  });
}

registerGitHubTool(
  'github_fetch_issue',
  'Fetch details about a GitHub issue. Returns issue title, description, labels, and other metadata.',
  {
    owner: z.string().describe('Repository owner (e.g., "dkador")'),
    repo: z.string().describe('Repository name (e.g., "takeover-game")'),
    issue_number: z.number().describe('Issue number'),
  },
  'github_fetch_issue',
);

registerGitHubTool(
  'github_clone_repo',
  'Clone a GitHub repository to the local workspace. Returns the path to the cloned repository.',
  {
    owner: z.string().describe('Repository owner'),
    repo: z.string().describe('Repository name'),
  },
  'github_clone_repo',
  120000, // 2 min timeout for clone
);

registerGitHubTool(
  'github_create_branch',
  'Create a new git branch in a cloned repository.',
  {
    repo_path: z.string().describe('Path to the cloned repository'),
    branch_name: z.string().describe('Name for the new branch (e.g., "bagel/issue-5-fix-bug")'),
    base_branch: z.string().optional().describe('Base branch to branch from (defaults to main/master)'),
  },
  'github_create_branch',
);

registerGitHubTool(
  'github_commit_push',
  'Commit all changes in the repository and push to remote.',
  {
    repo_path: z.string().describe('Path to the repository'),
    branch_name: z.string().describe('Branch name to push'),
    commit_message: z.string().describe('Commit message'),
  },
  'github_commit_push',
  60000,
);

registerGitHubTool(
  'github_create_pr',
  'Create a pull request on GitHub.',
  {
    owner: z.string().describe('Repository owner'),
    repo: z.string().describe('Repository name'),
    title: z.string().describe('PR title'),
    body: z.string().describe('PR description/body'),
    head: z.string().describe('Source branch (e.g., "bagel/issue-5-fix")'),
    base: z.string().optional().describe('Target branch (defaults to main)'),
  },
  'github_create_pr',
);

registerGitHubTool(
  'github_comment',
  'Add a comment to a GitHub issue or pull request.',
  {
    owner: z.string().describe('Repository owner'),
    repo: z.string().describe('Repository name'),
    issue_number: z.number().describe('Issue or PR number'),
    body: z.string().describe('Comment text'),
  },
  'github_comment',
);

registerGitHubTool(
  'github_merge_pr',
  'Merge a pull request on GitHub. Use after creating a PR to auto-merge it.',
  {
    owner: z.string().describe('Repository owner'),
    repo: z.string().describe('Repository name'),
    pull_number: z.number().describe('Pull request number'),
    merge_method: z.enum(['merge', 'squash', 'rebase']).optional().describe('Merge method (defaults to squash)'),
  },
  'github_merge_pr',
);

registerGitHubTool(
  'github_update_branch',
  'Update the current branch by merging the latest changes from the base branch (e.g. main). Use this when a merge or push fails because the branch is behind.',
  {
    repo_path: z.string().describe('Path to the repository'),
    base_branch: z.string().optional().describe('Base branch to merge from (defaults to main/master)'),
  },
  'github_update_branch',
  60000,
);

registerGitHubTool(
  'github_get_comments',
  'Get all comments on a GitHub issue or pull request. Use this to check for user approval after posting a plan.',
  {
    owner: z.string().describe('Repository owner'),
    repo: z.string().describe('Repository name'),
    issue_number: z.number().describe('Issue or PR number'),
  },
  'github_get_comments',
);

// ─── Start the stdio transport ───────────────────────────────────────────────

const transport = new StdioServerTransport();
await server.connect(transport);
