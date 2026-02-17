/**
 * IPC-based MCP Server for NanoClaw
 * Writes messages and tasks to files for the host process to pick up
 */

import { createSdkMcpServer, tool } from '@anthropic-ai/claude-agent-sdk';
import { z } from 'zod';
import crypto from 'crypto';
import fs from 'fs';
import path from 'path';
import { CronExpressionParser } from 'cron-parser';
import { search as memorySearch } from './memory-search.js';

const IPC_DIR = '/workspace/ipc';
const MESSAGES_DIR = path.join(IPC_DIR, 'messages');
const TASKS_DIR = path.join(IPC_DIR, 'tasks');

export interface IpcMcpContext {
  chatJid: string;
  groupFolder: string;
  isMain: boolean;
}

function writeIpcFile(dir: string, data: any): string {
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

  return requestId; // Return ID for reply matching
}

async function waitForReply(requestId: string, timeoutMs: number): Promise<any> {
  const REPLIES_DIR = path.join('/workspace/ipc', 'replies');
  const replyPath = path.join(REPLIES_DIR, `${requestId}.json`);
  const startTime = Date.now();

  while (Date.now() - startTime < timeoutMs) {
    try {
      if (fs.existsSync(replyPath)) {
        const reply = JSON.parse(fs.readFileSync(replyPath, 'utf-8'));
        fs.unlinkSync(replyPath); // Clean up
        return reply;
      }
    } catch (err) {
      // File might be mid-write, continue polling
    }
    await new Promise(resolve => setTimeout(resolve, 500)); // Poll every 500ms
  }

  throw new Error(`Timeout waiting for reply after ${timeoutMs}ms`);
}

export function createIpcMcp(ctx: IpcMcpContext) {
  const { chatJid, groupFolder, isMain } = ctx;

  return createSdkMcpServer({
    name: 'nanoclaw',
    version: '1.0.0',
    tools: [
      tool(
        'send_message',
        'Send a message to the user or group. The message is delivered immediately while you\'re still running. You can call this multiple times to send multiple messages.',
        {
          text: z.string().describe('The message text to send')
        },
        async (args: { text: string }) => {
          const data = {
            type: 'message',
            chatJid,
            text: args.text,
            groupFolder,
            timestamp: new Date().toISOString()
          };

          writeIpcFile(MESSAGES_DIR, data);

          return {
            content: [{
              type: 'text',
              text: 'Message sent.'
            }]
          };
        }
      ),

      tool(
        'send_voice_message',
        'Send a voice message (audio) to the user or group. The text will be converted to speech using OpenAI TTS. Use this when the user explicitly asks for a voice response.',
        {
          text: z.string().describe('The text to convert to speech and send as a voice message'),
          voice: z.enum(['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer']).default('nova').describe('Voice to use for TTS. nova is warm and friendly (default), alloy is neutral, echo is deep, fable is expressive, onyx is authoritative, shimmer is bright')
        },
        async (args: { text: string; voice: string }) => {
          const data = {
            type: 'voice_message',
            chatJid,
            text: args.text,
            voice: args.voice,
            groupFolder,
            timestamp: new Date().toISOString()
          };

          writeIpcFile(MESSAGES_DIR, data);

          return {
            content: [{
              type: 'text',
              text: 'Voice message queued for delivery.'
            }]
          };
        }
      ),

      tool(
        'send_image',
        'Send an image to the user or group. Provide either a local file path or a base64-encoded image string. IMPORTANT: image_path must be under /workspace/group/ or /workspace/project/ — these are the only paths accessible to the host. Files in /tmp/ or other container-only paths will NOT work. Save images to /workspace/group/ first if needed.',
        {
          caption: z.string().optional().describe('Optional caption/text to send with the image'),
          image_path: z.string().optional().describe('Absolute path under /workspace/group/ or /workspace/project/ (e.g., "/workspace/group/output.png"). Paths outside these directories are NOT accessible to the host and will fail.'),
          image_base64: z.string().optional().describe('Base64-encoded image data (without data:image/png;base64, prefix)'),
          filename: z.string().optional().describe('Filename for the image (e.g., "chart.png"). Required if using image_base64.')
        },
        async (args: { caption?: string; image_path?: string; image_base64?: string; filename?: string }) => {
          // Validate that exactly one image source is provided
          if (!args.image_path && !args.image_base64) {
            return {
              content: [{
                type: 'text',
                text: 'Error: Must provide either image_path or image_base64'
              }],
              isError: true
            };
          }

          if (args.image_path && args.image_base64) {
            return {
              content: [{
                type: 'text',
                text: 'Error: Cannot provide both image_path and image_base64'
              }],
              isError: true
            };
          }

          if (args.image_base64 && !args.filename) {
            return {
              content: [{
                type: 'text',
                text: 'Error: filename is required when using image_base64'
              }],
              isError: true
            };
          }

          // Validate image_path is under allowed directories
          if (args.image_path) {
            const resolved = path.resolve(args.image_path);
            const allowedPrefixes = ['/workspace/group/', '/workspace/project/'];
            if (!allowedPrefixes.some(prefix => resolved.startsWith(prefix))) {
              return {
                content: [{
                  type: 'text',
                  text: 'Error: image_path must be under /workspace/group/ or /workspace/project/'
                }],
                isError: true
              };
            }
          }

          const data = {
            type: 'image',
            chatJid,
            caption: args.caption,
            imagePath: args.image_path,
            imageBase64: args.image_base64,
            filename: args.filename,
            groupFolder,
            timestamp: new Date().toISOString()
          };

          writeIpcFile(MESSAGES_DIR, data);

          return {
            content: [{
              type: 'text',
              text: 'Image queued for delivery.'
            }]
          };
        }
      ),

      tool(
        'schedule_task',
        `Schedule a recurring or one-time task. The task will run as a full agent with access to all tools.

CONTEXT MODE - Choose based on task type:
• "group": Task runs in the group's conversation context, with access to chat history. Use for tasks that need context about ongoing discussions, user preferences, or recent interactions.
• "isolated": Task runs in a fresh session with no conversation history. Use for independent tasks that don't need prior context. When using isolated mode, include all necessary context in the prompt itself.

If unsure which mode to use, you can ask the user. Examples:
- "Remind me about our discussion" → group (needs conversation context)
- "Check the weather every morning" → isolated (self-contained task)
- "Follow up on my request" → group (needs to know what was requested)
- "Generate a daily report" → isolated (just needs instructions in prompt)

SCHEDULE VALUE FORMAT (all times are LOCAL timezone):
• cron: Standard cron expression (e.g., "*/5 * * * *" for every 5 minutes, "0 9 * * *" for daily at 9am LOCAL time)
• interval: Milliseconds between runs (e.g., "300000" for 5 minutes, "3600000" for 1 hour)
• once: Local time WITHOUT "Z" suffix (e.g., "2026-02-01T15:30:00"). Do NOT use UTC/Z suffix.`,
        {
          prompt: z.string().describe('What the agent should do when the task runs. For isolated mode, include all necessary context here.'),
          schedule_type: z.enum(['cron', 'interval', 'once']).describe('cron=recurring at specific times, interval=recurring every N ms, once=run once at specific time'),
          schedule_value: z.string().describe('cron: "*/5 * * * *" | interval: milliseconds like "300000" | once: local timestamp like "2026-02-01T15:30:00" (no Z suffix!)'),
          context_mode: z.enum(['group', 'isolated']).default('group').describe('group=runs with chat history and memory, isolated=fresh session (include context in prompt)'),
          ...(isMain ? { target_group_jid: z.string().optional().describe('JID of the group to schedule the task for. The group must be registered — look up JIDs in /workspace/project/data/registered_groups.json (the keys are JIDs). If the group is not registered, let the user know and ask if they want to activate it. Defaults to the current group.') } : {}),
        },
        async (args: { prompt: string; schedule_type: string; schedule_value: string; context_mode?: string; target_group_jid?: string }) => {
          // Validate schedule_value before writing IPC
          if (args.schedule_type === 'cron') {
            try {
              CronExpressionParser.parse(args.schedule_value);
            } catch (err) {
              return {
                content: [{ type: 'text', text: `Invalid cron: "${args.schedule_value}". Use format like "0 9 * * *" (daily 9am) or "*/5 * * * *" (every 5 min).` }],
                isError: true
              };
            }
          } else if (args.schedule_type === 'interval') {
            const ms = parseInt(args.schedule_value, 10);
            if (isNaN(ms) || ms <= 0) {
              return {
                content: [{ type: 'text', text: `Invalid interval: "${args.schedule_value}". Must be positive milliseconds (e.g., "300000" for 5 min).` }],
                isError: true
              };
            }
          } else if (args.schedule_type === 'once') {
            const date = new Date(args.schedule_value);
            if (isNaN(date.getTime())) {
              return {
                content: [{ type: 'text', text: `Invalid timestamp: "${args.schedule_value}". Use ISO 8601 format like "2026-02-01T15:30:00.000Z".` }],
                isError: true
              };
            }
          }

          // Non-main groups can only schedule for themselves
          const targetJid = isMain && args.target_group_jid ? args.target_group_jid : chatJid;

          const data = {
            type: 'schedule_task',
            prompt: args.prompt,
            schedule_type: args.schedule_type,
            schedule_value: args.schedule_value,
            context_mode: args.context_mode || 'group',
            targetJid,
            createdBy: groupFolder,
            timestamp: new Date().toISOString()
          };

          const filename = writeIpcFile(TASKS_DIR, data);

          return {
            content: [{
              type: 'text',
              text: `Task scheduled (${filename}): ${args.schedule_type} - ${args.schedule_value}`
            }]
          };
        }
      ),

      // Reads from current_tasks.json which host keeps updated
      tool(
        'list_tasks',
        'List all scheduled tasks. From main: shows all tasks. From other groups: shows only that group\'s tasks.',
        {},
        async () => {
          const tasksFile = path.join(IPC_DIR, 'current_tasks.json');

          try {
            if (!fs.existsSync(tasksFile)) {
              return {
                content: [{
                  type: 'text',
                  text: 'No scheduled tasks found.'
                }]
              };
            }

            const allTasks = JSON.parse(fs.readFileSync(tasksFile, 'utf-8'));

            const tasks = isMain
              ? allTasks
              : allTasks.filter((t: { groupFolder: string }) => t.groupFolder === groupFolder);

            if (tasks.length === 0) {
              return {
                content: [{
                  type: 'text',
                  text: 'No scheduled tasks found.'
                }]
              };
            }

            const formatted = tasks.map((t: { id: string; prompt: string; schedule_type: string; schedule_value: string; status: string; next_run: string }) =>
              `- [${t.id}] ${t.prompt.slice(0, 50)}... (${t.schedule_type}: ${t.schedule_value}) - ${t.status}, next: ${t.next_run || 'N/A'}`
            ).join('\n');

            return {
              content: [{
                type: 'text',
                text: `Scheduled tasks:\n${formatted}`
              }]
            };
          } catch (err) {
            return {
              content: [{
                type: 'text',
                text: `Error reading tasks: ${err instanceof Error ? err.message : String(err)}`
              }]
            };
          }
        }
      ),

      tool(
        'pause_task',
        'Pause a scheduled task. It will not run until resumed.',
        {
          task_id: z.string().describe('The task ID to pause')
        },
        async (args: { task_id: string }) => {
          const data = {
            type: 'pause_task',
            taskId: args.task_id,
            groupFolder,
            isMain,
            timestamp: new Date().toISOString()
          };

          writeIpcFile(TASKS_DIR, data);

          return {
            content: [{
              type: 'text',
              text: `Task ${args.task_id} pause requested.`
            }]
          };
        }
      ),

      tool(
        'resume_task',
        'Resume a paused task.',
        {
          task_id: z.string().describe('The task ID to resume')
        },
        async (args: { task_id: string }) => {
          const data = {
            type: 'resume_task',
            taskId: args.task_id,
            groupFolder,
            isMain,
            timestamp: new Date().toISOString()
          };

          writeIpcFile(TASKS_DIR, data);

          return {
            content: [{
              type: 'text',
              text: `Task ${args.task_id} resume requested.`
            }]
          };
        }
      ),

      tool(
        'cancel_task',
        'Cancel and delete a scheduled task.',
        {
          task_id: z.string().describe('The task ID to cancel')
        },
        async (args: { task_id: string }) => {
          const data = {
            type: 'cancel_task',
            taskId: args.task_id,
            groupFolder,
            isMain,
            timestamp: new Date().toISOString()
          };

          writeIpcFile(TASKS_DIR, data);

          return {
            content: [{
              type: 'text',
              text: `Task ${args.task_id} cancellation requested.`
            }]
          };
        }
      ),

      tool(
        'register_group',
        `Register a new WhatsApp group so the agent can respond to messages there. Main group only.

Use available_groups.json to find the JID for a group. The folder name should be lowercase with hyphens (e.g., "family-chat").`,
        {
          jid: z.string().describe('The WhatsApp JID (e.g., "120363336345536173@g.us")'),
          name: z.string().describe('Display name for the group'),
          folder: z.string().describe('Folder name for group files (lowercase, hyphens, e.g., "family-chat")'),
          trigger: z.string().describe('Trigger word (e.g., "@Bagel")')
        },
        async (args: { jid: string; name: string; folder: string; trigger: string }) => {
          if (!isMain) {
            return {
              content: [{ type: 'text', text: 'Only the main group can register new groups.' }],
              isError: true
            };
          }

          const data = {
            type: 'register_group',
            jid: args.jid,
            name: args.name,
            folder: args.folder,
            trigger: args.trigger,
            timestamp: new Date().toISOString()
          };

          writeIpcFile(TASKS_DIR, data);

          return {
            content: [{
              type: 'text',
              text: `Group "${args.name}" registered. It will start receiving messages immediately.`
            }]
          };
        }
      ),

      ...(isMain ? [tool(
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
• "host" — changes to src/*.ts (host process code)
• "container" — changes to container/agent-runner/ (agent code, MCP tools, Dockerfile)
• Both — changes spanning host and container code
• Neither — changes to CLAUDE.md, docs, or non-code files (just commits, no rebuild/restart)

After deploy, the service restarts automatically. Your session ends — the user will see the service come back online.`,
        {
          targets: z.array(z.enum(['host', 'container'])).describe('What to rebuild. Empty array = commit only (no rebuild/restart).'),
          commit_message: z.string().optional().describe('Git commit message describing your changes. Defaults to "chore: agent-initiated deploy".')
        },
        async (args: { targets: Array<'host' | 'container'>; commit_message?: string }) => {
          if (!isMain) {
            return {
              content: [{ type: 'text', text: 'Deploy is only available from the main group.' }],
              isError: true
            };
          }

          const data = {
            type: 'deploy',
            targets: args.targets,
            commitMessage: args.commit_message,
            chatJid,
            groupFolder,
            timestamp: new Date().toISOString()
          };

          const requestId = writeIpcFile(TASKS_DIR, data);

          try {
            // Wait for deploy completion (5 minutes for builds)
            const reply = await waitForReply(requestId, 300000);

            if (reply.status === 'success') {
              // If container was rebuilt, schedule container restart
              if (args.targets.includes('container')) {
                setTimeout(() => {
                  const restartData = {
                    type: 'restart_container',
                    groupFolder: 'main',
                    chatJid,
                    timestamp: new Date().toISOString()
                  };
                  writeIpcFile(TASKS_DIR, restartData);
                }, 2000);
              }

              return {
                content: [{
                  type: 'text',
                  text: `✅ Deploy succeeded!\n\n${reply.message || ''}\n\nTargets: ${args.targets.join(', ') || 'none (commit only)'}${args.targets.includes('container') ? '\n\nContainer will restart in 2 seconds to load new code.' : ''}`
                }]
              };
            } else {
              return {
                content: [{
                  type: 'text',
                  text: `❌ Deploy failed: ${reply.error}\n\nThe host has already notified the user with details. Check your changes and try again.`
                }],
                isError: true
              };
            }
          } catch (err) {
            return {
              content: [{
                type: 'text',
                text: `⏱️ Deploy timeout or error: ${err instanceof Error ? err.message : String(err)}\n\nThe host may still be processing (builds take time). Check host logs or ask the user.`
              }],
              isError: true
            };
          }
        }
      )] : []),

      // Test container build tool (main group only)
      ...(isMain ? [
        tool(
          'test_container_build',
          'Test the container build script to verify it works. Runs ./container/build.sh on the host and returns the output. Use this after modifying build.sh or Dockerfile to ensure builds work before deploying.',
          {},
          async () => {
            const data = {
              type: 'test_container_build',
              timestamp: new Date().toISOString()
            };

            writeIpcFile(TASKS_DIR, data);

            return {
              content: [{
                type: 'text',
                text: 'Container build test initiated. The host will run ./container/build.sh and return the results via a message.'
              }]
            };
          }
        )
      ] : []),

      // Restart container tool (main group only)
      ...(isMain ? [
        tool(
          'restart_container',
          'Restart the persistent container to pick up new code changes. Use this after deploying container changes to ensure the new code is loaded immediately instead of waiting for idle timeout.',
          {
            groupFolder: z.string().describe('The group folder name to restart (typically "main")')
          },
          async (args: { groupFolder: string }) => {
            const data = {
              type: 'restart_container',
              groupFolder: args.groupFolder,
              timestamp: new Date().toISOString()
            };

            writeIpcFile(TASKS_DIR, data);

            return {
              content: [{
                type: 'text',
                text: `Container restart requested for group: ${args.groupFolder}. The container will shut down and the next query will spawn a fresh container with new code.`
              }]
            };
          }
        )
      ] : []),

      tool(
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
          query: z.string().describe('Natural language search query — describe what you\'re looking for'),
          mode: z.enum(['hybrid', 'semantic', 'keyword']).default('hybrid').describe('hybrid (recommended): combines meaning + keywords. semantic: meaning only. keyword: exact word matches only.'),
          limit: z.number().int().min(1).max(20).default(5).describe('Number of results to return (1-20)')
        },
        async (args: { query: string; mode?: 'hybrid' | 'semantic' | 'keyword'; limit?: number }) => {
          try {
            // Main group searches all memories; non-main groups only search their own
            const results = await memorySearch(args.query, args.mode, args.limit, isMain ? undefined : groupFolder);

            if (results.length === 0) {
              return {
                content: [{
                  type: 'text',
                  text: 'No matching memories found.'
                }]
              };
            }

            const formatted = results.map((r, i) =>
              `[${i + 1}] Source: ${r.source} (${r.type})\n${r.content}`
            ).join('\n\n---\n\n');

            return {
              content: [{
                type: 'text',
                text: `Found ${results.length} relevant memories:\n\n${formatted}`
              }]
            };
          } catch (err) {
            return {
              content: [{
                type: 'text',
                text: `Memory search error: ${err instanceof Error ? err.message : String(err)}`
              }],
              isError: true
            };
          }
        }
      )
    ]
  });
}
