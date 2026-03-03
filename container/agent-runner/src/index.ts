/**
 * NanoClaw Agent Runner
 * Runs inside a container, receives config via stdin, outputs result to stdout
 *
 * Protocol:
 *   Stdin: Line-delimited JSON — one ContainerInput per line.
 *          Container stays alive between queries (persistent container).
 *          Shutdown: {"type":"shutdown"} or EOF.
 *   Stdout: Each result is wrapped in OUTPUT_START_MARKER / OUTPUT_END_MARKER pairs.
 */

import fs from 'fs';
import path from 'path';
import { createInterface } from 'readline';
import { query, HookCallback, PreCompactHookInput, PreToolUseHookInput } from '@anthropic-ai/claude-agent-sdk';
import { createIpcMcp } from './ipc-mcp.js';

interface ContainerInput {
  prompt: string;
  sessionId?: string;
  groupFolder: string;
  chatJid: string;
  isMain: boolean;
  isScheduledTask?: boolean;
  model?: string;
  assistantName?: string;
}

interface AgentResponse {
  outputType: 'message' | 'log';
  userMessage?: string;
  internalLog?: string;
}

const AGENT_RESPONSE_SCHEMA = {
  type: 'object',
  properties: {
    outputType: {
      type: 'string',
      enum: ['message', 'log'],
      description: '"message": the userMessage field contains a message to send to the user or group. "log": the output will not be sent to the user or group.',
    },
    userMessage: {
      type: 'string',
      description: 'A message to send to the user or group. Include when outputType is "message".',
    },
    internalLog: {
      type: 'string',
      description: 'Information that will be logged internally but not sent to the user or group.',
    },
  },
  required: ['outputType'],
} as const;

interface ContainerOutput {
  status: 'success' | 'error';
  result: AgentResponse | null;
  newSessionId?: string;
  error?: string;
}

interface SessionEntry {
  sessionId: string;
  fullPath: string;
  summary: string;
  firstPrompt: string;
}

interface SessionsIndex {
  entries: SessionEntry[];
}

/**
 * Yields one line at a time from stdin. Resolves the async iterator when stdin closes (EOF).
 */
function stdinLines(): AsyncIterable<string> {
  const rl = createInterface({ input: process.stdin, crlfDelay: Infinity });
  return rl;
}

const OUTPUT_START_MARKER = '---NANOCLAW_OUTPUT_START---';
const OUTPUT_END_MARKER = '---NANOCLAW_OUTPUT_END---';

function writeOutput(output: ContainerOutput): void {
  console.log(OUTPUT_START_MARKER);
  console.log(JSON.stringify(output));
  console.log(OUTPUT_END_MARKER);
}

function log(message: string): void {
  console.error(`[agent-runner] ${message}`);
}

function getSessionSummary(sessionId: string, transcriptPath: string): string | null {
  const projectDir = path.dirname(transcriptPath);
  const indexPath = path.join(projectDir, 'sessions-index.json');

  if (!fs.existsSync(indexPath)) {
    log(`Sessions index not found at ${indexPath}`);
    return null;
  }

  try {
    const index: SessionsIndex = JSON.parse(fs.readFileSync(indexPath, 'utf-8'));
    const entry = index.entries.find(e => e.sessionId === sessionId);
    if (entry?.summary) {
      return entry.summary;
    }
  } catch (err) {
    log(`Failed to read sessions index: ${err instanceof Error ? err.message : String(err)}`);
  }

  return null;
}

/**
 * Archive the full transcript to conversations/ before compaction.
 */
function createPreCompactHook(assistantName?: string): HookCallback {
  return async (input: any, _toolUseId: any, _context: any) => {
    const preCompact = input as PreCompactHookInput;
    const transcriptPath = preCompact.transcript_path;
    const sessionId = preCompact.session_id;

    if (!transcriptPath || !fs.existsSync(transcriptPath)) {
      log('No transcript found for archiving');
      return {};
    }

    try {
      const content = fs.readFileSync(transcriptPath, 'utf-8');
      const messages = parseTranscript(content);

      if (messages.length === 0) {
        log('No messages to archive');
        return {};
      }

      const summary = getSessionSummary(sessionId, transcriptPath);
      const name = summary ? sanitizeFilename(summary) : generateFallbackName();

      const conversationsDir = '/workspace/group/conversations';
      fs.mkdirSync(conversationsDir, { recursive: true });

      const date = new Date().toISOString().split('T')[0];
      const filename = `${date}-${name}.md`;
      const filePath = path.join(conversationsDir, filename);

      const markdown = formatTranscriptMarkdown(messages, summary, assistantName);
      fs.writeFileSync(filePath, markdown);

      log(`Archived conversation to ${filePath}`);
    } catch (err) {
      log(`Failed to archive transcript: ${err instanceof Error ? err.message : String(err)}`);
    }

    return {};
  };
}

// Secrets to strip from Bash tool subprocess environments.
// These are needed by claude-code for API auth but should never
// be visible to commands the agent runs.
const SECRET_ENV_VARS = ['ANTHROPIC_API_KEY', 'CLAUDE_CODE_OAUTH_TOKEN'];

function createSanitizeBashHook(): HookCallback {
  return async (input, _toolUseId, _context) => {
    const preInput = input as PreToolUseHookInput;
    const command = (preInput.tool_input as { command?: string })?.command;
    if (!command) return {};

    const unsetPrefix = `unset ${SECRET_ENV_VARS.join(' ')} 2>/dev/null; `;
    return {
      hookSpecificOutput: {
        hookEventName: 'PreToolUse',
        updatedInput: {
          ...(preInput.tool_input as Record<string, unknown>),
          command: unsetPrefix + command,
        },
      },
    };
  };
}

function sanitizeFilename(summary: string): string {
  return summary
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 50);
}

function generateFallbackName(): string {
  const time = new Date();
  return `conversation-${time.getHours().toString().padStart(2, '0')}${time.getMinutes().toString().padStart(2, '0')}`;
}

interface ParsedMessage {
  role: 'user' | 'assistant';
  content: string;
}

function parseTranscript(content: string): ParsedMessage[] {
  const messages: ParsedMessage[] = [];

  for (const line of content.split('\n')) {
    if (!line.trim()) continue;
    try {
      const entry = JSON.parse(line);
      if (entry.type === 'user' && entry.message?.content) {
        const text = typeof entry.message.content === 'string'
          ? entry.message.content
          : entry.message.content.map((c: { text?: string }) => c.text || '').join('');
        if (text) messages.push({ role: 'user', content: text });
      } else if (entry.type === 'assistant' && entry.message?.content) {
        const textParts = entry.message.content
          .filter((c: { type: string }) => c.type === 'text')
          .map((c: { text: string }) => c.text);
        const text = textParts.join('');
        if (text) messages.push({ role: 'assistant', content: text });
      }
    } catch {
    }
  }

  return messages;
}

function formatTranscriptMarkdown(messages: ParsedMessage[], title?: string | null, assistantName?: string): string {
  const now = new Date();
  const formatDateTime = (d: Date) => d.toLocaleString('en-US', {
    month: 'short',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
    hour12: true
  });

  const lines: string[] = [];
  lines.push(`# ${title || 'Conversation'}`);
  lines.push('');
  lines.push(`Archived: ${formatDateTime(now)}`);
  lines.push('');
  lines.push('---');
  lines.push('');

  for (const msg of messages) {
    const sender = msg.role === 'user' ? 'User' : (assistantName || 'Bagel');
    const content = msg.content.length > 2000
      ? msg.content.slice(0, 2000) + '...'
      : msg.content;
    lines.push(`**${sender}**: ${content}`);
    lines.push('');
  }

  return lines.join('\n');
}

/**
 * Process a single query: run the Claude agent SDK and return the output.
 */
async function processQuery(input: ContainerInput): Promise<ContainerOutput> {
  // Create IPC MCP with the current query's context (chatJid may differ per query)
  const ipcMcp = createIpcMcp({
    chatJid: input.chatJid,
    groupFolder: input.groupFolder,
    isMain: input.isMain
  });

  let newSessionId: string | undefined;

  // Add context for scheduled tasks
  let prompt = input.prompt;
  if (input.isScheduledTask) {
    prompt = `[SCHEDULED TASK - The following message was sent automatically and is not coming directly from the user or group.]\n\n${input.prompt}`;
  }

  // Load global CLAUDE.md as additional system context (shared across all groups)
  const globalClaudeMdPath = '/workspace/global/CLAUDE.md';
  let globalClaudeMd: string | undefined;
  if (!input.isMain && fs.existsSync(globalClaudeMdPath)) {
    globalClaudeMd = fs.readFileSync(globalClaudeMdPath, 'utf-8');
  }

  // Discover additional directories mounted at /workspace/extra/*
  // These are passed to the SDK so their CLAUDE.md files are loaded automatically
  const extraDirs: string[] = [];
  const extraBase = '/workspace/extra';
  if (fs.existsSync(extraBase)) {
    for (const entry of fs.readdirSync(extraBase)) {
      const fullPath = path.join(extraBase, entry);
      if (fs.statSync(fullPath).isDirectory()) {
        extraDirs.push(fullPath);
      }
    }
  }
  if (extraDirs.length > 0) {
    log(`Additional directories: ${extraDirs.join(', ')}`);
  }

  let lastProgressTime = 0;
  function emitProgress(status: string): void {
    // Rate limit: max 1 progress file per second
    const now = Date.now();
    if (now - lastProgressTime < 1000) return;
    lastProgressTime = now;

    // Write progress update to IPC for host to pick up
    const progressDir = '/workspace/ipc/progress';
    try {
      if (!fs.existsSync(progressDir)) {
        fs.mkdirSync(progressDir, { recursive: true });
      }

      const filename = `${now}.json`;
      const filePath = path.join(progressDir, filename);
      const data = JSON.stringify({ chatJid: input.chatJid, status, timestamp: new Date().toISOString() });

      fs.writeFileSync(filePath, data);
    } catch (err) {
      log(`Failed to emit progress: ${err instanceof Error ? err.message : String(err)}`);
    }
  }

  async function runAgent(sessionId: string | undefined): Promise<ContainerOutput> {
    let agentResult: AgentResponse | null = null;
    let agentSessionId: string | undefined;

    emitProgress('🤔 Thinking...');

    for await (const message of query({
      prompt,
      options: {
        ...(input.model ? { model: input.model } : {}),
        cwd: '/workspace/group',
        additionalDirectories: extraDirs.length > 0 ? extraDirs : undefined,
        resume: sessionId,
        systemPrompt: globalClaudeMd
          ? { type: 'preset' as const, preset: 'claude_code' as const, append: globalClaudeMd }
          : undefined,
        allowedTools: [
          'Bash',
          'Read', 'Write', 'Edit', 'Glob', 'Grep',
          'WebSearch', 'WebFetch',
          'Task', 'TaskOutput', 'TaskStop',
          'TeamCreate', 'TeamDelete', 'SendMessage',
          'TodoWrite', 'ToolSearch', 'Skill',
          'NotebookEdit',
          'mcp__nanoclaw__*'
        ],
        permissionMode: 'bypassPermissions',
        allowDangerouslySkipPermissions: true,
        settingSources: ['project', 'user'],
        mcpServers: {
          nanoclaw: ipcMcp
        },
        hooks: {
          PreCompact: [{ hooks: [createPreCompactHook(input.assistantName)] }],
          PreToolUse: [{ matcher: 'Bash', hooks: [createSanitizeBashHook()] }],
        },
        outputFormat: {
          type: 'json_schema',
          schema: AGENT_RESPONSE_SCHEMA,
        }
      }
    })) {
      // Log ALL message types to diagnose progress streaming
      const msgPreview = JSON.stringify(message).slice(0, 300);
      const subtype = (message as any).subtype;
      log(`[SDK_MESSAGE] type="${message.type}" ${subtype ? `subtype="${subtype}" ` : ''}preview: ${msgPreview}`);

      if (message.type === 'system' && (message as any).subtype === 'init') {
        agentSessionId = message.session_id;
        log(`Session initialized: ${agentSessionId}`);
      }

      // Emit progress for tool use
      if (message.type === 'assistant') {
        const msg = message as any;
        const content = msg.message?.content;
        if (Array.isArray(content)) {
          for (const block of content) {
            if (block.type === 'tool_use' && block.name) {
              const toolName = block.name;
              const toolInput = block.input || {};

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
                'mcp__nanoclaw__generate_image': '🎨',
                'Task': '🚀',
              };
              const emoji = toolEmoji[toolName] || '🔧';

              let description = '';
              if (toolInput.description) {
                description = toolInput.description;
              } else if (toolInput.file_path) {
                const fileName = toolInput.file_path.split('/').pop() || toolInput.file_path;
                description = fileName;
              } else if (toolInput.pattern) {
                description = toolInput.pattern;
              } else if (toolInput.query) {
                description = toolInput.query.slice(0, 50);
              } else if (toolInput.url) {
                try {
                  const urlObj = new URL(toolInput.url);
                  description = urlObj.hostname;
                } catch {
                  description = toolInput.url.slice(0, 50);
                }
              } else if (toolInput.command) {
                description = toolInput.command.slice(0, 50);
              }

              if (description.length > 60) {
                description = description.slice(0, 57) + '...';
              }

              const progressMessage = description
                ? `${emoji} ${description}`
                : `${emoji} ${toolName}...`;

              emitProgress(progressMessage);
              break;
            }
          }
        }
      }

      if (message.type === 'result') {
        if (message.subtype === 'success' && message.structured_output) {
          agentResult = message.structured_output as AgentResponse;
          if (agentResult.outputType === 'message' && !agentResult.userMessage) {
            log('Warning: outputType is "message" but userMessage is missing, treating as "log"');
            agentResult = { outputType: 'log', internalLog: agentResult.internalLog };
          }
          log(`Agent result: outputType=${agentResult.outputType}${agentResult.internalLog ? `, log=${agentResult.internalLog}` : ''}`);
        } else if (message.subtype === 'success' || message.subtype === 'error_max_structured_output_retries') {
          log(`Structured output unavailable (subtype=${message.subtype}), falling back to text`);
          const textResult = 'result' in message ? (message as { result?: string }).result : null;
          if (textResult) {
            agentResult = { outputType: 'message', userMessage: textResult };
          }
        }
      }
    }

    return {
      status: 'success',
      result: agentResult ?? { outputType: 'log' },
      newSessionId: agentSessionId
    };
  }

  try {
    log('Starting agent...');
    const output = await runAgent(input.sessionId);
    log('Agent completed successfully');
    newSessionId = output.newSessionId;
    return output;

  } catch (err) {
    const errorMessage = err instanceof Error ? err.message : String(err);

    // If session resume failed, retry without a session (start fresh)
    if (input.sessionId) {
      log(`Agent error with session resume: ${errorMessage}. Retrying without session...`);
      try {
        const output = await runAgent(undefined);
        log('Agent completed successfully (fresh session)');
        newSessionId = output.newSessionId;
        return output;
      } catch (retryErr) {
        const retryMessage = retryErr instanceof Error ? retryErr.message : String(retryErr);
        log(`Agent error on fresh session: ${retryMessage}`);
        return {
          status: 'error',
          result: null,
          newSessionId,
          error: retryMessage
        };
      }
    }

    log(`Agent error: ${errorMessage}`);
    return {
      status: 'error',
      result: null,
      newSessionId,
      error: errorMessage
    };
  }
}

/**
 * Main loop: read line-delimited JSON from stdin, process each query, write output.
 * Container stays alive between queries. Exits on EOF or shutdown message.
 */
async function main(): Promise<void> {
  log('Agent runner ready, waiting for queries on stdin...');

  for await (const line of stdinLines()) {
    const trimmed = line.trim();
    if (!trimmed) continue;

    let input: ContainerInput;
    try {
      const parsed = JSON.parse(trimmed);

      // Shutdown signal — exit cleanly
      if (parsed.type === 'shutdown') {
        log('Received shutdown signal, exiting');
        break;
      }

      input = parsed as ContainerInput;
      log(`Received query for group: ${input.groupFolder}, chatJid: ${input.chatJid}`);
    } catch (err) {
      writeOutput({
        status: 'error',
        result: null,
        error: `Failed to parse input: ${err instanceof Error ? err.message : String(err)}`
      });
      continue; // Don't exit — wait for next valid query
    }

    const output = await processQuery(input);
    writeOutput(output);
  }

  log('Stdin closed, exiting');
}

main();
