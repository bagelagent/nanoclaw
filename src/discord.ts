import {
  AttachmentBuilder,
  Client,
  Events,
  GatewayIntentBits,
  Message,
  Partials,
  TextChannel,
} from 'discord.js';
import fs from 'fs';
import path from 'path';
import { tmpdir } from 'os';

import { DISCORD_BOT_TOKEN } from './config.js';
import {
  storeChatMetadata,
  storeGenericMessage,
  getAllRegisteredGroups,
} from './db.js';
import { logger } from './logger.js';
import { initOpenAI } from './audio.js';

const DISCORD_API_TIMEOUT = 15000;

function withTimeout<T>(
  promise: Promise<T>,
  ms = DISCORD_API_TIMEOUT,
): Promise<T> {
  return Promise.race([
    promise,
    new Promise<never>((_, reject) =>
      setTimeout(
        () => reject(new Error(`Discord API call timed out after ${ms}ms`)),
        ms,
      ),
    ),
  ]);
}

let client: Client;

export interface DiscordCallbacks {
  onMessage: (chatJid: string, hasRegisteredGroup: boolean) => void;
}

/**
 * Convert a Discord channel/DM to a NanoClaw JID.
 */
function toJid(msg: Message): string {
  if (msg.guild) {
    return `discord:${msg.channelId}`;
  }
  return `discord:dm:${msg.author.id}`;
}

/**
 * Check if an attachment is an audio file
 */
function isAudioAttachment(
  contentType: string | null,
  name: string | null,
): boolean {
  if (contentType?.startsWith('audio/')) return true;
  if (
    name?.endsWith('.ogg') ||
    name?.endsWith('.mp3') ||
    name?.endsWith('.wav')
  )
    return true;
  return false;
}

/**
 * Transcribe an audio attachment from Discord
 */
async function transcribeDiscordAudio(
  url: string,
  filename: string,
): Promise<string | null> {
  try {
    // Download the audio file
    const response = await fetch(url);
    if (!response.ok) {
      logger.error(
        { url, status: response.status },
        'Failed to download Discord audio',
      );
      return null;
    }

    const buffer = Buffer.from(await response.arrayBuffer());
    if (!buffer || buffer.length === 0) {
      logger.warn('Empty audio buffer from Discord');
      return null;
    }

    // Save to temporary file
    const tempDir = path.join(tmpdir(), 'nanoclaw-discord-audio');
    fs.mkdirSync(tempDir, { recursive: true });
    const tempFilePath = path.join(tempDir, `${Date.now()}-${filename}`);
    fs.writeFileSync(tempFilePath, buffer);

    try {
      // Import OpenAI dynamically to get the client
      const { default: OpenAI } = await import('openai');
      const apiKey = process.env.OPENAI_API_KEY;
      if (!apiKey) {
        logger.warn(
          'OpenAI API key not set, skipping Discord audio transcription',
        );
        return null;
      }

      const openai = new OpenAI({ apiKey });

      // Transcribe with Whisper
      const transcription = await openai.audio.transcriptions.create({
        file: fs.createReadStream(tempFilePath),
        model: 'whisper-1',
      });

      logger.info(
        { text: transcription.text, filename },
        'Discord audio transcribed',
      );
      return transcription.text;
    } finally {
      // Clean up temp file
      if (fs.existsSync(tempFilePath)) {
        fs.unlinkSync(tempFilePath);
      }
    }
  } catch (error) {
    logger.error({ error, url }, 'Discord audio transcription failed');
    return null;
  }
}

/**
 * Build attachment description lines for a Discord message.
 * Returns URLs with content type hints so the agent can fetch/view them.
 * Audio files are transcribed automatically.
 */
async function buildAttachmentLines(msg: Message): Promise<string> {
  if (msg.attachments.size === 0) return '';

  const lines: string[] = [];

  for (const a of msg.attachments.values()) {
    const contentType = a.contentType;
    const name = a.name || 'unknown';

    // Check if it's an audio file
    if (isAudioAttachment(contentType, name)) {
      const transcription = await transcribeDiscordAudio(a.url, name);
      if (transcription) {
        lines.push(`[Voice message transcription]: ${transcription}`);
      } else {
        lines.push(`[Attached audio: ${name}] ${a.url}`);
      }
    } else {
      const type = contentType?.startsWith('image/') ? 'image' : 'file';
      lines.push(`[Attached ${type}: ${name}] ${a.url}`);
    }
  }

  return lines.join('\n');
}

/**
 * Build full content string from a Discord message (text + attachments).
 */
async function buildContent(msg: Message): Promise<string> {
  const attachmentLines = await buildAttachmentLines(msg);
  if (!attachmentLines) return msg.content;
  return msg.content ? `${msg.content}\n${attachmentLines}` : attachmentLines;
}

/**
 * Connect to Discord and start listening for messages.
 */
export async function connectDiscord(
  callbacks: DiscordCallbacks,
): Promise<void> {
  client = new Client({
    intents: [
      GatewayIntentBits.Guilds,
      GatewayIntentBits.GuildMessages,
      GatewayIntentBits.DirectMessages,
      GatewayIntentBits.MessageContent,
    ],
    partials: [Partials.Channel],
  });

  client.once(Events.ClientReady, (c) => {
    logger.info({ user: c.user.tag }, 'Connected to Discord');
  });

  client.on(Events.MessageCreate, async (msg) => {
    // Ignore bot messages (including our own)
    if (msg.author.bot) return;

    const chatJid = toJid(msg);
    const timestamp = new Date(msg.createdTimestamp).toISOString();
    const registeredGroups = getAllRegisteredGroups();
    const isRegistered = !!registeredGroups[chatJid];

    // For guild messages, auto-respond in any channel the bot is in
    if (msg.guild) {
      // Only respond to specific user (owner) in guild channels
      const OWNER_DISCORD_ID = '237014586480525313';
      if (msg.author.id !== OWNER_DISCORD_ID) {
        // Store message for context but don't process
        const content = await buildContent(msg);
        storeGenericMessage(
          msg.id,
          chatJid,
          msg.author.id,
          msg.author.displayName || msg.author.username,
          content,
          timestamp,
          false,
        );
        return;
      }

      // Always store chat metadata for discovery
      const channelName = (msg.channel as TextChannel).name || chatJid;
      storeChatMetadata(
        chatJid,
        timestamp,
        `#${channelName} (${msg.guild.name})`,
      );

      // Process all messages in guild channels without needing @mention
      // Each channel will be treated as its own group with separate memory
      const content = await buildContent(msg);
      storeGenericMessage(
        msg.id,
        chatJid,
        msg.author.id,
        msg.author.displayName || msg.author.username,
        content,
        timestamp,
        false,
      );

      // Trigger agent processing
      // Even if not registered, we'll process it (auto-registration will happen on agent side)
      callbacks.onMessage(chatJid, isRegistered);
      return;
    }

    // DMs: always store metadata and process
    if (!msg.guild) {
      storeChatMetadata(
        chatJid,
        timestamp,
        `DM: ${msg.author.displayName || msg.author.username}`,
      );
    }

    // Store full message content for registered groups
    if (isRegistered) {
      // Strip the bot mention from content for cleaner processing
      let content = msg.content;
      if (client.user) {
        content = content
          .replace(new RegExp(`<@!?${client.user.id}>`, 'g'), '')
          .trim();
      }

      // Append attachment URLs (images, files, etc.)
      const attachmentLines = await buildAttachmentLines(msg);
      if (attachmentLines) {
        content = content ? `${content}\n${attachmentLines}` : attachmentLines;
      }

      storeGenericMessage(
        msg.id,
        chatJid,
        msg.author.id,
        msg.author.displayName || msg.author.username,
        content,
        timestamp,
        false,
      );
    }

    callbacks.onMessage(chatJid, isRegistered);
  });

  await client.login(DISCORD_BOT_TOKEN);
}

/**
 * Send a message to a Discord channel or DM.
 */
// Active typing indicator intervals per JID
const typingIntervals = new Map<string, NodeJS.Timeout>();

/**
 * Show or hide the "Bot is typing..." indicator for a Discord channel/DM.
 * sendTyping() lasts ~10s, so we repeat every 9s while active.
 */
export async function setDiscordTyping(
  jid: string,
  isTyping: boolean,
): Promise<void> {
  if (!isTyping) {
    const interval = typingIntervals.get(jid);
    if (interval) {
      clearInterval(interval);
      typingIntervals.delete(jid);
    }
    return;
  }

  // Already typing for this channel
  if (typingIntervals.has(jid)) return;

  if (!client?.isReady()) return;

  try {
    let channelId: string;
    if (jid.startsWith('discord:dm:')) {
      const userId = jid.replace('discord:dm:', '');
      const user = await withTimeout(client.users.fetch(userId));
      const dmChannel = await withTimeout(user.createDM());
      channelId = dmChannel.id;
    } else {
      channelId = jid.replace('discord:', '');
    }

    const channel = await withTimeout(client.channels.fetch(channelId));
    if (!channel || !('sendTyping' in channel)) return;

    const textChannel = channel as TextChannel;
    // Fire immediately, then every 9s
    await withTimeout(textChannel.sendTyping());
    const interval = setInterval(async () => {
      try {
        await textChannel.sendTyping();
      } catch {
        // Channel gone or permissions changed — clean up
        clearInterval(interval);
        typingIntervals.delete(jid);
      }
    }, 9000);
    typingIntervals.set(jid, interval);
  } catch (err) {
    logger.debug({ jid, err }, 'Failed to set Discord typing indicator');
  }
}

/**
 * Send a message to a Discord channel or DM.
 */
export async function sendDiscordMessage(
  jid: string,
  text: string,
): Promise<void> {
  if (!client?.isReady()) {
    logger.error({ jid }, 'Discord client not ready, cannot send message');
    return;
  }

  try {
    let channelId: string;
    if (jid.startsWith('discord:dm:')) {
      // DM: fetch user and create/get DM channel
      const userId = jid.replace('discord:dm:', '');
      const user = await withTimeout(client.users.fetch(userId));
      const dmChannel = await withTimeout(user.createDM());
      channelId = dmChannel.id;
    } else {
      // Guild channel: extract channel ID
      channelId = jid.replace('discord:', '');
    }

    const channel = await withTimeout(client.channels.fetch(channelId));
    if (channel && 'send' in channel) {
      // Discord has a 2000 char limit per message
      if (text.length <= 2000) {
        await withTimeout((channel as TextChannel).send(text));
      } else {
        // Split long messages
        for (let i = 0; i < text.length; i += 2000) {
          await withTimeout(
            (channel as TextChannel).send(text.slice(i, i + 2000)),
          );
        }
      }
      logger.info({ jid, length: text.length }, 'Discord message sent');
      // Stop typing indicator — response has been delivered
      setDiscordTyping(jid, false);
    } else {
      logger.error({ jid, channelId }, 'Channel not found or not text-based');
    }
  } catch (err) {
    logger.error({ jid, err }, 'Failed to send Discord message');
  }
}

/**
 * Send a voice message (audio file) to Discord
 */
export async function sendDiscordVoiceMessage(
  jid: string,
  audioBuffer: Buffer,
): Promise<void> {
  if (!client?.isReady()) {
    logger.error(
      { jid },
      'Discord client not ready, cannot send voice message',
    );
    return;
  }

  try {
    let channelId: string;
    if (jid.startsWith('discord:dm:')) {
      // DM: fetch user and create/get DM channel
      const userId = jid.replace('discord:dm:', '');
      const user = await withTimeout(client.users.fetch(userId));
      const dmChannel = await withTimeout(user.createDM());
      channelId = dmChannel.id;
    } else {
      // Guild channel: extract channel ID
      channelId = jid.replace('discord:', '');
    }

    const channel = await withTimeout(client.channels.fetch(channelId));
    if (channel && 'send' in channel) {
      const attachment = new AttachmentBuilder(audioBuffer, {
        name: 'voice-message.ogg',
      });
      await withTimeout((channel as TextChannel).send({ files: [attachment] }));
      logger.info(
        { jid, size: audioBuffer.length },
        'Discord voice message sent',
      );
    } else {
      logger.error({ jid }, 'Channel not found or not a text channel');
    }
  } catch (err) {
    logger.error({ jid, err }, 'Failed to send Discord voice message');
    throw err;
  }
}

/**
 * Send an image to Discord
 */
export async function sendDiscordImage(
  jid: string,
  imageBuffer: Buffer,
  filename: string,
  caption?: string,
): Promise<void> {
  if (!client?.isReady()) {
    logger.error({ jid }, 'Discord client not ready, cannot send image');
    return;
  }

  try {
    let channelId: string;
    if (jid.startsWith('discord:dm:')) {
      // DM: fetch user and create/get DM channel
      const userId = jid.replace('discord:dm:', '');
      const user = await withTimeout(client.users.fetch(userId));
      const dmChannel = await withTimeout(user.createDM());
      channelId = dmChannel.id;
    } else {
      // Guild channel: extract channel ID
      channelId = jid.replace('discord:', '');
    }

    const channel = await withTimeout(client.channels.fetch(channelId));
    if (channel && 'send' in channel) {
      const attachment = new AttachmentBuilder(imageBuffer, {
        name: filename,
      });

      const messageOptions: any = { files: [attachment] };
      if (caption) {
        messageOptions.content = caption;
      }

      await withTimeout((channel as TextChannel).send(messageOptions));
      logger.info(
        { jid, size: imageBuffer.length, filename },
        'Discord image sent',
      );
    } else {
      logger.error({ jid }, 'Channel not found or not a text channel');
    }
  } catch (err) {
    logger.error({ jid, err }, 'Failed to send Discord image');
    throw err;
  }
}
