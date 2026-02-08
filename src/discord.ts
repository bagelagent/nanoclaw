import {
  Client,
  Events,
  GatewayIntentBits,
  Message,
  Partials,
  TextChannel,
} from 'discord.js';

import { DISCORD_BOT_TOKEN } from './config.js';
import { storeChatMetadata, storeGenericMessage, getAllRegisteredGroups } from './db.js';
import { logger } from './logger.js';

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
 * Connect to Discord and start listening for messages.
 */
export async function connectDiscord(callbacks: DiscordCallbacks): Promise<void> {
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

  client.on(Events.MessageCreate, (msg) => {
    // Ignore bot messages (including our own)
    if (msg.author.bot) return;

    const chatJid = toJid(msg);
    const timestamp = new Date(msg.createdTimestamp).toISOString();
    const registeredGroups = getAllRegisteredGroups();
    const isRegistered = !!registeredGroups[chatJid];

    // For guild messages, only process if bot is @mentioned
    if (msg.guild) {
      const isMentioned = client.user && msg.mentions.has(client.user.id);

      // Always store chat metadata for discovery
      const channelName = (msg.channel as TextChannel).name || chatJid;
      storeChatMetadata(chatJid, timestamp, `#${channelName} (${msg.guild.name})`);

      if (!isMentioned && isRegistered) {
        // Store message content for registered channels even without mention
        storeGenericMessage(
          msg.id,
          chatJid,
          msg.author.id,
          msg.author.displayName || msg.author.username,
          msg.content,
          timestamp,
          false,
        );
        return;
      }

      if (!isMentioned) return;
    }

    // DMs: always store metadata and process
    if (!msg.guild) {
      storeChatMetadata(chatJid, timestamp, `DM: ${msg.author.displayName || msg.author.username}`);
    }

    // Store full message content for registered groups
    if (isRegistered) {
      // Strip the bot mention from content for cleaner processing
      let content = msg.content;
      if (client.user) {
        content = content.replace(new RegExp(`<@!?${client.user.id}>`, 'g'), '').trim();
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
export async function setDiscordTyping(jid: string, isTyping: boolean): Promise<void> {
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
      const user = await client.users.fetch(userId);
      const dmChannel = await user.createDM();
      channelId = dmChannel.id;
    } else {
      channelId = jid.replace('discord:', '');
    }

    const channel = await client.channels.fetch(channelId);
    if (!channel || !('sendTyping' in channel)) return;

    const textChannel = channel as TextChannel;
    // Fire immediately, then every 9s
    await textChannel.sendTyping();
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
      const user = await client.users.fetch(userId);
      const dmChannel = await user.createDM();
      channelId = dmChannel.id;
    } else {
      // Guild channel: extract channel ID
      channelId = jid.replace('discord:', '');
    }

    const channel = await client.channels.fetch(channelId);
    if (channel && 'send' in channel) {
      // Discord has a 2000 char limit per message
      if (text.length <= 2000) {
        await (channel as TextChannel).send(text);
      } else {
        // Split long messages
        for (let i = 0; i < text.length; i += 2000) {
          await (channel as TextChannel).send(text.slice(i, i + 2000));
        }
      }
      logger.info({ jid, length: text.length }, 'Discord message sent');
    } else {
      logger.error({ jid, channelId }, 'Channel not found or not text-based');
    }
  } catch (err) {
    logger.error({ jid, err }, 'Failed to send Discord message');
  }
}
