import fs from 'fs';
import os from 'os';
import path from 'path';

import { ImapFlow } from 'imapflow';
import nodemailer from 'nodemailer';
import { simpleParser } from 'mailparser';

import { logger } from '../logger.js';
import { registerChannel, ChannelOpts } from './registry.js';
import {
  Channel,
  OnChatMetadata,
  OnInboundMessage,
  RegisteredGroup,
} from '../types.js';

interface YahooConfig {
  email: string;
  appPassword: string;
  allowedSenders: string[];
}

interface ThreadMeta {
  sender: string;
  senderName: string;
  subject: string;
  messageId: string; // RFC 2822 Message-ID for In-Reply-To
}

const CONFIG_DIR = path.join(os.homedir(), '.yahoo-email');
const CONFIG_PATH = path.join(CONFIG_DIR, 'config.json');

const IMAP_HOST = 'imap.mail.yahoo.com';
const IMAP_PORT = 993;
const SMTP_HOST = 'smtp.mail.yahoo.com';
const SMTP_PORT = 465;

const YAHOO_JID = 'yahoo:inbox';

export class YahooEmailChannel implements Channel {
  name = 'yahoo';

  private imapClient: ImapFlow | null = null;
  private smtpTransport: nodemailer.Transporter | null = null;
  private opts: ChannelOpts;
  private pollIntervalMs: number;
  private pollTimer: ReturnType<typeof setTimeout> | null = null;
  private processedIds = new Set<string>();
  private threadMeta = new Map<string, ThreadMeta>();
  private consecutiveErrors = 0;
  private userEmail = '';
  private allowedSenders = new Set<string>();
  private connected = false;

  constructor(opts: ChannelOpts, pollIntervalMs = 60000) {
    this.opts = opts;
    this.pollIntervalMs = pollIntervalMs;
  }

  async connect(): Promise<void> {
    if (!fs.existsSync(CONFIG_PATH)) {
      logger.warn(
        'Yahoo credentials not found in ~/.yahoo-email/config.json. Skipping Yahoo channel.',
      );
      return;
    }

    const config: YahooConfig = JSON.parse(
      fs.readFileSync(CONFIG_PATH, 'utf-8'),
    );
    this.userEmail = config.email;
    this.allowedSenders = new Set(
      config.allowedSenders.map((s) => s.toLowerCase()),
    );

    // Create IMAP client
    this.imapClient = new ImapFlow({
      host: IMAP_HOST,
      port: IMAP_PORT,
      secure: true,
      auth: {
        user: config.email,
        pass: config.appPassword,
      },
      logger: false,
    });

    // Create SMTP transport
    this.smtpTransport = nodemailer.createTransport({
      host: SMTP_HOST,
      port: SMTP_PORT,
      secure: true,
      auth: {
        user: config.email,
        pass: config.appPassword,
      },
    });

    // Connect IMAP
    await this.imapClient.connect();
    this.connected = true;
    logger.info({ email: this.userEmail }, 'Yahoo IMAP connected');

    // Verify SMTP
    await this.smtpTransport.verify();
    logger.info('Yahoo SMTP verified');

    // Report metadata so the group can be auto-registered
    this.opts.onChatMetadata(
      YAHOO_JID,
      new Date().toISOString(),
      'Email',
      'yahoo',
      false,
    );

    // Start polling
    const schedulePoll = () => {
      const backoffMs =
        this.consecutiveErrors > 0
          ? Math.min(
              this.pollIntervalMs * Math.pow(2, this.consecutiveErrors),
              30 * 60 * 1000,
            )
          : this.pollIntervalMs;
      this.pollTimer = setTimeout(() => {
        this.pollForMessages()
          .catch((err) => logger.error({ err }, 'Yahoo poll error'))
          .finally(() => {
            if (this.connected) schedulePoll();
          });
      }, backoffMs);
    };

    // Initial poll (non-blocking to avoid holding up other channel connections)
    this.pollForMessages()
      .catch((err) => logger.error({ err }, 'Yahoo initial poll error'))
      .finally(() => schedulePoll());
  }

  async sendMessage(jid: string, text: string): Promise<void> {
    if (!this.smtpTransport) {
      logger.warn('Yahoo SMTP not initialized');
      return;
    }

    // Check for thread metadata (reply to last email)
    // threadMeta is keyed by the YAHOO_JID for simplicity since it's a single inbox
    const meta = this.threadMeta.get(YAHOO_JID);

    let to: string;
    let subject: string;
    let inReplyTo: string | undefined;
    let body: string;

    if (meta) {
      // Reply to the last email sender
      to = meta.sender;
      subject = meta.subject.startsWith('Re:')
        ? meta.subject
        : `Re: ${meta.subject}`;
      inReplyTo = meta.messageId;
      body = text;
    } else {
      // Proactive/scheduled send — first line must be "To: address@example.com"
      const toMatch = text.match(/^To:\s*(.+)/i);
      if (!toMatch) {
        logger.error(
          'Yahoo send: no thread metadata and no "To:" header in first line. Cannot send.',
        );
        return;
      }
      to = toMatch[1].trim();
      body = text.slice(toMatch[0].length).trimStart();
      // Extract subject if second line is "Subject: ..."
      const subjectMatch = body.match(/^Subject:\s*(.+)/i);
      if (subjectMatch) {
        subject = subjectMatch[1].trim();
        body = body.slice(subjectMatch[0].length).trimStart();
      } else {
        subject = `Message from ${this.userEmail}`;
      }
    }

    try {
      await this.smtpTransport.sendMail({
        from: this.userEmail,
        to,
        subject,
        text: body,
        ...(inReplyTo ? { inReplyTo, references: inReplyTo } : {}),
      });
      logger.info({ to, subject }, 'Yahoo email sent');
    } catch (err) {
      logger.error({ to, err }, 'Failed to send Yahoo email');
    }
  }

  async sendEmail(opts: {
    to: string;
    subject: string;
    body: string;
    attachments?: Array<{ filename: string; path: string }>;
  }): Promise<void> {
    if (!this.smtpTransport) {
      logger.warn('Yahoo SMTP not initialized');
      return;
    }

    try {
      await this.smtpTransport.sendMail({
        from: this.userEmail,
        to: opts.to,
        subject: opts.subject,
        text: opts.body,
        attachments: opts.attachments?.map((a) => ({
          filename: a.filename,
          path: a.path,
        })),
      });
      logger.info(
        { to: opts.to, subject: opts.subject, attachments: opts.attachments?.length || 0 },
        'Yahoo email sent (structured)',
      );
    } catch (err) {
      logger.error({ to: opts.to, err }, 'Failed to send Yahoo email');
      throw err;
    }
  }

  isConnected(): boolean {
    return this.connected;
  }

  ownsJid(jid: string): boolean {
    return jid.startsWith('yahoo:');
  }

  async disconnect(): Promise<void> {
    if (this.pollTimer) {
      clearTimeout(this.pollTimer);
      this.pollTimer = null;
    }
    this.connected = false;
    if (this.imapClient) {
      try {
        await this.imapClient.logout();
      } catch {
        // ignore logout errors
      }
      this.imapClient = null;
    }
    this.smtpTransport = null;
    logger.info('Yahoo channel stopped');
  }

  // --- Private ---

  private async pollForMessages(): Promise<void> {
    if (!this.imapClient) return;

    logger.debug('Yahoo: polling for messages...');

    try {
      const lock = await this.imapClient.getMailboxLock('INBOX');
      try {
        // Search for unseen messages and filter by sender in code
        // Yahoo IMAP FROM/SINCE search is unreliable, so we get all unseen
        // and use envelope-only fetch to quickly filter by sender
        const unseenResult = await this.imapClient.search({
          seen: false,
        });
        const unseenMessages = unseenResult || [];

        logger.info(
          { count: unseenMessages.length },
          'Yahoo: unseen messages',
        );

        // Quick envelope check: only download full body for allowlisted senders
        for (const seq of unseenMessages) {
          const msgId = String(seq);
          if (this.processedIds.has(msgId)) continue;

          // Fetch envelope only (lightweight) to check sender
          const envelope = await this.imapClient.fetchOne(String(seq), {
            envelope: true,
          });
          if (!envelope || !envelope.envelope) {
            this.processedIds.add(msgId);
            continue;
          }

          const fromAddr = envelope.envelope.from?.[0];
          const senderEmail = (fromAddr?.address || '').toLowerCase();

          // Skip non-allowlisted senders without downloading body
          if (
            !this.allowedSenders.has(senderEmail) ||
            senderEmail === this.userEmail.toLowerCase()
          ) {
            this.processedIds.add(msgId);
            continue;
          }

          this.processedIds.add(msgId);
          await this.processMessage(seq);
        }
      } finally {
        lock.release();
      }

      // Cap processed ID set
      if (this.processedIds.size > 5000) {
        const ids = [...this.processedIds];
        this.processedIds = new Set(ids.slice(ids.length - 2500));
      }

      this.consecutiveErrors = 0;
    } catch (err) {
      this.consecutiveErrors++;
      const backoffMs = Math.min(
        this.pollIntervalMs * Math.pow(2, this.consecutiveErrors),
        30 * 60 * 1000,
      );
      logger.error(
        {
          err,
          consecutiveErrors: this.consecutiveErrors,
          nextPollMs: backoffMs,
        },
        'Yahoo poll failed',
      );
    }
  }

  private async processMessage(seq: number): Promise<void> {
    if (!this.imapClient) return;

    // Fetch the raw message source
    const downloaded = await this.imapClient.download(String(seq));
    if (!downloaded || !downloaded.content) {
      logger.debug({ seq }, 'Yahoo: could not download message');
      return;
    }

    const parsed = await simpleParser(downloaded.content);

    const fromAddr = parsed.from?.value?.[0];
    if (!fromAddr) return;

    const senderEmail = (fromAddr.address || '').toLowerCase();
    const senderName = fromAddr.name || senderEmail;
    const subject = parsed.subject || '(no subject)';
    const messageId = parsed.messageId || '';

    // Extract text body
    const body = typeof parsed.text === 'string' ? parsed.text : '';

    if (!body) {
      logger.debug({ seq, subject }, 'Yahoo: skipping email with no text body');
      await this.markSeen(seq);
      return;
    }

    // Cache thread metadata for replies
    this.threadMeta.set(YAHOO_JID, {
      sender: senderEmail,
      senderName,
      subject,
      messageId,
    });

    // Store chat metadata
    const now = new Date().toISOString();
    this.opts.onChatMetadata(
      YAHOO_JID,
      now,
      `Email: ${subject}`,
      'yahoo',
      false,
    );

    const content = `[Email from ${senderName} <${senderEmail}>]\nSubject: ${subject}\n\n${body}`;

    // Use current time as the storage timestamp so the message loop's cursor
    // (lastTimestamp) doesn't skip it — email send time can be in the past.
    const storageTimestamp = new Date().toISOString();

    this.opts.onMessage(YAHOO_JID, {
      id: `yahoo-${seq}-${Date.now()}`,
      chat_jid: YAHOO_JID,
      sender: senderEmail,
      sender_name: senderName,
      content,
      timestamp: storageTimestamp,
      is_from_me: false,
    });

    // Mark as seen
    await this.markSeen(seq);

    logger.info(
      { from: senderName, subject },
      'Yahoo email delivered to yahoo:inbox',
    );
  }

  private async markSeen(seq: number): Promise<void> {
    if (!this.imapClient) return;
    try {
      await this.imapClient.messageFlagsAdd(String(seq), ['\\Seen']);
    } catch (err) {
      logger.warn({ seq, err }, 'Yahoo: failed to mark message as seen');
    }
  }
}

registerChannel('yahoo', (opts: ChannelOpts) => {
  if (!fs.existsSync(CONFIG_PATH)) {
    return null;
  }
  return new YahooEmailChannel(opts);
});
