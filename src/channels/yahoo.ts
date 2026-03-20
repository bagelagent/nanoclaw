import fs from 'fs';
import os from 'os';
import path from 'path';

import { ImapFlow } from 'imapflow';
import nodemailer from 'nodemailer';
import { simpleParser } from 'mailparser';

import { GROUPS_DIR } from '../config.js';
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
  allRecipients: string[]; // All addresses on the thread (From + To + CC minus self)
}

const CONFIG_DIR = path.join(os.homedir(), '.yahoo-email');
const CONFIG_PATH = path.join(CONFIG_DIR, 'config.json');

const IMAP_HOST = 'imap.mail.yahoo.com';
const IMAP_PORT = 993;
const SMTP_HOST = 'smtp.mail.yahoo.com';
const SMTP_PORT = 465;

const YAHOO_JID = 'yahoo:inbox';
const PROCESSED_FOLDER = 'NanoClaw';

export class YahooEmailChannel implements Channel {
  name = 'yahoo';

  private imapClient: ImapFlow | null = null;
  private smtpTransport: nodemailer.Transporter | null = null;
  private opts: ChannelOpts;
  private pollIntervalMs: number;
  private pollTimer: ReturnType<typeof setTimeout> | null = null;
  private threadMeta = new Map<string, ThreadMeta>();
  private consecutiveErrors = 0;
  private userEmail = '';
  private allowedSenders = new Set<string>();
  private connected = false;
  private config: YahooConfig | null = null;

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
    this.config = config;
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

    // Create processed folder if it doesn't exist.
    // Only attempt once — mailboxCreate can disrupt IMAP session state.
    const mailboxes = await this.imapClient.list();
    const folderExists = mailboxes.some((m) => m.path === PROCESSED_FOLDER);
    if (!folderExists) {
      try {
        await this.imapClient.mailboxCreate(PROCESSED_FOLDER);
        logger.info('Yahoo: created NanoClaw folder');
      } catch {
        // Folder may already exist under a different path format
      }
      // Reconnect to reset IMAP state after mailboxCreate
      await this.imapClient.logout();
      await this.imapClient.connect();
      logger.info('Yahoo: reconnected after folder creation');
    }

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
              5 * 60 * 1000,
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

    // Parse headers from the agent's message:
    //   To: recipient@example.com     (required — identifies the thread)
    //   Reply-Mode: sender            (optional — "all" or "sender", defaults to "all")
    //   Subject: custom subject       (optional — overrides Re: subject)
    const toMatch = text.match(/^To:\s*(.+)/i);
    if (!toMatch) {
      logger.error(
        'Yahoo send: no "To:" header in first line. Cannot send.',
      );
      return;
    }
    const toAddr = toMatch[1].trim().toLowerCase();
    let body = text.slice(toMatch[0].length).trimStart();

    // Parse optional Reply-Mode header
    let replyMode: 'all' | 'sender' = 'all';
    const modeMatch = body.match(/^Reply-Mode:\s*(all|sender)/i);
    if (modeMatch) {
      replyMode = modeMatch[1].toLowerCase() as 'all' | 'sender';
      body = body.slice(modeMatch[0].length).trimStart();
    }

    // Look up thread metadata by recipient for proper reply threading
    const meta = this.threadMeta.get(toAddr);

    let subject: string;
    let inReplyTo: string | undefined;
    let cc: string | undefined;

    // Check for explicit Subject: line
    const subjectMatch = body.match(/^Subject:\s*(.+)/i);
    if (subjectMatch) {
      subject = subjectMatch[1].trim();
      body = body.slice(subjectMatch[0].length).trimStart();
    } else if (meta) {
      // Thread reply — use Re: original subject
      subject = meta.subject.startsWith('Re:')
        ? meta.subject
        : `Re: ${meta.subject}`;
      inReplyTo = meta.messageId;
    } else {
      subject = `Message from ${this.userEmail}`;
    }

    // Build CC list for reply-all
    if (replyMode === 'all' && meta && meta.allRecipients.length > 1) {
      const ccList = meta.allRecipients.filter((a) => a !== toAddr);
      if (ccList.length > 0) {
        cc = ccList.join(', ');
      }
    }

    try {
      await this.smtpTransport.sendMail({
        from: this.userEmail,
        to: toAddr,
        subject,
        text: body,
        ...(cc ? { cc } : {}),
        ...(inReplyTo ? { inReplyTo, references: inReplyTo } : {}),
      });
      logger.info({ to: toAddr, cc, subject }, 'Yahoo email sent');
    } catch (err) {
      logger.error({ to: toAddr, err }, 'Failed to send Yahoo email');
    }
  }

  async sendEmail(opts: {
    to: string;
    subject: string;
    body: string;
    attachments?: Array<{ filename: string; path: string; inline?: boolean }>;
  }): Promise<void> {
    if (!this.smtpTransport) {
      logger.warn('Yahoo SMTP not initialized');
      return;
    }

    try {
      const inlineAttachments = opts.attachments?.filter((a) => a.inline) || [];
      const regularAttachments =
        opts.attachments?.filter((a) => !a.inline) || [];

      // Build HTML body if there are inline images
      let html: string | undefined;
      if (inlineAttachments.length > 0) {
        // Convert plain text body to HTML paragraphs
        const bodyHtml = opts.body
          .split('\n')
          .map((line) => (line.trim() ? `<p>${line}</p>` : '<br>'))
          .join('\n');
        const imagesHtml = inlineAttachments
          .map(
            (a) =>
              `<p><img src="cid:${a.filename}" style="max-width:100%;" /></p>`,
          )
          .join('\n');
        html = bodyHtml + '\n' + imagesHtml;
      }

      const nodemailerAttachments = [
        ...regularAttachments.map((a) => ({
          filename: a.filename,
          path: a.path,
        })),
        ...inlineAttachments.map((a) => ({
          filename: a.filename,
          path: a.path,
          cid: a.filename,
        })),
      ];

      await this.smtpTransport.sendMail({
        from: this.userEmail,
        to: opts.to,
        subject: opts.subject,
        text: opts.body,
        ...(html ? { html } : {}),
        ...(nodemailerAttachments.length > 0
          ? { attachments: nodemailerAttachments }
          : {}),
      });
      logger.info(
        {
          to: opts.to,
          subject: opts.subject,
          attachments: opts.attachments?.length || 0,
        },
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

    try {
      const lock = await this.imapClient.getMailboxLock('INBOX');
      try {
        // Get ALL messages in inbox. Every message gets moved to NanoClaw
        // folder to maintain inbox zero. Allowlisted senders get processed first.
        const allResult = await this.imapClient.search({ all: true });
        const allMessages = allResult || [];

        if (allMessages.length === 0) return;

        logger.info({ count: allMessages.length }, 'Yahoo: messages in inbox');

        // Process in reverse (newest first) since messageMove shifts sequence numbers
        for (const seq of [...allMessages].reverse()) {
          const envelope = await this.imapClient.fetchOne(String(seq), {
            envelope: true,
          });

          const fromAddr = envelope ? envelope.envelope?.from?.[0] : undefined;
          const senderEmail = (
            (fromAddr && 'address' in fromAddr ? fromAddr.address : '') || ''
          ).toLowerCase();

          // Allowlisted sender (not self) — process before moving
          if (
            this.allowedSenders.has(senderEmail) &&
            senderEmail !== this.userEmail.toLowerCase()
          ) {
            await this.processMessage(seq);
          }

          // Move ALL messages to NanoClaw folder
          await this.moveToProcessed(seq);
        }
      } finally {
        lock.release();
      }

      this.consecutiveErrors = 0;
    } catch (err) {
      this.consecutiveErrors++;
      logger.error(
        {
          err,
          consecutiveErrors: this.consecutiveErrors,
        },
        'Yahoo poll failed',
      );

      // Reconnect IMAP if connection was lost
      const errMsg = err instanceof Error ? err.message : String(err);
      if (
        errMsg.includes('not available') ||
        errMsg.includes('closed') ||
        errMsg.includes('ECONNRESET')
      ) {
        await this.reconnectImap();
      }
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

    // Collect all recipients (To + CC) for reply-all support
    const collectAddresses = (
      field: typeof parsed.to,
    ): string[] => {
      if (!field) return [];
      const items = Array.isArray(field) ? field : [field];
      return items.flatMap((obj) =>
        (obj.value || []).map((a) => (a.address || '').toLowerCase()),
      );
    };
    const toAddresses = collectAddresses(parsed.to);
    const ccAddresses = collectAddresses(parsed.cc);
    const selfEmail = this.userEmail.toLowerCase();

    // All recipients = From + To + CC, deduplicated, minus self
    const allRecipients = [
      ...new Set([senderEmail, ...toAddresses, ...ccAddresses]),
    ].filter((a) => a && a !== selfEmail);

    // Format To/CC for display in message content
    const toDisplay = toAddresses.filter((a) => a !== selfEmail);
    const ccDisplay = ccAddresses.filter((a) => a !== selfEmail);

    // Extract text body
    const body = typeof parsed.text === 'string' ? parsed.text : '';

    // Cache thread metadata keyed by sender email for reply threading
    this.threadMeta.set(senderEmail, {
      sender: senderEmail,
      senderName,
      subject,
      messageId,
      allRecipients,
    });

    // Save attachments from allowlisted senders to the group workspace
    const attachmentPaths: string[] = [];
    if (
      parsed.attachments &&
      parsed.attachments.length > 0 &&
      this.allowedSenders.has(senderEmail)
    ) {
      const attachDir = path.join(GROUPS_DIR, 'email', 'attachments');
      fs.mkdirSync(attachDir, { recursive: true });

      for (const att of parsed.attachments) {
        if (!att.filename || !att.content) continue;
        // Sanitize filename and add timestamp to avoid collisions
        const safeName = att.filename.replace(/[^a-zA-Z0-9._-]/g, '_');
        const filename = `${Date.now()}-${safeName}`;
        const filePath = path.join(attachDir, filename);
        fs.writeFileSync(filePath, att.content);
        attachmentPaths.push(filename);
        logger.info(
          { filename, size: att.content.length },
          'Yahoo: saved attachment',
        );
      }
    }

    if (!body && attachmentPaths.length === 0) {
      logger.debug({ seq, subject }, 'Yahoo: skipping email with no text body and no attachments');
      return;
    }

    // Store chat metadata
    const now = new Date().toISOString();
    this.opts.onChatMetadata(
      YAHOO_JID,
      now,
      `Email: ${subject}`,
      'yahoo',
      false,
    );

    // Build message content with recipient info and attachment info
    let header = `[Email from ${senderName} <${senderEmail}>]`;
    if (toDisplay.length > 0) {
      header += `\nTo: ${toDisplay.join(', ')}`;
    }
    if (ccDisplay.length > 0) {
      header += `\nCC: ${ccDisplay.join(', ')}`;
    }
    header += `\nSubject: ${subject}`;
    let content = `${header}\n\n${body}`;
    if (attachmentPaths.length > 0) {
      const attList = attachmentPaths
        .map((f) => `  - /workspace/group/attachments/${f}`)
        .join('\n');
      content += `\n\n[Attachments - saved to workspace, read if needed to answer]\n${attList}`;
    }

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

    logger.info(
      { from: senderName, subject },
      'Yahoo email delivered to yahoo:inbox',
    );
  }

  private async moveToProcessed(seq: number): Promise<void> {
    if (!this.imapClient) return;
    try {
      await this.imapClient.messageMove(String(seq), PROCESSED_FOLDER);
    } catch (err) {
      logger.warn({ seq, err }, 'Yahoo: failed to move message to NanoClaw');
    }
  }

  private async reconnectImap(): Promise<void> {
    if (!this.config) return;
    logger.info('Yahoo: reconnecting IMAP...');

    // Close old connection
    if (this.imapClient) {
      try {
        await this.imapClient.logout();
      } catch {
        // ignore
      }
    }

    try {
      this.imapClient = new ImapFlow({
        host: IMAP_HOST,
        port: IMAP_PORT,
        secure: true,
        auth: {
          user: this.config.email,
          pass: this.config.appPassword,
        },
        logger: false,
      });
      await this.imapClient.connect();
      this.consecutiveErrors = 0;
      logger.info('Yahoo: IMAP reconnected');
    } catch (err) {
      logger.error({ err }, 'Yahoo: IMAP reconnect failed');
    }
  }
}

registerChannel('yahoo', (opts: ChannelOpts) => {
  if (!fs.existsSync(CONFIG_PATH)) {
    return null;
  }
  return new YahooEmailChannel(opts);
});
