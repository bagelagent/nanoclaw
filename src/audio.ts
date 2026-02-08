/**
 * Audio processing module for voice messages
 * Handles transcription and TTS using OpenAI
 */
import fs from 'fs';
import path from 'path';
import { Readable } from 'stream';

import { downloadMediaMessage, proto } from '@whiskeysockets/baileys';
import OpenAI from 'openai';

import { logger } from './logger.js';

let openai: OpenAI | null = null;

/**
 * Initialize OpenAI client with API key
 */
export function initOpenAI(apiKey: string): void {
  if (!apiKey) {
    logger.warn('OpenAI API key not provided, audio features disabled');
    return;
  }
  openai = new OpenAI({ apiKey });
  logger.info('OpenAI audio features initialized');
}

/**
 * Check if a message contains audio
 */
export function isAudioMessage(msg: proto.IWebMessageInfo): boolean {
  return !!(
    msg.message?.audioMessage ||
    msg.message?.extendedTextMessage?.contextInfo?.quotedMessage?.audioMessage
  );
}

/**
 * Transcribe an audio message using OpenAI Whisper
 */
export async function transcribeAudio(
  msg: proto.IWebMessageInfo,
  tmpDir: string,
): Promise<string | null> {
  if (!openai) {
    logger.warn('OpenAI not initialized, skipping transcription');
    return null;
  }

  try {
    // Download audio buffer
    const buffer = await downloadMediaMessage(
      msg as any, // Type mismatch between IWebMessageInfo and WAMessage
      'buffer',
      {},
      {
        logger: logger as any,
        reuploadRequest: async () => {
          throw new Error('Media re-upload not supported');
        },
      },
    );

    if (!buffer || buffer.length === 0) {
      logger.warn('Empty audio buffer');
      return null;
    }

    // Save to temporary file (Whisper API requires a file)
    const tempFilePath = path.join(tmpDir, `audio-${Date.now()}.ogg`);
    fs.mkdirSync(tmpDir, { recursive: true });
    fs.writeFileSync(tempFilePath, buffer);

    try {
      // Transcribe with Whisper
      const transcription = await openai.audio.transcriptions.create({
        file: fs.createReadStream(tempFilePath),
        model: 'whisper-1',
      });

      logger.info({ text: transcription.text }, 'Audio transcribed');
      return transcription.text;
    } finally {
      // Clean up temp file
      if (fs.existsSync(tempFilePath)) {
        fs.unlinkSync(tempFilePath);
      }
    }
  } catch (error) {
    logger.error({ error }, 'Audio transcription failed');
    return null;
  }
}

/**
 * Generate speech from text using OpenAI TTS
 * Returns the audio buffer
 */
export async function generateSpeech(
  text: string,
  voice: 'alloy' | 'echo' | 'fable' | 'onyx' | 'nova' | 'shimmer' = 'nova',
): Promise<Buffer | null> {
  if (!openai) {
    logger.warn('OpenAI not initialized, skipping TTS');
    return null;
  }

  try {
    const response = await openai.audio.speech.create({
      model: 'tts-1',
      voice,
      input: text,
      response_format: 'opus', // WhatsApp-compatible format
    });

    // Convert response to buffer
    const arrayBuffer = await response.arrayBuffer();
    return Buffer.from(arrayBuffer);
  } catch (error) {
    logger.error({ error }, 'TTS generation failed');
    return null;
  }
}
