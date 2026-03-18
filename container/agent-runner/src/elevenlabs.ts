/**
 * ElevenLabs API client for container-side audio generation.
 * Uses raw fetch() — no SDK dependency needed.
 */
import fs from 'fs';
import path from 'path';

const API_BASE = 'https://api.elevenlabs.io/v1';
const DEFAULT_VOICE = process.env.ELEVENLABS_DEFAULT_VOICE || 'JBFqnCBsd6RMkjVDRZzb'; // Rachel
const DEFAULT_MODEL = 'eleven_multilingual_v2';

function getApiKey(): string {
  const key = process.env.ELEVENLABS_API_KEY;
  if (!key) throw new Error('ELEVENLABS_API_KEY not set');
  return key;
}

/**
 * Generate speech from text using ElevenLabs TTS.
 * Output format: opus in ogg container (WhatsApp-compatible).
 */
export async function generateTts(
  text: string,
  voiceId: string = DEFAULT_VOICE,
  modelId: string = DEFAULT_MODEL,
  outputDir: string,
): Promise<string> {
  const res = await fetch(`${API_BASE}/text-to-speech/${voiceId}?output_format=opus`, {
    method: 'POST',
    headers: {
      'xi-api-key': getApiKey(),
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      text,
      model_id: modelId,
    }),
  });

  if (!res.ok) {
    const body = await res.text();
    throw new Error(`ElevenLabs TTS failed (${res.status}): ${body}`);
  }

  const buffer = Buffer.from(await res.arrayBuffer());
  const filename = `elevenlabs-tts-${Date.now()}.ogg`;
  const outputPath = path.join(outputDir, filename);
  fs.writeFileSync(outputPath, buffer);
  return outputPath;
}

/**
 * Generate a sound effect from a text prompt.
 */
export async function generateSoundEffect(
  prompt: string,
  durationSeconds?: number,
  outputDir: string = '/workspace/group',
): Promise<string> {
  const body: Record<string, any> = { text: prompt };
  if (durationSeconds != null) body.duration_seconds = durationSeconds;

  const res = await fetch(`${API_BASE}/sound-generation`, {
    method: 'POST',
    headers: {
      'xi-api-key': getApiKey(),
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`ElevenLabs sound effect failed (${res.status}): ${text}`);
  }

  const buffer = Buffer.from(await res.arrayBuffer());
  const filename = `elevenlabs-sfx-${Date.now()}.mp3`;
  const outputPath = path.join(outputDir, filename);
  fs.writeFileSync(outputPath, buffer);
  return outputPath;
}

/**
 * Generate music from a text prompt.
 */
export async function generateMusic(
  prompt: string,
  durationMs?: number,
  forceInstrumental?: boolean,
  outputDir: string = '/workspace/group',
): Promise<string> {
  const body: Record<string, any> = { prompt };
  if (durationMs != null) body.duration_ms = durationMs;
  if (forceInstrumental != null) body.instrumental = forceInstrumental;

  const res = await fetch(`${API_BASE}/music`, {
    method: 'POST',
    headers: {
      'xi-api-key': getApiKey(),
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`ElevenLabs music generation failed (${res.status}): ${text}`);
  }

  const buffer = Buffer.from(await res.arrayBuffer());
  const filename = `elevenlabs-music-${Date.now()}.mp3`;
  const outputPath = path.join(outputDir, filename);
  fs.writeFileSync(outputPath, buffer);
  return outputPath;
}
