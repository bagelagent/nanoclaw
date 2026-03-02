import fs from 'fs';
import path from 'path';
import { GoogleGenAI } from '@google/genai';
import { logger } from './logger.js';

let genai: GoogleGenAI | null = null;
const IMAGE_MODEL = process.env.GEMINI_IMAGE_MODEL || 'gemini-2.5-flash-image';

export function initGemini(apiKey: string): void {
  if (!apiKey) {
    logger.warn('Google API key not provided, image generation disabled');
    return;
  }
  genai = new GoogleGenAI({ apiKey });
  logger.info('Gemini image generation initialized');
}

export function isGeminiEnabled(): boolean {
  return genai !== null;
}

export async function generateImageGemini(
  prompt: string,
  aspectRatio: string,
  groupDir: string,
): Promise<{ hostPath: string; containerPath: string; filename: string }> {
  if (!genai) throw new Error('Gemini not initialized (missing GOOGLE_API_KEY)');

  const response = await genai.models.generateContent({
    model: IMAGE_MODEL,
    contents: prompt,
    config: {
      responseModalities: ['IMAGE'],
      imageConfig: { aspectRatio },
    },
  });

  const parts = response.candidates?.[0]?.content?.parts;
  if (!parts) throw new Error('No content in Gemini response');

  for (const part of parts) {
    if (part.inlineData?.data) {
      const mimeType = part.inlineData.mimeType || 'image/png';
      const ext = mimeType.includes('jpeg') || mimeType.includes('jpg') ? 'jpg' : 'png';
      const filename = `gemini-${Date.now()}.${ext}`;

      const tmpDir = path.join(groupDir, 'tmp');
      fs.mkdirSync(tmpDir, { recursive: true });
      const hostPath = path.join(tmpDir, filename);

      // Atomic write
      const tempPath = `${hostPath}.tmp`;
      fs.writeFileSync(tempPath, Buffer.from(part.inlineData.data, 'base64'));
      fs.renameSync(tempPath, hostPath);

      logger.info({ prompt: prompt.slice(0, 100), filename }, 'Gemini image generated');
      return { hostPath, containerPath: `/workspace/group/tmp/${filename}`, filename };
    }
  }

  throw new Error('No image data in Gemini response');
}
