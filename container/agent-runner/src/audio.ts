/**
 * Audio utilities for the agent container
 * Handles transcription using OpenAI Whisper
 */
import fs from 'fs';
import OpenAI from 'openai';

let openai: OpenAI | null = null;

// Initialize OpenAI client if API key is available
const apiKey = process.env.OPENAI_API_KEY;
if (apiKey) {
  openai = new OpenAI({ apiKey });
}

/**
 * Transcribe an audio file using OpenAI Whisper
 * @param audioPath Path to the audio file
 * @returns Transcribed text or null if transcription fails
 */
export async function transcribeAudioFile(audioPath: string): Promise<string | null> {
  if (!openai) {
    console.warn('OpenAI not initialized, skipping transcription');
    return null;
  }

  if (!fs.existsSync(audioPath)) {
    console.error(`Audio file not found: ${audioPath}`);
    return null;
  }

  try {
    const transcription = await openai.audio.transcriptions.create({
      file: fs.createReadStream(audioPath),
      model: 'whisper-1',
    });

    return transcription.text;
  } catch (error) {
    console.error('Audio transcription failed:', error);
    return null;
  }
}

/**
 * Download and transcribe an audio file from a URL
 * @param url URL of the audio file
 * @param outputPath Path to save the downloaded audio
 * @returns Transcribed text or null if transcription fails
 */
export async function transcribeAudioUrl(url: string, outputPath: string): Promise<string | null> {
  try {
    // Download the audio file
    const response = await fetch(url);
    if (!response.ok) {
      console.error(`Failed to download audio: ${response.status}`);
      return null;
    }

    const arrayBuffer = await response.arrayBuffer();
    fs.writeFileSync(outputPath, Buffer.from(arrayBuffer));

    // Transcribe it
    const transcription = await transcribeAudioFile(outputPath);

    // Clean up
    if (fs.existsSync(outputPath)) {
      fs.unlinkSync(outputPath);
    }

    return transcription;
  } catch (error) {
    console.error('Failed to download and transcribe audio:', error);
    return null;
  }
}
