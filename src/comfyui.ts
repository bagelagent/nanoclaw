import fs from 'fs';
import path from 'path';
import { logger } from './logger.js';

let comfyuiUrl = '';

export function initComfyUI(url: string): void {
  if (!url) {
    logger.warn('COMFYUI_URL not provided, ComfyUI image generation disabled');
    return;
  }
  comfyuiUrl = url.replace(/\/+$/, '');
  logger.info({ url: comfyuiUrl }, 'ComfyUI image generation initialized');
}

export function isComfyUIEnabled(): boolean {
  return !!comfyuiUrl;
}

/**
 * Check if the ComfyUI server is reachable right now.
 * Returns status info on success, or null if the PC is off / unreachable.
 */
export async function checkComfyUIAvailable(): Promise<{ status: string; vram_free?: number } | null> {
  if (!comfyuiUrl) return null;
  try {
    const res = await fetch(`${comfyuiUrl}/system_stats`, {
      signal: AbortSignal.timeout(5000),
    });
    if (!res.ok) return null;
    const stats = await res.json() as any;
    const gpu = stats?.devices?.[0];
    return {
      status: 'online',
      vram_free: gpu ? gpu.vram_free : undefined,
    };
  } catch {
    return null;
  }
}

interface ComfyUIOptions {
  prompt: string;
  negativePrompt?: string;
  width?: number;
  height?: number;
  steps?: number;
  cfgScale?: number;
  checkpoint?: string;
}

const DEFAULT_WORKFLOW = {
  '1': {
    class_type: 'UNETLoader',
    inputs: { unet_name: 'z_image_turbo_bf16.safetensors', weight_dtype: 'default' },
  },
  '2': {
    class_type: 'CLIPLoader',
    inputs: { clip_name: 'qwen_3_4b.safetensors', type: 'lumina2', device: 'default' },
  },
  '3': {
    class_type: 'VAELoader',
    inputs: { vae_name: 'ae.safetensors' },
  },
  '4': {
    class_type: 'CLIPTextEncode',
    _meta: { title: 'Positive Prompt' },
    inputs: { text: '', clip: ['2', 0] },
  },
  '5': {
    class_type: 'ConditioningZeroOut',
    inputs: { conditioning: ['4', 0] },
  },
  '6': {
    class_type: 'EmptySD3LatentImage',
    inputs: { width: 1024, height: 1024, batch_size: 1 },
  },
  '7': {
    class_type: 'ModelSamplingAuraFlow',
    inputs: { shift: 3.0, model: ['1', 0] },
  },
  '8': {
    class_type: 'KSampler',
    inputs: {
      seed: 0,
      steps: 8,
      cfg: 1.0,
      sampler_name: 'res_multistep',
      scheduler: 'simple',
      denoise: 1.0,
      model: ['7', 0],
      positive: ['4', 0],
      negative: ['5', 0],
      latent_image: ['6', 0],
    },
  },
  '9': {
    class_type: 'VAEDecode',
    inputs: { samples: ['8', 0], vae: ['3', 0] },
  },
  '10': {
    class_type: 'SaveImage',
    inputs: { filename_prefix: 'nanoclaw', images: ['9', 0] },
  },
};

function loadWorkflow(): Record<string, any> {
  const workflowPath = path.join(process.cwd(), 'data', 'comfyui-workflow.json');
  try {
    if (fs.existsSync(workflowPath)) {
      return JSON.parse(fs.readFileSync(workflowPath, 'utf-8'));
    }
  } catch (err) {
    logger.warn({ err }, 'Failed to load comfyui-workflow.json, using default');
  }
  return JSON.parse(JSON.stringify(DEFAULT_WORKFLOW));
}

function findNodeByClass(workflow: Record<string, any>, classType: string): any | null {
  for (const node of Object.values(workflow)) {
    if (node.class_type === classType) return node;
  }
  return null;
}

function findPositivePromptNode(workflow: Record<string, any>): any | null {
  const clips = Object.values(workflow).filter(
    (n: any) => n.class_type === 'CLIPTextEncode',
  );
  if (clips.length === 0) return null;
  const positive = clips.find(
    (n: any) => n._meta?.title?.toLowerCase().includes('positive'),
  );
  return positive || clips[0];
}

function findNegativePromptNode(workflow: Record<string, any>): any | null {
  const clips = Object.values(workflow).filter(
    (n: any) => n.class_type === 'CLIPTextEncode',
  );
  if (clips.length < 2) return null;
  const negative = clips.find(
    (n: any) => n._meta?.title?.toLowerCase().includes('negative'),
  );
  return negative || clips[1];
}

/** Returns true if the error is a network connectivity failure. */
function isConnectionError(err: unknown): boolean {
  if (err instanceof TypeError && err.message === 'fetch failed') return true;
  const code = (err as any)?.cause?.code;
  return code === 'ECONNREFUSED' || code === 'ETIMEDOUT' || code === 'ENOTFOUND'
    || code === 'EHOSTUNREACH' || code === 'ENETUNREACH' || code === 'ECONNRESET';
}

const GENERATION_TIMEOUT = 600000; // 10 minutes

export async function generateImageComfyUI(
  opts: ComfyUIOptions,
  groupDir: string,
): Promise<{ hostPath: string; containerPath: string; filename: string }> {
  if (!comfyuiUrl) throw new Error('ComfyUI not initialized (missing COMFYUI_URL)');

  // Pre-flight availability check
  const available = await checkComfyUIAvailable();
  if (!available) {
    throw new Error(
      'ComfyUI server is not reachable — the PC may be turned off or ComfyUI is not running. ' +
      'Please make sure the Windows PC is on and ComfyUI is started with --listen 0.0.0.0',
    );
  }

  const workflow = loadWorkflow();

  // Inject checkpoint/model — support both CheckpointLoaderSimple and UNETLoader
  if (opts.checkpoint) {
    const checkpoint = findNodeByClass(workflow, 'CheckpointLoaderSimple');
    const unetLoader = findNodeByClass(workflow, 'UNETLoader');
    if (checkpoint) {
      checkpoint.inputs.ckpt_name = opts.checkpoint;
    } else if (unetLoader) {
      unetLoader.inputs.unet_name = opts.checkpoint;
    }
  }

  const positiveNode = findPositivePromptNode(workflow);
  if (positiveNode) {
    positiveNode.inputs.text = opts.prompt;
  }

  const negativeNode = findNegativePromptNode(workflow);
  if (negativeNode) {
    negativeNode.inputs.text = opts.negativePrompt || '';
  }

  const latent = findNodeByClass(workflow, 'EmptySD3LatentImage')
    || findNodeByClass(workflow, 'EmptyLatentImage');
  if (latent) {
    if (opts.width) latent.inputs.width = opts.width;
    if (opts.height) latent.inputs.height = opts.height;
  }

  const sampler = findNodeByClass(workflow, 'KSampler');
  if (sampler) {
    if (opts.steps) sampler.inputs.steps = opts.steps;
    if (opts.cfgScale !== undefined) sampler.inputs.cfg = opts.cfgScale;
    sampler.inputs.seed = Math.floor(Math.random() * Number.MAX_SAFE_INTEGER);
  }

  // Queue the prompt
  let queueRes: Response;
  try {
    queueRes = await fetch(`${comfyuiUrl}/prompt`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt: workflow }),
      signal: AbortSignal.timeout(30000),
    });
  } catch (err) {
    if (isConnectionError(err)) {
      throw new Error(
        'Lost connection to ComfyUI while submitting the prompt — the PC may have gone to sleep or ComfyUI crashed.',
      );
    }
    throw err;
  }

  if (!queueRes.ok) {
    const body = await queueRes.text();
    throw new Error(`ComfyUI /prompt failed (${queueRes.status}): ${body}`);
  }

  const { prompt_id } = (await queueRes.json()) as { prompt_id: string };
  logger.debug({ prompt_id }, 'ComfyUI prompt queued');

  // Poll for completion
  const POLL_INTERVAL = 3000;
  const startTime = Date.now();

  while (Date.now() - startTime < GENERATION_TIMEOUT) {
    await new Promise((r) => setTimeout(r, POLL_INTERVAL));

    let histRes: Response;
    try {
      histRes = await fetch(`${comfyuiUrl}/history/${prompt_id}`, {
        signal: AbortSignal.timeout(10000),
      });
    } catch (err) {
      if (isConnectionError(err)) {
        throw new Error(
          'Lost connection to ComfyUI during image generation — the PC may have gone to sleep or the network dropped.',
        );
      }
      continue;
    }
    if (!histRes.ok) continue;

    const history = (await histRes.json()) as Record<string, any>;
    const entry = history[prompt_id];
    if (!entry) continue;

    if (entry.status?.status_str === 'error') {
      throw new Error(`ComfyUI generation failed: ${JSON.stringify(entry.status)}`);
    }

    if (entry.outputs) {
      // Find the SaveImage output
      for (const nodeOutput of Object.values(entry.outputs) as any[]) {
        if (nodeOutput.images && nodeOutput.images.length > 0) {
          const img = nodeOutput.images[0];
          const viewUrl = `${comfyuiUrl}/view?filename=${encodeURIComponent(img.filename)}&subfolder=${encodeURIComponent(img.subfolder || '')}&type=${encodeURIComponent(img.type || 'output')}`;

          const imgRes = await fetch(viewUrl, {
            signal: AbortSignal.timeout(30000),
          });
          if (!imgRes.ok) {
            throw new Error(`ComfyUI /view failed (${imgRes.status})`);
          }

          const imageBuffer = Buffer.from(await imgRes.arrayBuffer());
          const filename = `comfyui-${Date.now()}.png`;
          const tmpDir = path.join(groupDir, 'tmp');
          fs.mkdirSync(tmpDir, { recursive: true });
          const hostPath = path.join(tmpDir, filename);

          // Atomic write
          const tempPath = `${hostPath}.tmp`;
          fs.writeFileSync(tempPath, imageBuffer);
          fs.renameSync(tempPath, hostPath);

          logger.info(
            { prompt: opts.prompt.slice(0, 100), filename, elapsed: Date.now() - startTime },
            'ComfyUI image generated',
          );
          return {
            hostPath,
            containerPath: `/workspace/group/tmp/${filename}`,
            filename,
          };
        }
      }
      throw new Error('ComfyUI completed but no image output found');
    }
  }

  throw new Error(`ComfyUI generation timed out after ${GENERATION_TIMEOUT / 1000}s`);
}
