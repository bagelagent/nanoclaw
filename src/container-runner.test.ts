import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest';

// Mock config
vi.mock('./config.js', () => ({
  CONTAINER_IMAGE: 'nanoclaw-agent:latest',
  DATA_DIR: '/tmp/nanoclaw-test-data',
  GROUPS_DIR: '/tmp/nanoclaw-test-groups',
  IDLE_TIMEOUT: 1800000, // 30min
  TIMEZONE: 'America/Los_Angeles',
}));

// Mock logger
vi.mock('./logger.js', () => ({
  logger: {
    debug: vi.fn(),
    info: vi.fn(),
    warn: vi.fn(),
    error: vi.fn(),
  },
}));

// Mock fs
vi.mock('fs', async () => {
  const actual = await vi.importActual<typeof import('fs')>('fs');
  return {
    ...actual,
    default: {
      ...actual,
      existsSync: vi.fn(() => false),
      mkdirSync: vi.fn(),
      writeFileSync: vi.fn(),
      readFileSync: vi.fn(() => '{}'),
      readdirSync: vi.fn(() => []),
      statSync: vi.fn(() => ({ isDirectory: () => false })),
      cpSync: vi.fn(),
    },
  };
});

// Mock mount-security
vi.mock('./mount-security.js', () => ({
  validateAdditionalMounts: vi.fn(() => []),
}));

// Mock child_process
vi.mock('child_process', async () => {
  const actual =
    await vi.importActual<typeof import('child_process')>('child_process');
  return {
    ...actual,
    execSync: vi.fn(() => ''),
    exec: vi.fn(
      (_cmd: string, _opts: unknown, cb?: (err: Error | null, stdout?: string) => void) => {
        if (cb) cb(null, '');
      },
    ),
  };
});

import { ContainerOutput } from './container-runner.js';

describe('ContainerOutput type', () => {
  it('has expected shape with null result', () => {
    const output: ContainerOutput = {
      status: 'success',
      result: null,
    };
    expect(output.status).toBe('success');
    expect(output.result).toBeNull();
  });

  it('can represent errors', () => {
    const output: ContainerOutput = {
      status: 'error',
      result: null,
      error: 'Container timed out',
    };
    expect(output.status).toBe('error');
    expect(output.error).toBe('Container timed out');
  });
});
