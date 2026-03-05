/**
 * Semantic Memory Indexer for NanoClaw
 * Scans all group .md files, chunks them, embeds via OpenAI, stores in SQLite.
 * Provides hybrid search (vector + FTS5 keyword + RRF fusion).
 */
import Database from 'better-sqlite3';
import fs from 'fs';
import path from 'path';

import {
  DATA_DIR,
  GROUPS_DIR,
  MEMORY_CHUNK_OVERLAP,
  MEMORY_CHUNK_SIZE,
  MEMORY_INDEX_INTERVAL,
  OPENAI_API_KEY,
} from './config.js';
import { logger } from './logger.js';

const EMBEDDINGS_DB_PATH = path.join(DATA_DIR, 'embeddings.db');
const EMBEDDING_MODEL = 'text-embedding-3-large';
const EMBEDDING_DIMS = 3072;

let db: Database.Database | null = null;
let indexerRunning = false;

// ─── Database ────────────────────────────────────────────────────────────────

export function closeEmbeddingsDb(): void {
  if (db) {
    db.close();
    db = null;
  }
}

export function openEmbeddingsDb(): Database.Database {
  if (db) return db;

  fs.mkdirSync(path.dirname(EMBEDDINGS_DB_PATH), { recursive: true });

  db = new Database(EMBEDDINGS_DB_PATH);
  db.pragma('journal_mode = WAL');
  db.pragma('busy_timeout = 5000');

  db.exec(`
    CREATE TABLE IF NOT EXISTS chunks (
      id TEXT PRIMARY KEY,
      source TEXT NOT NULL,
      group_folder TEXT NOT NULL,
      type TEXT NOT NULL,
      content TEXT NOT NULL,
      embedding BLOB NOT NULL,
      parent_id TEXT,
      chunk_index INTEGER NOT NULL,
      timestamp TEXT NOT NULL,
      indexed_at TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source);

    CREATE TABLE IF NOT EXISTS index_state (
      source TEXT PRIMARY KEY,
      mtime_ms INTEGER NOT NULL,
      chunk_count INTEGER NOT NULL
    );
  `);

  // FTS5 table — create only if it doesn't exist
  try {
    db.exec(
      `CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(id UNINDEXED, content);`,
    );
  } catch {
    // FTS5 table already exists with different schema — ignore
  }

  return db;
}

// ─── File Scanning ───────────────────────────────────────────────────────────

const EXCLUDE_DIRS = new Set(['logs', 'node_modules', '.git', 'dist', 'data']);

function scanMarkdownFiles(dir: string, basePath = ''): string[] {
  const results: string[] = [];

  let entries: fs.Dirent[];
  try {
    entries = fs.readdirSync(dir, { withFileTypes: true });
  } catch {
    return results;
  }

  for (const entry of entries) {
    if (entry.name.startsWith('.')) continue;

    if (entry.isDirectory()) {
      if (EXCLUDE_DIRS.has(entry.name)) continue;
      results.push(
        ...scanMarkdownFiles(
          path.join(dir, entry.name),
          path.join(basePath, entry.name),
        ),
      );
    } else if (entry.isFile() && entry.name.endsWith('.md')) {
      results.push(path.join(basePath, entry.name));
    }
  }

  return results;
}

function detectType(
  filePath: string,
): 'identity' | 'conversation' | 'daily_note' | 'document' {
  const name = path.basename(filePath);
  if (name === 'CLAUDE.md') return 'identity';
  if (filePath.includes('conversations/')) return 'conversation';
  // Dated files like 2026-02-06-dinner.md
  if (/\d{4}-\d{2}-\d{2}/.test(name)) return 'daily_note';
  return 'document';
}

// ─── Chunking ────────────────────────────────────────────────────────────────

function chunkText(text: string, maxSize: number, overlap: number): string[] {
  if (text.length <= maxSize) return [text];

  const paragraphs = text.split(/\n\n+/);
  const chunks: string[] = [];
  let current = '';

  for (const para of paragraphs) {
    if (current.length + para.length + 2 > maxSize && current.length > 0) {
      chunks.push(current.trim());
      // Overlap: keep the last `overlap` chars of current as start of next chunk
      const overlapStart = current.length - overlap;
      current =
        overlapStart > 0
          ? current.slice(overlapStart).trim() + '\n\n' + para
          : para;
    } else {
      current = current ? current + '\n\n' + para : para;
    }
  }

  if (current.trim()) {
    chunks.push(current.trim());
  }

  return chunks;
}

// ─── Embedding ───────────────────────────────────────────────────────────────

async function embedTexts(
  texts: string[],
  maxRetries = 3,
): Promise<Float32Array[]> {
  if (texts.length === 0) return [];

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    const response = await fetch('https://api.openai.com/v1/embeddings', {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${OPENAI_API_KEY}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: EMBEDDING_MODEL,
        input: texts,
        encoding_format: 'float',
      }),
      signal: AbortSignal.timeout(15000),
    });

    if (response.status === 429 && attempt < maxRetries) {
      // Parse retry-after hint from error body, fall back to exponential backoff
      let waitMs = 1000 * 2 ** attempt;
      try {
        const body = (await response.json()) as any;
        const msg: string = body?.error?.message || '';
        const match = msg.match(/try again in ([\d.]+)s/i);
        if (match) waitMs = Math.ceil(parseFloat(match[1]) * 1000) + 200;
      } catch {
        /* ignore parse errors */
      }
      logger.warn(
        { attempt: attempt + 1, waitMs },
        'Embedding rate limited, retrying',
      );
      await new Promise((r) => setTimeout(r, waitMs));
      continue;
    }

    if (!response.ok) {
      const body = await response.text();
      throw new Error(`OpenAI embedding API error ${response.status}: ${body}`);
    }

    const json = (await response.json()) as {
      data: Array<{ embedding: number[] }>;
    };

    return json.data.map((d) => new Float32Array(d.embedding));
  }

  throw new Error('Embedding request failed after max retries');
}

// ─── Indexing ────────────────────────────────────────────────────────────────

async function indexGroup(
  database: Database.Database,
  groupFolder: string,
): Promise<number> {
  const groupDir = path.join(GROUPS_DIR, groupFolder);
  if (!fs.existsSync(groupDir)) return 0;

  const mdFiles = scanMarkdownFiles(groupDir);
  let chunksIndexed = 0;

  // Check which files need re-indexing
  const getState = database.prepare(
    'SELECT mtime_ms FROM index_state WHERE source = ?',
  );
  const upsertState = database.prepare(
    'INSERT OR REPLACE INTO index_state (source, mtime_ms, chunk_count) VALUES (?, ?, ?)',
  );
  const deleteChunks = database.prepare('DELETE FROM chunks WHERE source = ?');
  const deleteFts = database.prepare(
    'DELETE FROM chunks_fts WHERE id IN (SELECT id FROM chunks WHERE source = ?)',
  );
  const insertChunk = database.prepare(
    'INSERT OR REPLACE INTO chunks (id, source, group_folder, type, content, embedding, parent_id, chunk_index, timestamp, indexed_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
  );
  const insertFts = database.prepare(
    'INSERT INTO chunks_fts (id, content) VALUES (?, ?)',
  );

  const filesToIndex: Array<{
    relPath: string;
    absPath: string;
    mtimeMs: number;
  }> = [];

  for (const relPath of mdFiles) {
    const absPath = path.join(groupDir, relPath);
    let stat: fs.Stats;
    try {
      stat = fs.statSync(absPath);
    } catch {
      continue;
    }

    const source = `${groupFolder}/${relPath}`;
    const existing = getState.get(source) as { mtime_ms: number } | undefined;

    if (!existing || existing.mtime_ms !== Math.floor(stat.mtimeMs)) {
      filesToIndex.push({
        relPath,
        absPath,
        mtimeMs: Math.floor(stat.mtimeMs),
      });
    }
  }

  if (filesToIndex.length === 0) return 0;

  // Batch all texts for embedding
  const allChunks: Array<{
    source: string;
    id: string;
    type: string;
    content: string;
    chunkIndex: number;
    parentId: string | null;
    timestamp: string;
    mtimeMs: number;
  }> = [];

  for (const file of filesToIndex) {
    let content: string;
    try {
      content = fs.readFileSync(file.absPath, 'utf-8');
    } catch {
      continue;
    }

    if (!content.trim()) continue;

    const source = `${groupFolder}/${file.relPath}`;
    const type = detectType(file.relPath);
    const chunks = chunkText(content, MEMORY_CHUNK_SIZE, MEMORY_CHUNK_OVERLAP);
    const timestamp = new Date(file.mtimeMs).toISOString();

    for (let i = 0; i < chunks.length; i++) {
      allChunks.push({
        source,
        id: `${source}:chunk-${i}`,
        type,
        content: chunks[i],
        chunkIndex: i,
        parentId: chunks.length > 1 ? source : null,
        timestamp,
        mtimeMs: file.mtimeMs,
      });
    }
  }

  if (allChunks.length === 0) return 0;

  // Embed in small batches with pauses to stay under OpenAI's 1M TPM limit.
  // 20 chunks × ~850 tokens/chunk ≈ 17K tokens per batch.
  const BATCH_SIZE = 20;
  const BATCH_DELAY_MS = 5000;
  const allEmbeddings: Float32Array[] = [];

  for (let i = 0; i < allChunks.length; i += BATCH_SIZE) {
    if (i > 0) await new Promise((r) => setTimeout(r, BATCH_DELAY_MS));
    const batch = allChunks.slice(i, i + BATCH_SIZE);
    const embeddings = await embedTexts(batch.map((c) => c.content));
    allEmbeddings.push(...embeddings);
  }

  // Write to DB in a transaction
  const txn = database.transaction(() => {
    // Track which sources we're updating to clean old chunks
    const sourcesToClean = new Set(allChunks.map((c) => c.source));
    for (const source of sourcesToClean) {
      // Delete FTS entries first (before chunks are removed)
      deleteFts.run(source);
      deleteChunks.run(source);
    }

    const now = new Date().toISOString();
    for (let i = 0; i < allChunks.length; i++) {
      const chunk = allChunks[i];
      const embeddingBuf = Buffer.from(allEmbeddings[i].buffer);

      insertChunk.run(
        chunk.id,
        chunk.source,
        groupFolder,
        chunk.type,
        chunk.content,
        embeddingBuf,
        chunk.parentId,
        chunk.chunkIndex,
        chunk.timestamp,
        now,
      );
      insertFts.run(chunk.id, chunk.content);
    }

    // Update index state per source
    const chunkCounts = new Map<string, number>();
    for (const chunk of allChunks) {
      chunkCounts.set(chunk.source, (chunkCounts.get(chunk.source) || 0) + 1);
    }
    for (const [source, count] of chunkCounts) {
      const fileChunk = allChunks.find((c) => c.source === source)!;
      upsertState.run(source, fileChunk.mtimeMs, count);
    }
  });

  txn();
  chunksIndexed = allChunks.length;

  // Clean stale entries (files that no longer exist)
  const validSources = new Set(mdFiles.map((f) => `${groupFolder}/${f}`));
  const allIndexed = database
    .prepare('SELECT source FROM index_state WHERE source LIKE ?')
    .all(`${groupFolder}/%`) as Array<{ source: string }>;

  const staleCleanup = database.transaction(() => {
    for (const row of allIndexed) {
      if (!validSources.has(row.source)) {
        deleteFts.run(row.source);
        deleteChunks.run(row.source);
        database
          .prepare('DELETE FROM index_state WHERE source = ?')
          .run(row.source);
      }
    }
  });
  staleCleanup();

  return chunksIndexed;
}

// ─── Search ──────────────────────────────────────────────────────────────────

function cosineSimilarity(a: Float32Array, b: Float32Array): number {
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

interface SearchResult {
  id: string;
  source: string;
  group_folder: string;
  type: string;
  content: string;
  score: number;
}

function semanticSearch(
  database: Database.Database,
  queryEmbedding: Float32Array,
  limit: number,
): SearchResult[] {
  const rows = database
    .prepare(
      'SELECT id, source, group_folder, type, content, embedding FROM chunks',
    )
    .all() as Array<{
    id: string;
    source: string;
    group_folder: string;
    type: string;
    content: string;
    embedding: Buffer;
  }>;

  const scored = rows.map((row) => {
    const embedding = new Float32Array(
      row.embedding.buffer,
      row.embedding.byteOffset,
      row.embedding.byteLength / 4,
    );
    return {
      id: row.id,
      source: row.source,
      group_folder: row.group_folder,
      type: row.type,
      content: row.content,
      score: cosineSimilarity(queryEmbedding, embedding),
    };
  });

  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, limit);
}

function keywordSearch(
  database: Database.Database,
  query: string,
  limit: number,
): SearchResult[] {
  // Escape FTS5 special chars and create a simple query
  const sanitized = query.replace(/['"(){}[\]*:^~!]/g, ' ').trim();
  if (!sanitized) return [];

  try {
    const rows = database
      .prepare(
        `SELECT c.id, c.source, c.group_folder, c.type, c.content,
              rank * -1 as score
       FROM chunks_fts f
       JOIN chunks c ON c.id = f.id
       WHERE chunks_fts MATCH ?
       ORDER BY rank
       LIMIT ?`,
      )
      .all(sanitized, limit) as SearchResult[];

    return rows;
  } catch {
    return [];
  }
}

function rrfFusion(
  semantic: SearchResult[],
  keyword: SearchResult[],
  limit: number,
  k = 60,
): SearchResult[] {
  const scores = new Map<string, { score: number; result: SearchResult }>();

  for (let i = 0; i < semantic.length; i++) {
    const r = semantic[i];
    const rrf = 1 / (k + i + 1);
    scores.set(r.id, { score: rrf, result: r });
  }

  for (let i = 0; i < keyword.length; i++) {
    const r = keyword[i];
    const rrf = 1 / (k + i + 1);
    const existing = scores.get(r.id);
    if (existing) {
      existing.score += rrf;
    } else {
      scores.set(r.id, { score: rrf, result: r });
    }
  }

  const fused = Array.from(scores.values());
  fused.sort((a, b) => b.score - a.score);

  return fused.slice(0, limit).map((f) => ({
    ...f.result,
    score: f.score,
  }));
}

export async function searchMemory(
  query: string,
  mode: 'hybrid' | 'semantic' | 'keyword' = 'hybrid',
  limit = 5,
): Promise<SearchResult[]> {
  const database = openEmbeddingsDb();

  // Check if we have any chunks
  const count = database.prepare('SELECT COUNT(*) as c FROM chunks').get() as {
    c: number;
  };
  if (count.c === 0) return [];

  if (mode === 'keyword') {
    return keywordSearch(database, query, limit);
  }

  const [queryEmbedding] = await embedTexts([query]);

  if (mode === 'semantic') {
    return semanticSearch(database, queryEmbedding, limit);
  }

  // Hybrid: RRF fusion of both
  const semanticResults = semanticSearch(database, queryEmbedding, limit * 2);
  const keywordResults = keywordSearch(database, query, limit * 2);
  return rrfFusion(semanticResults, keywordResults, limit);
}

// ─── Indexer Loop ────────────────────────────────────────────────────────────

export function startMemoryIndexer(getGroupFolders: () => string[]): void {
  if (indexerRunning) {
    logger.warn('Memory indexer already running, skipping duplicate start');
    return;
  }
  indexerRunning = true;

  const runIndex = async () => {
    try {
      const database = openEmbeddingsDb();
      const folders = getGroupFolders();
      let totalChunks = 0;

      for (const folder of folders) {
        const count = await indexGroup(database, folder);
        totalChunks += count;
      }

      if (totalChunks > 0) {
        logger.info(
          { totalChunks, groups: folders.length },
          'Memory indexing complete',
        );
      }
    } catch (err) {
      logger.error({ err }, 'Memory indexing error');
    }
  };

  // Run immediately, then on interval
  runIndex();
  setInterval(runIndex, MEMORY_INDEX_INTERVAL);
  logger.info({ intervalMs: MEMORY_INDEX_INTERVAL }, 'Memory indexer started');
}
