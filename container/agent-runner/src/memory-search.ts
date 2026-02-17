/**
 * Container-side semantic memory search for NanoClaw
 * Reads the shared embeddings.db (mounted read-only) and performs
 * hybrid search (vector cosine sim + FTS5 keyword + RRF fusion).
 */
import Database from 'better-sqlite3';
import fs from 'fs';

const EMBEDDINGS_DB_PATH = '/workspace/embeddings.db';
const EMBEDDING_MODEL = 'text-embedding-3-large';

interface SearchResult {
  id: string;
  source: string;
  group_folder: string;
  type: string;
  content: string;
  score: number;
}

function cosineSimilarity(a: Float32Array, b: Float32Array): number {
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  if (normA === 0 || normB === 0) return 0;
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

async function embedQuery(query: string): Promise<Float32Array> {
  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) throw new Error('OPENAI_API_KEY not set');

  const response = await fetch('https://api.openai.com/v1/embeddings', {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${apiKey}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model: EMBEDDING_MODEL,
      input: [query],
      encoding_format: 'float',
    }),
    signal: AbortSignal.timeout(15000),
  });

  if (!response.ok) {
    const body = await response.text();
    throw new Error(`OpenAI embedding API error ${response.status}: ${body}`);
  }

  const json = (await response.json()) as {
    data: Array<{ embedding: number[] }>;
  };

  return new Float32Array(json.data[0].embedding);
}

function semanticSearch(
  db: Database.Database,
  queryEmbedding: Float32Array,
  limit: number,
  groupFolder?: string,
): SearchResult[] {
  const rows = (groupFolder
    ? db.prepare('SELECT id, source, group_folder, type, content, embedding FROM chunks WHERE group_folder = ?').all(groupFolder)
    : db.prepare('SELECT id, source, group_folder, type, content, embedding FROM chunks').all()
  ) as Array<{
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
  db: Database.Database,
  query: string,
  limit: number,
  groupFolder?: string,
): SearchResult[] {
  const sanitized = query.replace(/['"(){}[\]*:^~!]/g, ' ').trim();
  if (!sanitized) return [];

  try {
    const rows = groupFolder
      ? db.prepare(
          `SELECT c.id, c.source, c.group_folder, c.type, c.content,
                  rank * -1 as score
           FROM chunks_fts f
           JOIN chunks c ON c.id = f.id
           WHERE chunks_fts MATCH ? AND c.group_folder = ?
           ORDER BY rank
           LIMIT ?`,
        ).all(sanitized, groupFolder, limit) as SearchResult[]
      : db.prepare(
          `SELECT c.id, c.source, c.group_folder, c.type, c.content,
                  rank * -1 as score
           FROM chunks_fts f
           JOIN chunks c ON c.id = f.id
           WHERE chunks_fts MATCH ?
           ORDER BY rank
           LIMIT ?`,
        ).all(sanitized, limit) as SearchResult[];

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

export async function search(
  query: string,
  mode: 'hybrid' | 'semantic' | 'keyword' = 'hybrid',
  limit = 5,
  groupFolder?: string,
): Promise<SearchResult[]> {
  if (!fs.existsSync(EMBEDDINGS_DB_PATH)) {
    return [];
  }

  const db = new Database(EMBEDDINGS_DB_PATH, { readonly: true });
  db.pragma('busy_timeout = 5000');

  try {
    const count = db.prepare('SELECT COUNT(*) as c FROM chunks').get() as {
      c: number;
    };
    if (count.c === 0) return [];

    if (mode === 'keyword') {
      return keywordSearch(db, query, limit, groupFolder);
    }

    const queryEmbedding = await embedQuery(query);

    if (mode === 'semantic') {
      return semanticSearch(db, queryEmbedding, limit, groupFolder);
    }

    // Hybrid: RRF fusion
    const semanticResults = semanticSearch(db, queryEmbedding, limit * 2, groupFolder);
    const keywordResults = keywordSearch(db, query, limit * 2, groupFolder);
    return rrfFusion(semanticResults, keywordResults, limit);
  } finally {
    db.close();
  }
}
