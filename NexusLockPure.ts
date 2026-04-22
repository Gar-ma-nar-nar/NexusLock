/* Pure functions — testable without React Native */

export type FaceList = 'trusted' | 'target';
export type Embedding = number[];

export interface SavedFace {
  id: string;
  name: string;
  list: FaceList;
  photoUri: string;
  embedding: Embedding;
  engine: 'facenet';
  createdAt: number;
}

export function cosine(a: number[], b: number[]): number {
  if (a.length !== b.length || a.length === 0) return 0;
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na  += a[i] * a[i];
    nb  += b[i] * b[i];
  }
  const d = Math.sqrt(na) * Math.sqrt(nb);
  return d === 0 ? 0 : dot / d;
}

export function bestMatch(
  sig: Embedding, faces: SavedFace[], threshold: number,
): { face: SavedFace; score: number } | null {
  let best: { face: SavedFace; score: number } | null = null;
  for (const f of faces) {
    if (f.embedding.length !== sig.length) continue;
    const s = cosine(sig, f.embedding);
    if (!best || s > best.score) best = { face: f, score: s };
  }
  return best && best.score >= threshold ? best : null;
}

export function l2Normalize(v: number[]): number[] {
  const n = Math.sqrt(v.reduce((s, x) => s + x * x, 0));
  return n === 0 ? v.slice() : v.map(x => x / n);
}

export function averageEmbeddings(embeddings: Embedding[]): Embedding {
  if (embeddings.length === 0) return [];
  if (embeddings.length === 1) return l2Normalize(embeddings[0]);
  const dim = embeddings[0].length;
  const avg = new Array(dim).fill(0);
  for (const e of embeddings) {
    for (let i = 0; i < dim; i++) avg[i] += e[i];
  }
  for (let i = 0; i < dim; i++) avg[i] /= embeddings.length;
  return l2Normalize(avg);
}
