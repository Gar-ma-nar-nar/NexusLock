import { cosine, bestMatch, l2Normalize, averageEmbeddings, SavedFace } from './NexusLockPure';

describe('cosine', () => {
  it('returns 1 for identical vectors', () => expect(cosine([1,0,0],[1,0,0])).toBeCloseTo(1));
  it('returns 0 for orthogonal',       () => expect(cosine([1,0],[0,1])).toBeCloseTo(0));
  it('returns -1 for opposite',        () => expect(cosine([1,0],[-1,0])).toBeCloseTo(-1));
  it('returns 0 for empty',            () => expect(cosine([],[])).toBe(0));
  it('returns 0 for mismatched',       () => expect(cosine([1,2],[1,2,3])).toBe(0));
  it('returns 0 for zero vectors',     () => expect(cosine([0,0],[0,0])).toBe(0));
});

describe('l2Normalize', () => {
  it('normalizes to unit length', () => {
    const r = l2Normalize([3,4]);
    expect(Math.sqrt(r[0]**2 + r[1]**2)).toBeCloseTo(1);
  });
  it('handles zero vector', () => {
    const r = l2Normalize([0,0,0]);
    expect(r).toEqual([0,0,0]);
  });
  it('returns copy', () => {
    const orig = [0,0];
    expect(l2Normalize(orig)).not.toBe(orig);
  });
});

describe('averageEmbeddings', () => {
  it('returns [] for empty input', () => {
    expect(averageEmbeddings([])).toEqual([]);
  });
  it('returns normalized single embedding', () => {
    const r = averageEmbeddings([[3, 4]]);
    expect(Math.sqrt(r[0]**2 + r[1]**2)).toBeCloseTo(1);
  });
  it('averages multiple and normalizes', () => {
    const r = averageEmbeddings([[1, 0], [0, 1]]);
    expect(r[0]).toBeCloseTo(r[1]); // should be equal (both ~0.707)
    expect(Math.sqrt(r[0]**2 + r[1]**2)).toBeCloseTo(1);
  });
});

describe('bestMatch', () => {
  const face = (name: string, emb: number[], list: 'trusted'|'target' = 'trusted'): SavedFace =>
    ({ id: name, name, list, photoUri: '', embedding: emb, engine: 'facenet', createdAt: 0 });

  it('null for empty list',    () => expect(bestMatch([1,0], [], 0.5)).toBeNull());
  it('null below threshold',   () => expect(bestMatch([1,0], [face('A',[0,1])], 0.99)).toBeNull());
  it('finds best above threshold', () => {
    const r = bestMatch(l2Normalize([1,0]), [face('A',l2Normalize([1,0.1])), face('B',l2Normalize([1,0]))], 0.5);
    expect(r!.face.name).toBe('B');
  });
  it('skips mismatched dims',  () => expect(bestMatch([1,0], [face('A',[1,0,0])], 0.5)).toBeNull());
});
