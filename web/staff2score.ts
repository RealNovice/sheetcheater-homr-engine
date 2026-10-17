/**
 * homr's transformer inference (homr/transformer/staff2score.py, encoder_inference.py, decoder_inference.py),
 * ported to TypeScript over onnxruntime-web. AGPL-3.0, like homr (../LICENSE).
 *
 * The encoder reads one staff image (1 × 1 × 256 × 1280); the decoder then writes the staff's symbols one at a
 * time with a key/value cache, greedily, each symbol in six branches (rhythm, pitch, lift, position,
 * articulation, slur) until the end token.
 */
import vocabulary from './vocabulary.json';

/** The part of onnxruntime-web's API used here (the browser's `ort` global, or the npm package in Node). */
export interface OrtLike {
  Tensor: new (type: 'float32' | 'int64', data: Float32Array | BigInt64Array, dims: readonly number[]) => OrtTensor;
}
export interface OrtTensor {
  readonly data: Float32Array | BigInt64Array | ArrayLike<number>;
  readonly dims: readonly number[];
}
export interface OrtSessionLike {
  run(feeds: Record<string, OrtTensor>): Promise<Record<string, OrtTensor>>;
}

/** One symbol as homr's EncodedSymbol: the six branches, as their token text. */
export interface HomrSymbol {
  readonly rhythm: string;
  readonly pitch: string;
  readonly lift: string;
  readonly position: string;
  readonly articulation: string;
  readonly slur: string;
  /**
   * How sure the decoder was (softmax of the chosen token) in the branches that make a note: its length, its
   * pitch and its accidental, and its runner-up pitch and accidental with their probabilities (task 0016).
   */
  readonly conf?: HomrConfidence;
}

export interface HomrConfidence {
  readonly rhythm: number;
  readonly pitch: number;
  readonly lift: number;
  readonly pitch2: string;
  readonly pitch2p: number;
  readonly lift2: string;
  readonly lift2p: number;
}

export const VOCAB = vocabulary as { rhythm: string[]; pitch: string[]; lift: string[]; articulation: string[]; slur: string[]; position: string[]; config: { heads: number; dim: number; depth: number; maxSeqLen: number; eos: number; height: number; width: number } };

function argmax(data: ArrayLike<number>): number {
  let best = 0;
  for (let i = 1; i < data.length; i++) if (data[i]! > data[best]!) best = i;
  return best;
}

/** Softmax probabilities of the best and the second-best token of a branch's logits. */
export function topTwo(data: ArrayLike<number>): { best: number; p: number; second: number; p2: number } {
  const best = argmax(data);
  let second = best === 0 ? 1 : 0;
  for (let i = 0; i < data.length; i++) if (i !== best && data[i]! > data[second]!) second = i;
  const max = data[best]!;
  let sum = 0;
  for (let i = 0; i < data.length; i++) sum += Math.exp(data[i]! - max);
  return { best, p: 1 / sum, second, p2: Math.exp(data[second]! - max) / sum };
}

const BRANCHES = [
  ['out_rhythms', 'rhythm'],
  ['out_pitchs', 'pitch'],
  ['out_lifts', 'lift'],
  ['out_positions', 'position'],
  ['out_articulations', 'articulation'],
  ['out_slurs', 'slur'],
] as const;

const round = (p: number): number => Math.round(p * 1e4) / 1e4;

/** Reads one staff image: the symbols homr's transformer writes for it. */
export async function readStaff(ort: OrtLike, encoder: OrtSessionLike, decoder: OrtSessionLike, input: Float32Array, maxSteps: number = VOCAB.config.maxSeqLen): Promise<HomrSymbol[]> {
  const { heads, dim, depth, eos } = VOCAB.config;
  const headDim = dim / heads;
  const enc = await encoder.run({ input: new ort.Tensor('float32', input, [1, 1, VOCAB.config.height, VOCAB.config.width]) });
  const context = enc.output!;
  const contextData = context.data as Float32Array;
  // After the first step only the context's first row goes in (homr: x_transformers' split of the context).
  const reduced = new ort.Tensor('float32', contextData.slice(0, dim), [1, 1, dim]);
  let cache: OrtTensor[] = Array.from({ length: depth * 4 }, () => new ort.Tensor('float32', new Float32Array(0), [1, heads, 0, headDim]));
  const id = (n: number): OrtTensor => new ort.Tensor('int64', BigInt64Array.from([BigInt(n)]), [1, 1]);
  let last = { rhythm: 1, pitch: 0, lift: 0, articulation: 0, slur: 0 };
  const out: HomrSymbol[] = [];
  for (let step = 0; step < maxSteps; step++) {
    const feeds: Record<string, OrtTensor> = {
      rhythms: id(last.rhythm),
      pitchs: id(last.pitch),
      lifts: id(last.lift),
      articulations: id(last.articulation),
      slurs: id(last.slur),
      context: step === 0 ? context : reduced,
      cache_len: new ort.Tensor('int64', BigInt64Array.from([BigInt(step)]), [1]),
    };
    cache.forEach((c, i) => (feeds[`cache_in${i}`] = c));
    const r = await decoder.run(feeds);
    cache = cache.map((_, i) => r[`cache_out${i}`]!);
    const pick = Object.fromEntries(BRANCHES.map(([name, branch]) => [branch, argmax(r[name]!.data as ArrayLike<number>)])) as Record<(typeof BRANCHES)[number][1], number>;
    if (pick.rhythm === eos) break;
    const rhythm = topTwo(r.out_rhythms!.data as ArrayLike<number>);
    const pitch = topTwo(r.out_pitchs!.data as ArrayLike<number>);
    const lift = topTwo(r.out_lifts!.data as ArrayLike<number>);
    const conf: HomrConfidence = { rhythm: round(rhythm.p), pitch: round(pitch.p), lift: round(lift.p), pitch2: VOCAB.pitch[pitch.second]!, pitch2p: round(pitch.p2), lift2: VOCAB.lift[lift.second]!, lift2p: round(lift.p2) };
    out.push({ rhythm: VOCAB.rhythm[pick.rhythm]!, pitch: VOCAB.pitch[pick.pitch]!, lift: VOCAB.lift[pick.lift]!, position: VOCAB.position[pick.position]!, articulation: VOCAB.articulation[pick.articulation]!, slur: VOCAB.slur[pick.slur]!, conf });
    last = { rhythm: pick.rhythm, pitch: pick.pitch, lift: pick.lift, articulation: pick.articulation, slur: pick.slur };
  }
  return out;
}
