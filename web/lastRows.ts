/**
 * The decoder's key/value caches without copying them out every step (task 0023 of the host app). AGPL-3.0, like
 * homr (../LICENSE).
 *
 * homr's decoder hands back all 32 caches on every step (`cache_out<i>`, [1, 8, rows, 64]) and takes them in again on
 * the next. Every step adds one row to each, and the rows before stay exactly as they were. Sixteen of the caches are
 * the cross-attention ones with 1 280 rows, so each step copied about 42 MB out of the runtime into new arrays:
 * gigabytes of garbage per page, which is what drove a phone's memory up.
 *
 * Two parts:
 * - `withLastRows` adds one output per cache to the decoder model, `cache_new<i>`: its last row (a Slice). It
 *   appends a second `graph` field to the model's bytes. Protocol buffers merge a message field that appears twice,
 *   so the decoder's graph gains 32 nodes, 32 outputs and 3 constants, and nothing else changes.
 * - `GrowingCache` keeps a cache in one buffer of its own and adds the new row each step. The decoder then gets
 *   the same numbers as before: tested bit for bit (scripts/scratch/homr-rows-check.mts).
 */

/** How many caches the decoder has: four per layer (self and cross attention, keys and values), eight layers. */
export const CACHES = 32;

function varint(n: number | bigint): number[] {
  let v = BigInt.asUintN(64, BigInt(n));
  const out: number[] = [];
  do {
    let b = Number(v & 0x7fn);
    v >>= 7n;
    if (v) b |= 0x80;
    out.push(b);
  } while (v);
  return out;
}

const utf8 = (s: string): number[] => Array.from(new TextEncoder().encode(s));
/** A length-delimited field (strings, bytes, messages). */
const field = (n: number, bytes: readonly number[]): number[] => [...varint((n << 3) | 2), ...varint(bytes.length), ...bytes];
/** A varint field. */
const int = (n: number, v: number | bigint): number[] => [...varint(n << 3), ...varint(v)];

function int64Tensor(name: string, value: bigint): number[] {
  const raw = new Uint8Array(8);
  new DataView(raw.buffer).setBigInt64(0, value, true);
  // TensorProto: dims (1), data_type (2: 7 is int64), name (8), raw_data (9).
  return [...int(1, 1), ...int(2, 7), ...field(8, utf8(name)), ...field(9, Array.from(raw))];
}

function floatOutput(name: string, dims: readonly number[]): number[] {
  // ValueInfoProto: name (1), type (2) = TypeProto { tensor_type (1) = { elem_type (1): 1 is float, shape (2) } }.
  const shape = dims.flatMap((d) => field(1, int(1, d)));
  return [...field(1, utf8(name)), ...field(2, field(1, [...int(1, 1), ...field(2, shape)]))];
}

/** The decoder model with `cache_new<i>` (the last row of `cache_out<i>`) as extra outputs. */
export function withLastRows(model: Uint8Array, caches: number = CACHES, heads: number = 8, headDim: number = 64): Uint8Array {
  const names = { start: 'lastrow_start', end: 'lastrow_end', axis: 'lastrow_axis' };
  const graph: number[] = [];
  for (let i = 0; i < caches; i++) {
    // NodeProto: input (1), output (2), name (3), op_type (4).
    const node = [...field(1, utf8(`cache_out${i}`)), ...field(1, utf8(names.start)), ...field(1, utf8(names.end)), ...field(1, utf8(names.axis)), ...field(2, utf8(`cache_new${i}`)), ...field(3, utf8(`lastrow_${i}`)), ...field(4, utf8('Slice'))];
    graph.push(...field(1, node));
  }
  // GraphProto: node (1), initializer (5), output (12).
  graph.push(...field(5, int64Tensor(names.start, -1n)), ...field(5, int64Tensor(names.end, 0x7fffffffffffffffn)), ...field(5, int64Tensor(names.axis, 2n)));
  for (let i = 0; i < caches; i++) graph.push(...field(12, floatOutput(`cache_new${i}`, [1, heads, 1, headDim])));
  // ModelProto: graph (7), a second time; merged into the first.
  const tail = field(7, graph);
  const out = new Uint8Array(model.byteLength + tail.length);
  out.set(model);
  out.set(tail, model.byteLength);
  return out;
}

/** True when the decoder hands back the caches' last rows (`withLastRows`): its outputs' names. */
export function hasLastRows(outputNames: readonly string[]): boolean {
  return outputNames.includes('cache_new0') && outputNames.includes(`cache_new${CACHES - 1}`);
}

/**
 * One key/value cache, [1, heads, rows, headDim], in a buffer that grows: each step one row per head is added at
 * the end of that head's rows. `view` is the cache as the decoder takes it, without a copy.
 */
export class GrowingCache {
  private buffer: Float32Array;
  private rows: number;

  constructor(
    readonly heads: number,
    readonly headDim: number,
    first: Float32Array,
    rows: number,
    room?: number,
  ) {
    this.rows = rows;
    this.buffer = new Float32Array(heads * (rows + Math.max(1, room ?? 64)) * headDim);
    this.buffer.set(first.subarray(0, heads * rows * headDim));
  }

  get length(): number {
    return this.rows;
  }

  /** Adds `row` ([1, heads, 1, headDim]): every head's block moves up by the heads before it, its new row at its end. */
  append(row: ArrayLike<number>): void {
    const { heads, headDim: d } = this;
    const n = this.rows;
    if (heads * (n + 1) * d > this.buffer.length) {
      // An eighth more (at least 64 rows): a cross-attention cache of 1 280 rows grows once in a long staff.
      const bigger = new Float32Array(heads * (n + Math.max(64, n >> 3)) * d);
      bigger.set(this.buffer.subarray(0, heads * n * d));
      this.buffer = bigger;
    }
    const b = this.buffer;
    for (let h = heads - 1; h >= 0; h--) {
      b.copyWithin(h * (n + 1) * d, h * n * d, (h + 1) * n * d);
      for (let k = 0; k < d; k++) b[(h * (n + 1) + n) * d + k] = row[h * d + k]!;
    }
    this.rows = n + 1;
  }

  view(): Float32Array {
    return this.buffer.subarray(0, this.heads * this.rows * this.headDim);
  }
}
