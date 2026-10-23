// Tests for the decoder's growing caches (AGPL-3.0, ../LICENSE).
import { describe, expect, it } from 'vitest';

import { GrowingCache, hasLastRows, withLastRows } from './lastRows';
import { readStaff } from './staff2score';

/** A cache [1, heads, rows, d] whose value at (h, j, k) is h * 1000 + j * 10 + k. */
function cache(heads: number, rows: number, d: number): Float32Array {
  const out = new Float32Array(heads * rows * d);
  for (let h = 0; h < heads; h++) for (let j = 0; j < rows; j++) for (let k = 0; k < d; k++) out[(h * rows + j) * d + k] = h * 1000 + j * 10 + k;
  return out;
}

describe('the decoder caches kept here (task 0023)', () => {
  it('adds one row per head at the end of that head, as the decoder would hand back the whole cache, and grows past its room', () => {
    const g = new GrowingCache(3, 4, cache(3, 5, 4), 5, 1);
    for (let rows = 6; rows <= 80; rows++) {
      const row = new Float32Array(3 * 4);
      for (let h = 0; h < 3; h++) for (let k = 0; k < 4; k++) row[h * 4 + k] = h * 1000 + (rows - 1) * 10 + k;
      g.append(row);
      expect(g.length).toBe(rows);
      expect(Array.from(g.view())).toEqual(Array.from(cache(3, rows, 4)));
    }
  });

  it('adds the last-row outputs to a model as a second graph field, the model before it untouched', () => {
    const model = Uint8Array.from([0x08, 0x08, 0x3a, 0x02, 0x0a, 0x00]);
    const out = withLastRows(model, 2);
    expect(Array.from(out.subarray(0, model.length))).toEqual(Array.from(model));
    // Field 7, length-delimited: the graph's tag.
    expect(out[model.length]).toBe((7 << 3) | 2);
    const text = new TextDecoder().decode(out);
    for (const name of ['cache_new0', 'cache_new1', 'lastrow_start', 'Slice']) expect(text).toContain(name);
    expect(text).not.toContain('cache_new2');
  });

  it('reads a staff with only the last rows coming out after the first step, the same symbols as with whole caches', async () => {
    class Tensor {
      constructor(
        readonly type: string,
        readonly data: Float32Array | BigInt64Array,
        readonly dims: readonly number[],
      ) {}
    }
    const ort = { Tensor };
    const encoder = { run: async () => ({ output: new Tensor('float32', new Float32Array(1280 * 512), [1, 1280, 512]) }) };
    // A decoder whose caches gain a row a step (cross caches start at 3 rows) and whose rhythm is the sum of the
    // caches' last values, so a cache that came back wrong changes the symbols. It ends after five steps.
    const make = (lastRows: boolean) => {
      const fetched: (readonly string[] | undefined)[] = [];
      return {
        fetched,
        run: async (feeds: Record<string, Tensor>, fetches?: readonly string[]) => {
          fetched.push(fetches);
          const step = Number((feeds.cache_len!.data as BigInt64Array)[0]);
          const out: Record<string, Tensor> = {};
          let sum = 0;
          for (let i = 0; i < 32; i++) {
            const prev = feeds[`cache_in${i}`]!;
            const rows = prev.dims[2]! + (step === 0 && i % 4 >= 2 ? 3 : 1);
            const data = new Float32Array(8 * rows * 64);
            for (let h = 0; h < 8; h++) {
              data.set((prev.data as Float32Array).subarray(h * prev.dims[2]! * 64, (h + 1) * prev.dims[2]! * 64), h * rows * 64);
              for (let j = prev.dims[2]!; j < rows; j++) for (let k = 0; k < 64; k++) data[(h * rows + j) * 64 + k] = i + h + j * 0.5 + k * 0.01 + step;
            }
            sum += data[data.length - 1]! + (prev.data as Float32Array).reduce((a, v) => a + v, 0) * 1e-6;
            out[`cache_out${i}`] = new Tensor('float32', data, [1, 8, rows, 64]);
            const last = new Float32Array(8 * 64);
            for (let h = 0; h < 8; h++) last.set(data.subarray((h * rows + rows - 1) * 64, (h * rows + rows) * 64), h * 64);
            if (lastRows) out[`cache_new${i}`] = new Tensor('float32', last, [1, 8, 1, 64]);
          }
          for (const [name, n] of [['out_rhythms', 259], ['out_pitchs', 72], ['out_lifts', 7], ['out_positions', 3], ['out_articulations', 54], ['out_slurs', 5]] as const) {
            const d = new Float32Array(n);
            d[name === 'out_rhythms' ? (step === 5 ? 2 : 3 + (Math.round(sum) % 200)) : Math.round(sum) % n] = 10;
            out[name] = new Tensor('float32', d, [1, 1, n]);
          }
          return out;
        },
      };
    };
    const whole = make(false);
    const rows = make(true);
    const a = await readStaff(ort as never, encoder as never, whole as never, new Float32Array(256 * 1280));
    const b = await readStaff(ort as never, encoder as never, rows as never, new Float32Array(256 * 1280));
    expect(a).toHaveLength(5);
    expect(b).toEqual(a);
    expect(hasLastRows(Array.from({ length: 32 }, (_, i) => `cache_new${i}`))).toBe(true);
    expect(hasLastRows(['cache_new0'])).toBe(false);
    expect(whole.fetched.every((f) => f === undefined)).toBe(true);
    expect(rows.fetched[0]).toBeUndefined();
    expect(rows.fetched.slice(1).every((f) => f!.includes('cache_new31') && !f!.some((n) => n.startsWith('cache_out')))).toBe(true);
  });
});
