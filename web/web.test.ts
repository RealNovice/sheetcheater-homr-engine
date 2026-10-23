// Tests for the browser port of homr (AGPL-3.0, ../LICENSE).
import { describe, expect, it } from 'vitest';

import { CANVAS, inkMap, joined, staffExtent, staffInput, type Gray } from './crop';
import { mergeStaves, readImage, shrinkGray } from './engine';
import { topTwo, type HomrSymbol } from './staff2score';
import { estimateSpacing, findStaves } from './staves';
import { kernDuration, timeline } from './tokens';

const sym = (rhythm: string, pitch = '.', lift = '_', position = 'upper'): HomrSymbol => ({ rhythm, pitch, lift, position, articulation: '_', slur: '_' });

/** A white page with one five-line staff from x = 100 to 900, spacing 10, top line at y = 100. */
function page(): Gray {
  const width = 1000;
  const height = 300;
  const data = new Uint8Array(width * height).fill(230);
  for (let k = 0; k < 5; k++) for (let x = 100; x <= 900; x++) data[(100 + 10 * k) * width + x] = 20;
  return { width, height, data };
}

/** A page of staves: each [top line y at x = 0, tilt, x0, x1], spacing 12, lines two pixels thick, a little noise. */
function staves(width: number, height: number, list: readonly (readonly [number, number, number, number])[]): Gray {
  const data = new Uint8Array(width * height);
  let seed = 7;
  for (let i = 0; i < data.length; i++) {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    data[i] = 215 + (seed % 20);
  }
  for (const [top, tilt, x0, x1] of list) {
    for (let k = 0; k < 5; k++) {
      for (let x = x0; x <= x1; x++) {
        const y = Math.round(top + 12 * k + x * tilt);
        for (const yy of [y, y + 1]) if (yy >= 0 && yy < height) data[yy * width + x] = 30;
      }
    }
  }
  return { width, height, data };
}

describe('homr in the browser', () => {
  it('reads kern durations as homr does: dots, tuplets, grace notes, whole-bar rests', () => {
    expect(kernDuration('4')).toMatchObject({ beats: 1, duration: { base: 4, dots: 0 } });
    expect(kernDuration('8.')).toMatchObject({ beats: 0.75, duration: { base: 8, dots: 1 } });
    expect(kernDuration('12')).toMatchObject({ duration: { base: 8, dots: 0, tuplet: [3, 2] } });
    expect(kernDuration('12').beats).toBeCloseTo(1 / 3);
    expect(kernDuration('8G')).toMatchObject({ beats: 0, grace: true });
    expect(kernDuration('0')).toMatchObject({ bar: true });
  });

  it('turns symbols into a timeline: chords, the shortest note moving on, a long note in a second voice, bars', () => {
    const t = timeline(
      [
        sym('clef_G2'),
        sym('keySignature_-1'),
        sym('timeSignature/4'),
        sym('note_2', 'C5'),
        sym('chord'),
        sym('note_4', 'E5'),
        sym('note_4', 'F5'),
        sym('note_2', 'B4', 'b'),
        sym('barline'),
        sym('rest_1'),
        sym('barline'),
      ],
      false,
    );
    expect(t.bars).toBe(2);
    expect(t.fifths).toBe(-1);
    expect(t.time).toEqual([4, 4]);
    expect(t.notes.filter((n) => !n.rest).map((n) => [n.bar, n.onset, n.midi, n.voice])).toEqual([
      [0, 0, 72, 1],
      [0, 0, 76, 2],
      [0, 1, 77, 2],
      [0, 2, 70, 1],
    ]);
    expect(t.notes.find((n) => n.midi === 70)!.printed).toBe(false);
  });

  it('puts a grand staff image’s lower notes on staff 2 and keeps a single stave’s lower notes out', () => {
    const symbols = [sym('clef_G2'), sym('clef_F4', '.', '_', 'lower'), sym('note_4', 'C5'), sym('chord'), sym('note_4', 'C3', '_', 'lower')];
    expect(timeline(symbols, true).notes.map((n) => [n.staff, n.midi, n.clef.sign])).toEqual([
      [1, 72, 'G'],
      [2, 48, 'F'],
    ]);
    expect(timeline(symbols, false).notes.map((n) => n.midi)).toEqual([72]);
  });

  it('finds where a staff runs and makes its 1280 × 256 image, white around it', () => {
    const g = page();
    const lines = [100, 110, 120, 130, 140];
    const ext = staffExtent(g, lines, 0, 10);
    expect(ext.x0).toBeGreaterThanOrEqual(85);
    expect(ext.x0).toBeLessThanOrEqual(105);
    expect(ext.x1).toBeGreaterThanOrEqual(895);
    const input = staffInput(g, { lines, tilt: 0, spacing: 10, ...ext }, []);
    expect(input.data).toHaveLength(CANVAS.width * CANVAS.height);
    const white = (1 - CANVAS.mean) / CANVAS.std;
    expect(input.data[0]).toBeCloseTo(white);
    expect(input.data.reduce((m, v) => Math.min(m, v), Infinity)).toBeLessThan(0);
  });

  it('joins two staves when a line runs down between them at their left end', () => {
    const g = page();
    expect(joined(g, { bottom: 140, x0: 100, spacing: 10 }, { top: 200, x0: 100 })).toBe(false);
    for (let y = 140; y <= 200; y++) g.data[y * g.width + 100] = 10;
    expect(joined(g, { bottom: 140, x0: 100, spacing: 10 }, { top: 200, x0: 100 })).toBe(true);
  });

  it('joins staves by a leaning line, not by the edge of a dark background, and finds ink in a shadow (task 0018)', () => {
    const lean = page();
    for (let y = 140; y <= 200; y++) lean.data[y * lean.width + 100 + Math.round((y - 140) / 8)] = 10;
    expect(joined(lean, { bottom: 140, x0: 100, spacing: 10 }, { top: 200, x0: 100 })).toBe(true);
    const edge = page();
    for (let y = 0; y < edge.height; y++) for (let x = 0; x <= 95; x++) edge.data[y * edge.width + x] = 30;
    expect(joined(edge, { bottom: 140, x0: 100, spacing: 10 }, { top: 200, x0: 100 })).toBe(false);
    const shade = page();
    for (let y = 0; y < shade.height; y++) for (let x = 0; x < shade.width; x++) shade.data[y * shade.width + x] = Math.round(shade.data[y * shade.width + x]! * 0.45);
    for (let y = 140; y <= 200; y++) shade.data[y * shade.width + 100] = 5;
    const ink = inkMap(shade, 10);
    expect(ink.data[170 * shade.width + 100]).toBe(0);
    expect(ink.data[170 * shade.width + 300]).toBe(255);
    expect(joined(ink, { bottom: 140, x0: 100, spacing: 10 }, { top: 200, x0: 100 })).toBe(true);
  });
  it('keeps how sure the decoder was of a note, and its runner-up pitch and accidental (task 0016)', () => {
    const t2 = topTwo([0, Math.log(3), Math.log(6)]);
    expect(t2).toMatchObject({ best: 2, second: 1 });
    expect(t2.p).toBeCloseTo(0.6);
    expect(t2.p2).toBeCloseTo(0.3);
    const conf = { rhythm: 0.9, pitch: 0.55, lift: 0.8, pitch2: 'B4', pitch2p: 0.4, lift2: '#', lift2p: 0.15 };
    const t = timeline([sym('clef_G2'), sym('keySignature_-1'), { ...sym('note_4', 'C5'), conf }, { ...sym('note_4', 'C5', 'b'), conf }], false);
    // C5 in F major: its runner-up B4 takes the key's flat; the C with a printed flat keeps it on the B.
    expect(t.notes[0]).toMatchObject({ p: { rhythm: 0.9, pitch: 0.55, lift: 0.8 }, alts: [{ midi: 70, diatonic: 34, p: 0.4 }, { midi: 73, diatonic: 35, p: 0.15 }] });
    expect(t.notes[1]!.alts![0]).toEqual({ midi: 70, diatonic: 34, p: 0.4 });
  });

  it('finds staves itself: tilted ones, and the two pages of an open book in turn (task 0018)', () => {
    const tilted = staves(1600, 900, [
      [100, 0.03, 100, 1500],
      [300, 0.03, 100, 1500],
      [500, 0.03, 100, 1500],
    ]);
    expect(estimateSpacing(tilted)).toBeGreaterThanOrEqual(11);
    expect(estimateSpacing(tilted)).toBeLessThanOrEqual(13);
    const found = findStaves(tilted, null);
    expect(found).toHaveLength(3);
    for (const [i, f] of found.entries()) {
      expect(f.tilt).toBeCloseTo(0.03, 2);
      expect(Math.abs(f.lines[0]! + 800 * f.tilt + f.bend[800]! - (100 + 200 * i + 800 * 0.03))).toBeLessThan(2);
      expect(f.spacing).toBeCloseTo(12, 0);
    }
    const book = staves(2000, 700, [
      [100, 0, 80, 900],
      [300, 0, 80, 900],
      [100, 0, 1100, 1920],
      [300, 0, 1100, 1920],
    ]);
    const pages = findStaves(book, 12);
    expect(pages.map((f) => f.column)).toEqual([0, 0, 1, 1]);
    pages.forEach((f, i) => expect(Math.abs(f.lines[0]! - [100, 300][i % 2]!)).toBeLessThanOrEqual(1));
    // A blank page finds nothing.
    expect(findStaves(staves(800, 600, []), 12)).toEqual([]);
  });

  it("merges the host's staves with the engine's: the host's where both found one, the engine's where it has none", () => {
    const found = findStaves(staves(1600, 900, [
      [100, 0, 100, 1500],
      [300, 0, 100, 1500],
      [500, 0, 100, 1500],
    ]), 12);
    const host = [
      { lines: [100.5, 112.5, 124.5, 136.5, 148.5], tilt: 0, spacing: 12 },
      { lines: [700, 712, 724, 736, 748], tilt: 0, spacing: 12 },
    ];
    const merged = mergeStaves(host, found);
    expect(merged.map((s) => Math.round(s.lines[0]! / 10) * 10)).toEqual([100, 300, 500, 700]);
    expect(merged[0]!.lines[0]).toBe(100.5);
    expect(merged[0]!.bend).toBeUndefined();
    expect(merged[1]!.bend).toBeDefined();
  });

  it('reads a page shrunk to its reading spacing and says where its staves are in the page as given (task 0018)', async () => {
    const g = { width: 4, height: 2, data: Uint8Array.from([0, 100, 200, 200, 0, 100, 200, 200]) };
    expect(Array.from(shrinkGray(g, 0.5).data)).toEqual([50, 200]);
    // A page at 28 px a spacing, read at 14: the staff's grid comes back in the page's own pixels.
    const width = 1600;
    const height = 400;
    const rgba = new Uint8ClampedArray(width * height * 4).fill(235);
    for (let k = 0; k < 5; k++) for (let x = 100; x <= 1500; x++) for (const dy of [0, 1, 2]) rgba.fill(20, ((100 + 28 * k + dy) * width + x) * 4, ((100 + 28 * k + dy) * width + x) * 4 + 3);
    const tensor = (type: string, data: Float32Array | BigInt64Array, dims: readonly number[]) => ({ type, data, dims });
    class Tensor {
      constructor(
        readonly type: string,
        readonly data: Float32Array | BigInt64Array,
        readonly dims: readonly number[],
      ) {}
    }
    const ort = { Tensor };
    const encoder = { run: async () => ({ output: tensor('float32', new Float32Array(512), [1, 1, 512]) }) };
    // The decoder ends every staff at once (the end token in the rhythm branch).
    const decoder = {
      run: async () => {
        const out: Record<string, unknown> = {};
        for (const [name, n] of [['out_rhythms', 259], ['out_pitchs', 72], ['out_lifts', 7], ['out_positions', 3], ['out_articulations', 54], ['out_slurs', 5]] as const) {
          const d = new Float32Array(n);
          if (name === 'out_rhythms') d[2] = 10;
          out[name] = tensor('float32', d, [1, 1, n]);
        }
        for (let i = 0; i < 32; i++) out[`cache_out${i}`] = tensor('float32', new Float32Array(0), [1, 8, 0, 64]);
        return out;
      },
    };
    const readings = await readImage(ort as never, encoder as never, decoder as never, { width, height, data: rgba }, [{ lines: [101, 129, 157, 185, 213], tilt: 0, spacing: 28 }], undefined, undefined, { staves: 'host' });
    expect(readings).toHaveLength(1);
    const grid = readings[0]!.grid!;
    expect(grid[0]![0]!).toBeLessThan(120);
    expect(grid[grid.length - 1]![0]!).toBeGreaterThan(1480);
    for (const p of grid) expect(Math.abs(p[1]! - 101)).toBeLessThanOrEqual(0.15 * 28);
    // One model at a time (task 0023): the encoder reads every staff image first and is released, the same answer.
    let released = 0;
    const order: string[] = [];
    const enc = { run: async () => (order.push('encoder'), encoder.run()), release: async () => void (released++, order.push('release')) };
    const dec = { run: async () => (order.push('decoder'), decoder.run()) };
    const one = await readImage(ort as never, enc as never, dec as never, { width, height, data: rgba }, [{ lines: [101, 129, 157, 185, 213], tilt: 0, spacing: 28 }], undefined, undefined, { staves: 'host', oneAtATime: true });
    expect(one.map(({ ms: _ms, ...r }) => r)).toEqual(readings.map(({ ms: _ms, ...r }) => r));
    expect(released).toBe(1);
    expect(order).toEqual(['encoder', 'release', 'decoder']);
  });
});
