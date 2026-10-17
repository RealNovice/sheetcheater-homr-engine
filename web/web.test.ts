// Tests for the browser port of homr (AGPL-3.0, ../LICENSE).
import { describe, expect, it } from 'vitest';

import { CANVAS, joined, staffExtent, staffInput, type Gray } from './crop';
import { topTwo, type HomrSymbol } from './staff2score';
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
});
