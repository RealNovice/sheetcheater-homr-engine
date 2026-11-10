/**
 * homr's symbols to a timeline (the parts of homr/transformer/vocabulary.py and homr/music_xml_generator.py that
 * give notes their time: chords, kern durations, tuplets, grace notes, clefs, keys, barlines), ported to
 * TypeScript. AGPL-3.0, like homr (../LICENSE).
 *
 * Where homr writes MusicXML, this gives each note its bar, onset and length directly. Voices: homr writes a
 * chord's notes with their own lengths and moves on by the shortest; a note that outlasts it rings on in another
 * voice, as homr's rebalance_measure_voices does.
 */
import type { HomrSymbol } from './staff2score';

export type Clef = { readonly sign: 'G' | 'F' | 'C'; readonly line: number; readonly octave: number };
export type Base = 1 | 2 | 4 | 8 | 16 | 32;

export interface TimedNote {
  /** 1 upper, 2 lower (a grand staff read as one image); 1 otherwise. */
  readonly staff: 1 | 2;
  readonly bar: number;
  /** Quarter notes from the bar's start. */
  readonly onset: number;
  /** Quarter notes; 0 for a grace note. */
  readonly beats: number;
  readonly duration: { readonly base: Base; readonly dots: number; readonly tuplet?: readonly [number, number] };
  readonly rest: boolean;
  readonly voice: number;
  readonly grace: boolean;
  readonly midi: number | null;
  readonly diatonic: number | null;
  readonly alter: number;
  /** homr says nothing of printed accidentals: a natural it names, or an alteration the key does not give. */
  readonly printed: boolean;
  readonly clef: Clef;
  readonly fifths: number;
  /** The decoder's probability of this note's length, pitch and accidental (task 0016); absent on old readings. */
  readonly p?: { readonly rhythm: number; readonly pitch: number; readonly lift: number };
  /** The decoder's runner-up readings of this note (its second pitch, its second accidental), likeliest first. */
  readonly alts?: readonly { readonly midi: number; readonly diatonic: number; readonly p: number }[];
}

export interface StaffTimeline {
  readonly notes: readonly TimedNote[];
  readonly bars: number;
  readonly fifths: number | null;
  /** Beats and beat type; homr names only the beat type, the beats come from the bars' lengths. */
  readonly time: readonly [number, number] | null;
}

const STEP: Record<string, number> = { C: 0, D: 2, E: 4, F: 5, G: 7, A: 9, B: 11 };
const LETTER: Record<string, number> = { C: 0, D: 1, E: 2, F: 3, G: 4, A: 5, B: 6 };
const LIFT: Record<string, number> = { '#': 1, '##': 2, N: 0, b: -1, bb: -2, _: 0, '.': 0 };
const SHARPS = [3, 0, 4, 1, 5, 2, 6];
const FLATS = [6, 2, 5, 1, 4, 0, 3];

function keyAlter(letter: number, fifths: number): number {
  if (fifths > 0) return SHARPS.slice(0, fifths).includes(letter) ? 1 : 0;
  if (fifths < 0) return FLATS.slice(0, -fifths).includes(letter) ? -1 : 0;
  return 0;
}

/** A **kern duration ("4.", "12", "8G", "0") as homr reads it: quarter beats and the written note value. */
export function kernDuration(kern: string): { beats: number; grace: boolean; bar: boolean; duration: TimedNote['duration'] } {
  const m = /^(\d*)(G?)(\.*)$/.exec(kern.replace(/m$/, ''));
  const base = m && m[1] ? Number(m[1]) : 4;
  const dots = m ? m[3]!.length : 0;
  const grace = !!m && m[2] === 'G';
  const dot = (b: number): number => {
    let d = b;
    let add = b / 2;
    for (let i = 0; i < dots; i++) {
      d += add;
      add /= 2;
    }
    return d;
  };
  if (base === 0 || /m$/.test(kern)) return { beats: 4, grace: false, bar: true, duration: { base: 1, dots: 0 } };
  const pow2 = (base & (base - 1)) === 0;
  const normal = pow2 ? base : 1 << (31 - Math.clz32(base));
  const written = Math.min(32, Math.max(1, normal)) as Base;
  const g = (a: number, b: number): number => (b ? g(b, a % b) : a);
  const k = g(base, normal);
  const tuplet: readonly [number, number] | undefined = pow2 ? undefined : [base / k, normal / k];
  return { beats: grace ? 0 : dot(4 / base), grace, bar: false, duration: { base: written, dots, ...(tuplet ? { tuplet } : {}) } };
}

function clefOf(rhythm: string): Clef | null {
  const m = /^clef_([GFC])(\d)$/.exec(rhythm);
  return m ? { sign: m[1] as Clef['sign'], line: Number(m[2]), octave: 0 } : null;
}

/** homr's symbols for one staff image (one stave, or the two of a grand staff) as a timeline. */
export function timeline(symbols: readonly HomrSymbol[], grand: boolean): StaffTimeline {
  // Chords: a "chord" symbol joins the next symbol to the one before (sort_token_chords).
  const groups: HomrSymbol[][] = [];
  let join = false;
  for (const s of symbols) {
    if (s.rhythm === 'chord') {
      join = true;
      continue;
    }
    if (!grand && s.position === 'lower') continue;
    if (join && groups.length) groups[groups.length - 1]!.push(s);
    else groups.push([s]);
    join = false;
  }
  const clefs: Record<1 | 2, Clef> = { 1: { sign: 'G', line: 2, octave: 0 }, 2: { sign: 'F', line: 4, octave: 0 } };
  let fifths: number | null = null;
  let beatType: number | null = null;
  let bar = 0;
  let at = 0;
  let barHas = false;
  const notes: TimedNote[] = [];
  const barLengths: number[] = [];
  // Voices: when each is free again, per staff.
  let free: Record<1 | 2, number[]> = { 1: [], 2: [] };
  const voiceFor = (staff: 1 | 2, onset: number, end: number): number => {
    const list = free[staff];
    let v = list.findIndex((f) => f <= onset + 1e-6);
    if (v < 0) {
      list.push(0);
      v = list.length - 1;
    }
    list[v] = end;
    return v + 1;
  };
  const closeBar = (): void => {
    if (!barHas && at === 0) return;
    barLengths.push(at);
    bar++;
    at = 0;
    barHas = false;
    free = { 1: [], 2: [] };
  };
  for (const g of groups) {
    const r = g[0]!.rhythm;
    if (/barline|^repeat/.test(r)) {
      closeBar();
      continue;
    }
    if (r.startsWith('clef')) {
      for (const s of g) {
        const c = clefOf(s.rhythm);
        if (c) clefs[grand && s.position === 'lower' ? 2 : 1] = c;
      }
      continue;
    }
    if (r.startsWith('keySignature_')) {
      fifths = Number(r.slice('keySignature_'.length));
      continue;
    }
    if (r.startsWith('timeSignature/')) {
      beatType = Number(r.slice('timeSignature/'.length));
      continue;
    }
    if (!/^(note|rest)_/.test(r)) continue;
    const items = g.filter((s) => /^(note|rest)_/.test(s.rhythm)).map((s) => ({ s, d: kernDuration(s.rhythm.replace(/^(note|rest)_/, '')) }));
    const grace = items.every((i) => i.d.grace);
    const advance = grace ? 0 : Math.min(...items.filter((i) => !i.d.grace).map((i) => i.d.beats));
    // Notes of one length in one staff share a voice; a longer note rings on in the next free one.
    const lanes = new Map<string, number>();
    for (const { s, d } of items) {
      const staff: 1 | 2 = grand && s.position === 'lower' ? 2 : 1;
      const rest = s.rhythm.startsWith('rest');
      const key = `${staff}/${d.beats}/${rest}`;
      let voice = lanes.get(key);
      if (voice === undefined) {
        voice = d.grace ? Math.max(1, free[staff].length) : voiceFor(staff, at, at + d.beats);
        lanes.set(key, voice);
      }
      const pm = /^([A-G])(\d)$/.exec(s.pitch);
      const k = fifths ?? 0;
      const alter = LIFT[s.lift] ?? 0;
      const letter = pm ? LETTER[pm[1]!]! : 0;
      let sure: Pick<TimedNote, 'p' | 'alts'> = {};
      if (s.conf && pm && !rest) {
        const octave = Number(pm[2]);
        const alts: { midi: number; diatonic: number; p: number }[] = [];
        const p2 = /^([A-G])(\d)$/.exec(s.conf.pitch2);
        if (p2) {
          // The runner-up letter keeps the accidental if it was the note's own, else takes the key's.
          const l2 = LETTER[p2[1]!]!;
          const a2 = alter === keyAlter(letter, k) ? keyAlter(l2, k) : alter;
          alts.push({ midi: 12 * (Number(p2[2]) + 1) + STEP[p2[1]!]! + a2, diatonic: Number(p2[2]) * 7 + l2, p: s.conf.pitch2p });
        }
        const lift2 = LIFT[s.conf.lift2];
        if (lift2 !== undefined && s.conf.lift2 !== '.') {
          const a2 = lift2;
          if (a2 !== alter) alts.push({ midi: 12 * (octave + 1) + STEP[pm[1]!]! + a2, diatonic: octave * 7 + letter, p: s.conf.lift2p });
        }
        alts.sort((a, b) => b.p - a.p);
        sure = { p: { rhythm: s.conf.rhythm, pitch: s.conf.pitch, lift: s.conf.lift }, alts };
      }
      notes.push({
        staff,
        bar,
        onset: at,
        beats: d.bar ? 0 : d.beats,
        duration: d.duration,
        rest: rest || !pm,
        voice,
        grace: d.grace,
        midi: pm && !rest ? 12 * (Number(pm[2]) + 1) + STEP[pm[1]!]! + alter : null,
        diatonic: pm && !rest ? Number(pm[2]) * 7 + letter : null,
        alter,
        printed: !rest && !!pm && (s.lift === 'N' ? keyAlter(letter, k) !== 0 : alter !== keyAlter(letter, k)),
        clef: clefs[staff],
        fifths: k,
        ...sure,
      });
      barHas = true;
    }
    at += items.some((i) => i.d.bar) ? 0 : advance;
  }
  closeBar();
  // Beats per bar: the commonest length of a full bar.
  let time: [number, number] | null = null;
  if (beatType) {
    const counts = new Map<number, number>();
    for (const l of barLengths) if (l > 0) counts.set(Math.round(l * 4) / 4, (counts.get(Math.round(l * 4) / 4) ?? 0) + 1);
    const common = [...counts.entries()].sort((a, b) => b[1] - a[1])[0]?.[0];
    const beats = common ? Math.round((common * beatType) / 4) : null;
    if (beats && beats > 0 && beats <= 16) time = [beats, beatType];
  }
  return { notes, bars: bar, fifths, time };
}
