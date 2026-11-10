/**
 * homr in the browser, one page: every staff image through the transformer and into a timeline (homr's
 * parse_staffs). The host app gives the straightened page and the staves it found; the engine finds its own as
 * well (staves.ts, task 0018) and reads those where the host's are missing. AGPL-3.0, like homr (../LICENSE).
 */
import { clahe, inkMap, joined, lineOffsets, staffExtent, staffInput, toGray, type Gray } from './crop';
import { encodeStaff, readStaff, type HomrSymbol, type OrtLike, type OrtSessionLike, type OrtTensor } from './staff2score';
import { estimateSpacing, findStaves, type FoundStave } from './staves';
import { timeline, type StaffTimeline } from './tokens';

export const ENGINE = { name: 'homr', version: '0.7.0', port: 'web 0018.9' } as const;

/** A stave as the host app found it: five line ys at x = 0, tilt, spacing, and where its lines run if known. */
export interface HostStave {
  readonly lines: readonly number[];
  readonly tilt: number;
  readonly spacing: number;
  readonly x0?: number;
  readonly x1?: number;
  /** Per page column, how far the lines stand from the straight ones (a curl), when known. */
  readonly bend?: ArrayLike<number>;
}

/**
 * Which staves are read (task 0018): the host's, the engine's own (staves.ts), the engine's where they are more, or
 * both (`merge`: the host's where the engine found the same stave, the engine's where the host has none).
 */
export type StaffSource = 'host' | 'own' | 'auto' | 'merge';
/** Measured in task 0018 (accuracy-all.mjs --homr-local). */
export const STAFF_SOURCE: StaffSource = 'merge';

export interface StaffReading {
  /** Index of the stave in the request, or (`own`) of the engine's own staves, top to bottom. */
  readonly stave: number;
  /** Read on staves the engine found itself: `stave` counts those, not the request's. */
  readonly own?: boolean;
  /**
   * Where the stave's lines run on the page, as `[x, y0 … y4]` (ten ys for a grand staff) every few spacings: the
   * lines the staff image was cut along.
   */
  readonly grid?: readonly (readonly number[])[];
  /** The system it is in (staves joined by a line at their left ends), from 0 down the page. */
  readonly system: number;
  /** Read with the stave below it as one grand staff image: its notes on staff 2 are that stave's. */
  readonly grand: boolean;
  readonly x0: number;
  readonly x1: number;
  readonly symbols: readonly HomrSymbol[];
  readonly timeline: StaffTimeline;
  readonly ms: number;
}

export interface PageImage {
  readonly width: number;
  readonly height: number;
  readonly data: Uint8Array | Uint8ClampedArray;
}

/** The page as homr reads it: gray, then contrast-limited equalisation (homr's apply_clahe). */
export function preparePage(rgba: PageImage): Gray {
  return clahe(toGray(rgba));
}

/**
 * The staff spacing a page is read at (task 0018). A staff image keeps 10 to 20 px a spacing (its 256 rows hold
 * twelve spacings; a long staff is narrower still), so a photo decoded at 24 px a spacing carries pixels no
 * staff image uses; homr itself reads every page at 1920 px wide, about 14 px a spacing.
 */
export const READ_SPACING = 14;

/** A gray image shrunk by `f` (< 1), every output pixel the mean of the input it covers. */
export function shrinkGray(g: Gray, f: number): Gray {
  const w = Math.max(1, Math.round(g.width * f));
  const h = Math.max(1, Math.round(g.height * f));
  const fx = g.width / w;
  const fy = g.height / h;
  // Rows first: each output column the mean of the input columns it covers (fractions at both ends).
  const rows = new Float32Array(w * g.height);
  for (let x = 0; x < w; x++) {
    const a = x * fx;
    const b = a + fx;
    const i0 = Math.floor(a);
    const i1 = Math.min(g.width - 1, Math.ceil(b) - 1);
    for (let y = 0; y < g.height; y++) {
      const row = y * g.width;
      let sum = 0;
      for (let i = i0; i <= i1; i++) sum += g.data[row + i]! * (Math.min(b, i + 1) - Math.max(a, i));
      rows[y * w + x] = sum / fx;
    }
  }
  const out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    const a = y * fy;
    const b = a + fy;
    const j0 = Math.floor(a);
    const j1 = Math.min(g.height - 1, Math.ceil(b) - 1);
    for (let x = 0; x < w; x++) {
      let sum = 0;
      for (let j = j0; j <= j1; j++) sum += rows[j * w + x]! * (Math.min(b, j + 1) - Math.max(a, j));
      out[y * w + x] = Math.round(sum / fy);
    }
  }
  return { width: w, height: h, data: out };
}

/**
 * Reads a page as the host hands it (RGBA, its staves): shrunk to `READ_SPACING` when its staves are wider
 * apart, prepared as homr prepares a page, every staff read, and everything said in the host's pixels again.
 */
export async function readImage(
  ort: OrtLike,
  encoder: OrtSessionLike,
  decoder: OrtSessionLike,
  rgba: PageImage,
  staves: readonly HostStave[],
  onStave?: (done: number, total: number) => void,
  aborted?: () => boolean,
  options: ReadOptions & { readonly spacing?: number } = {},
): Promise<StaffReading[]> {
  const gray = toGray(rgba);
  const hint = staves.length ? [...staves.map((s) => s.spacing)].sort((a, b) => a - b)[staves.length >> 1]! : estimateSpacing(gray);
  const target = options.spacing ?? READ_SPACING;
  const f = hint && hint > 1.15 * target ? target / hint : 1;
  const page = clahe(f < 1 ? shrinkGray(gray, f) : gray);
  const scaled: HostStave[] = f < 1 ? staves.map((s) => ({ lines: s.lines.map((l) => l * f), tilt: s.tilt, spacing: s.spacing * f, ...(s.x0 !== undefined ? { x0: s.x0 * f } : {}), ...(s.x1 !== undefined ? { x1: s.x1 * f } : {}) })) : [...staves];
  const readings = await readPage(ort, encoder, decoder, page, scaled, onStave, aborted, options);
  if (f >= 1) return readings;
  const back = (v: number): number => Math.round((v / f) * 10) / 10;
  return readings.map((r) => ({ ...r, x0: back(r.x0), x1: back(r.x1), ...(r.grid ? { grid: r.grid.map((p) => p.map(back)) } : {}) }));
}

/** Where a page's staves run, which system each is in, and the staff images to read (one stave, or a grand pair). */
export function pageLayout(page: Gray, staves: readonly HostStave[], options: { readonly grand?: boolean } = { grand: false }): { spans: { top: number; bottom: number }[]; shifts: Float32Array[]; extents: { x0: number; x1: number }[]; systems: number[]; units: number[][] } {
  const spans = staves.map((s) => {
    const x = ((s.x0 ?? 0) + (s.x1 ?? page.width)) / 2;
    return { top: s.lines[0]! + x * s.tilt, bottom: s.lines[s.lines.length - 1]! + x * s.tilt };
  });
  // Where each staff's lines really run along the page (they bend on a photo), then where they start and end.
  // A stave the engine found is followed along its own stretch (the middle of an open book is its gutter).
  const shifts = staves.map((s) => lineOffsets(page, { lines: s.lines, tilt: s.tilt, spacing: s.spacing, ...(s.bend && s.x0 !== undefined && s.x1 !== undefined ? { x0: s.x0, x1: s.x1 } : { x0: 0, x1: page.width - 1 }) }, s.bend));
  const own = staves.map((s, i) => {
    if (s.x0 === undefined || s.x1 === undefined) return staffExtent(page, s.lines, s.tilt, s.spacing, shifts[i]);
    if (!s.bend) return { x0: s.x0, x1: s.x1 };
    // Found by the engine: where its lines stand out, within reach of where they were followed.
    const e = staffExtent(page, s.lines, s.tilt, s.spacing, shifts[i]);
    const x0 = Math.max(e.x0, s.x0 - 6 * s.spacing);
    const x1 = Math.min(e.x1, s.x1 + 6 * s.spacing);
    return x1 - x0 >= 0.5 * (s.x1 - s.x0) ? { x0, x1 } : { x0: s.x0, x1: s.x1 };
  });
  // Systems: staves joined at their left ends; a system's staves share its extent (its lines end together).
  const systems: number[] = [];
  const ink = staves.length > 1 ? inkMap(page, [...staves.map((s) => s.spacing)].sort((a, b) => a - b)[staves.length >> 1]!) : page;
  staves.forEach((s, i) => {
    if (i === 0) return systems.push(0);
    const up = staves[i - 1]!;
    const bottom = up.lines[up.lines.length - 1]! + own[i - 1]!.x0 * up.tilt + shifts[i - 1]![Math.round(own[i - 1]!.x0)]!;
    const top = s.lines[0]! + own[i]!.x0 * s.tilt + shifts[i]![Math.round(own[i]!.x0)]!;
    return systems.push(joined(ink, { bottom, x0: own[i - 1]!.x0, spacing: up.spacing }, { top, x0: own[i]!.x0 }) ? systems[i - 1]! : systems[i - 1]! + 1);
  });
  const extents = staves.map((_, i) => {
    const mates = own.filter((__, k) => systems[k] === systems[i]);
    return { x0: Math.min(...mates.map((e) => e.x0)), x1: Math.max(...mates.map((e) => e.x1)) };
  });
  // Staff images: one per stave, or (grand) the last two staves of a system together, as homr reads a braced pair.
  const units: number[][] = [];
  for (let i = 0; i < staves.length; i++) {
    const next = i + 1 < staves.length && systems[i + 1] === systems[i] && (i + 2 >= staves.length || systems[i + 2] !== systems[i]);
    if (options.grand && next) {
      units.push([i, i + 1]);
      i++;
    } else units.push([i]);
  }
  return { spans, shifts, extents, systems, units };
}

export interface ReadOptions {
  readonly grand?: boolean;
  readonly staves?: StaffSource;
  /**
   * One model at a time (a phone, task 0023): every staff image through the encoder first, then the encoder is
   * released (`OrtSessionLike.release`) before the decoder writes the symbols. The same symbols.
   */
  readonly oneAtATime?: boolean;
}

/** Reads every stave of a prepared page, top to bottom, one staff image each. */
export async function readPage(
  ort: OrtLike,
  encoder: OrtSessionLike,
  decoder: OrtSessionLike,
  page: Gray,
  staves: readonly HostStave[],
  onStave?: (done: number, total: number) => void,
  aborted?: () => boolean,
  options: ReadOptions = {},
): Promise<StaffReading[]> {
  const out: StaffReading[] = [];
  const source = options.staves ?? STAFF_SOURCE;
  let own = false;
  if (source !== 'host') {
    const hint = staves.length ? [...staves.map((s) => s.spacing)].sort((a, b) => a - b)[staves.length >> 1]! : null;
    const found = findStaves(page, hint);
    if (source === 'merge' && found.length) {
      staves = mergeStaves(staves, found);
      own = true;
    } else if (source === 'own' ? found.length > 0 : source === 'auto' && found.length > staves.length) {
      staves = found;
      own = true;
    }
  }
  const { spans, shifts, extents, systems, units } = pageLayout(page, staves, options);
  const imageOf = (unit: readonly number[]): Float32Array => {
    const i = unit[0]!;
    const lines = unit.flatMap((k) => staves[k]!.lines);
    const spacing = unit.reduce((a, k) => a + staves[k]!.spacing, 0) / unit.length;
    const shift = unit.length === 2 ? ([shifts[unit[0]!]!, shifts[unit[1]!]!] as const) : shifts[i]!;
    return staffInput(page, { lines, tilt: staves[i]!.tilt, spacing, ...extents[i]! }, spans.filter((_, k) => !unit.includes(k)), shift).data;
  };
  const encoded: OrtTensor[] = [];
  const encodeMs: number[] = [];
  if (options.oneAtATime) {
    for (const unit of units) {
      if (aborted?.()) break;
      const t = Date.now();
      encoded.push(await encodeStaff(ort, encoder, imageOf(unit)));
      encodeMs.push(Date.now() - t);
    }
    await encoder.release?.();
  }
  for (const [n, unit] of units.entries()) {
    if (aborted?.()) break;
    const t = Date.now() - (encodeMs[n] ?? 0);
    const i = unit[0]!;
    const extent = extents[i]!;
    const spacing = unit.reduce((a, k) => a + staves[k]!.spacing, 0) / unit.length;
    const symbols = encoded[n] ? await readStaff(ort, null, decoder, null, undefined, encoded[n]) : await readStaff(ort, encoder, decoder, imageOf(unit));
    const grid: number[][] = [];
    const every = Math.max(8, Math.round(4 * spacing));
    for (let x = extent.x0; ; x = Math.min(extent.x1, x + every)) {
      const k = Math.max(0, Math.min(page.width - 1, Math.round(x)));
      grid.push([Math.round(x * 10) / 10, ...unit.flatMap((u) => staves[u]!.lines.map((l) => Math.round((l + x * staves[u]!.tilt + shifts[u]![k]!) * 10) / 10))]);
      if (x >= extent.x1) break;
    }
    out.push({ stave: i, ...(own ? { own: true } : {}), grid, system: systems[i]!, grand: unit.length === 2, ...extent, symbols, timeline: timeline(symbols, unit.length === 2), ms: Date.now() - t });
    onStave?.(n + 1, units.length);
  }
  return out;
}

/**
 * The host's staves and the engine's together: where both found a stave (its middle line within 0.4 spacings of
 * the other's at three places along it, the same spacing), the host's (its page was straightened along them);
 * the engine's where the host has none; a host stave the engine did not find only where no stave of the engine's
 * lies on it. On an open book our staves keep to the page the engine found them on. Top to bottom, the pages of an open book in turn.
 */
export function mergeStaves(host: readonly HostStave[], found: readonly FoundStave[]): HostStave[] {
  const used = new Set<number>();
  const book = found.some((f) => f.column > 0);
  const out: (HostStave & { column: number })[] = found.map((f) => {
    const at = (x: number): number => f.lines[2]! + x * f.tilt + f.bend[Math.max(0, Math.min(f.bend.length - 1, Math.round(x)))]!;
    const xs = [0.2, 0.5, 0.8].map((t) => f.x0 + t * (f.x1 - f.x0));
    const k = host.findIndex((h, i) => !used.has(i) && Math.abs(h.spacing - f.spacing) <= 0.12 * f.spacing && xs.every((x) => Math.abs(h.lines[2]! + x * h.tilt - at(x)) <= 0.4 * f.spacing));
    if (k < 0) return f;
    used.add(k);
    const h = host[k]!;
    // Our lines; on an open book, on the stretch the engine followed them along (the other page is not this
    // stave's, and the middle of the page is the gutter).
    if (book) return { lines: h.lines, tilt: h.tilt, spacing: h.spacing, x0: f.x0, x1: f.x1, bend: new Float32Array(f.bend.length), column: f.column };
    return { lines: h.lines, tilt: h.tilt, spacing: h.spacing, ...(h.x0 !== undefined ? { x0: h.x0 } : {}), ...(h.x1 !== undefined ? { x1: h.x1 } : {}), column: f.column };
  });
  host.forEach((h, i) => {
    if (used.has(i)) return;
    const clash = found.some((f) => {
      const x = (f.x0 + f.x1) / 2;
      return Math.abs(h.lines[2]! + x * h.tilt - (f.lines[2]! + x * f.tilt + f.bend[Math.round(x)]!)) <= 4.5 * f.spacing;
    });
    if (!clash) out.push({ ...h, column: 0 });
  });
  const mid = (s: HostStave): number => s.lines[2]! + (((s.x0 ?? 0) + (s.x1 ?? 0)) / 2) * s.tilt;
  return out.sort((a, b) => a.column - b.column || mid(a) - mid(b)).map(({ column: _column, ...s }) => s);
}
