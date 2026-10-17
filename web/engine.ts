/**
 * homr in the browser, one page: every staff image through the transformer and into a timeline (homr's
 * parse_staffs, without its staff detection: the host app gives the straightened page and its staves).
 * AGPL-3.0, like homr (../LICENSE).
 */
import { clahe, joined, lineOffsets, staffExtent, staffInput, toGray, type Gray } from './crop';
import { readStaff, type HomrSymbol, type OrtLike, type OrtSessionLike } from './staff2score';
import { timeline, type StaffTimeline } from './tokens';

export const ENGINE = { name: 'homr', version: '0.7.0', port: 'web 3, p' } as const;

/** A stave as the host app found it: five line ys at x = 0, tilt, spacing, and where its lines run if known. */
export interface HostStave {
  readonly lines: readonly number[];
  readonly tilt: number;
  readonly spacing: number;
  readonly x0?: number;
  readonly x1?: number;
}

export interface StaffReading {
  /** Index of the stave in the request. */
  readonly stave: number;
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

/** Reads every stave of a prepared page, top to bottom, one staff image each. */
export async function readPage(
  ort: OrtLike,
  encoder: OrtSessionLike,
  decoder: OrtSessionLike,
  page: Gray,
  staves: readonly HostStave[],
  onStave?: (done: number, total: number) => void,
  aborted?: () => boolean,
  options: { readonly grand?: boolean } = { grand: false },
): Promise<StaffReading[]> {
  const out: StaffReading[] = [];
  const spans = staves.map((s) => {
    const x = ((s.x0 ?? 0) + (s.x1 ?? page.width)) / 2;
    return { top: s.lines[0]! + x * s.tilt, bottom: s.lines[s.lines.length - 1]! + x * s.tilt };
  });
  // Where each staff's lines really run along the page (they bend on a photo), then where they start and end.
  const shifts = staves.map((s) => lineOffsets(page, { lines: s.lines, tilt: s.tilt, spacing: s.spacing, x0: 0, x1: page.width - 1 }));
  const own = staves.map((s, i) => (s.x0 !== undefined && s.x1 !== undefined ? { x0: s.x0, x1: s.x1 } : staffExtent(page, s.lines, s.tilt, s.spacing, shifts[i])));
  // Systems: staves joined at their left ends; a system's staves share its extent (its lines end together).
  const systems: number[] = [];
  staves.forEach((s, i) => {
    if (i === 0) return systems.push(0);
    const up = staves[i - 1]!;
    const bottom = up.lines[up.lines.length - 1]! + own[i - 1]!.x0 * up.tilt + shifts[i - 1]![Math.round(own[i - 1]!.x0)]!;
    const top = s.lines[0]! + own[i]!.x0 * s.tilt + shifts[i]![Math.round(own[i]!.x0)]!;
    return systems.push(joined(page, { bottom, x0: own[i - 1]!.x0, spacing: up.spacing }, { top, x0: own[i]!.x0 }) ? systems[i - 1]! : systems[i - 1]! + 1);
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
  for (const [n, unit] of units.entries()) {
    if (aborted?.()) break;
    const t = Date.now();
    const i = unit[0]!;
    const s = staves[i]!;
    const extent = extents[i]!;
    const lines = unit.flatMap((k) => staves[k]!.lines);
    const spacing = unit.reduce((a, k) => a + staves[k]!.spacing, 0) / unit.length;
    const shift = unit.length === 2 ? ([shifts[unit[0]!]!, shifts[unit[1]!]!] as const) : shifts[i]!;
    const input = staffInput(page, { lines, tilt: s.tilt, spacing, ...extent }, spans.filter((_, k) => !unit.includes(k)), shift);
    const symbols = await readStaff(ort, encoder, decoder, input.data);
    out.push({ stave: i, system: systems[i]!, grand: unit.length === 2, ...extent, symbols, timeline: timeline(symbols, unit.length === 2), ms: Date.now() - t });
    onStave?.(n + 1, units.length);
  }
  return out;
}
