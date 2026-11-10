/**
 * The engine's own staff finding (task 0018), for pages whose staves the host found only in part or not at all: a
 * photo taken at an angle, a curled page, an open book, a shadow. homr finds staves from its segmentation model's
 * line mask (staff_detection.py: line fragments joined where five run together); this does the same from the page
 * itself: in narrow vertical strips, five evenly spaced thin dark lines, followed strip to strip across the page.
 * AGPL-3.0, like homr (../LICENSE).
 */
import { inkMap, type Gray } from './crop';

export interface FoundStave {
  /** The five line ys at x = 0 of the straight line fitted to the stave. */
  readonly lines: readonly number[];
  readonly tilt: number;
  readonly spacing: number;
  readonly x0: number;
  readonly x1: number;
  /** Per page column, how far the stave's lines stand from the fitted straight lines (a curl, a bend). */
  readonly bend: Float32Array;
  /** Which page of an open book (0 left, 1 right); 0 on a single page. */
  readonly column: number;
}

export const FIND = {
  /** Strip width and step, in spacings. */
  stripWidth: 2,
  stripStep: 1,
  /** The spacings tried about the expected one. */
  spacingRange: [0.8, 1.25] as const,
  spacingStep: 0.025,
  /** A candidate's weakest line against its five lines' mean. */
  minLineShare: 0.35,
  /** The five lines' mean against the mean between them. */
  minContrast: 2.5,
  /** A candidate against the page's strong candidates (their 95th percentile). */
  minScoreShare: 0.18,
  /** Strips a stave may skip (a beam, a word across it) and still be followed. */
  maxGapStrips: 4,
  /** Pieces of one stave are joined across at most this many spacings, never across an open book's gutter. */
  maxJoinGap: 40,
  /** Following a stave: how far it may move from where it was heading, in spacings, and its spacing change. */
  followReach: 0.6,
  followSpacing: 0.15,
  /** A stave runs at least this many spacings, and this share of the page's longest. */
  minLengthSpacings: 12,
  minLengthShare: 0.4,
  /** The longest stave runs at least this share of the page's width. */
  minPageShare: 0.3,
  /** Strips of its span in which a stave was found. */
  minDensity: 0.3,
  /** A stave's median candidate against the page's strong staves'. */
  minStrengthShare: 0.45,
} as const;

/** Horizontal thin-line response per column block: for every strip of `step` columns, the summed ridge per row. */
function blockProfiles(page: Gray, step: number, d: number): Float32Array[] {
  const { width: w, height: h, data } = page;
  const blocks = Math.ceil(w / step);
  const out = Array.from({ length: blocks }, () => new Float32Array(h));
  for (let y = d; y < h - d; y++) {
    const row = y * w;
    const up = (y - d) * w;
    const down = (y + d) * w;
    for (let x = 0; x < w; x++) {
      const v = data[row + x]!;
      const a = data[up + x]!;
      const b = data[down + x]!;
      const r = (a < b ? a : b) - v;
      if (r > 0) out[(x / step) | 0]![y]! += r;
    }
  }
  return out;
}

/**
 * The page's staff spacing from the lines alone: down every few columns, the commonest distance from one thin
 * dark run to the next (a white run between two short black ones), on ink judged against its surroundings.
 */
export function estimateSpacing(page: Gray): number | null {
  const ink = inkMap(page, 10);
  const { width: w, height: h, data } = ink;
  const hist = new Float64Array(161);
  for (let x = 0; x < w; x += 4) {
    let y = 0;
    let lastBlack = -1;
    let lastBlackLen = 0;
    while (y < h) {
      const black = data[y * w + x] === 0;
      let e = y;
      while (e < h && (data[e * w + x] === 0) === black) e++;
      const len = e - y;
      if (black) {
        if (lastBlack >= 0) {
          const white = y - lastBlack;
          if (lastBlackLen <= Math.max(3, white / 3) && len <= Math.max(3, white / 3)) {
            const d = Math.round(white + (lastBlackLen + len) / 2);
            if (d >= 4 && d <= 160) hist[d]! += 1;
          }
        }
        lastBlack = e;
        lastBlackLen = len;
      }
      y = e;
    }
  }
  let best = -1;
  let bestV = 0;
  for (let d = 5; d < 160; d++) {
    const v = hist[d - 1]! + 2 * hist[d]! + hist[d + 1]!;
    if (v > bestV) [best, bestV] = [d, v];
  }
  return best > 0 ? best : null;
}

interface Candidate {
  readonly strip: number;
  readonly x: number;
  readonly y: number;
  readonly s: number;
  readonly score: number;
}

/** Five-line candidates in one strip's profile: best spacing per row, then the strongest rows at least a staff apart. */
function stripCandidates(p: Float32Array, spacings: readonly number[], strip: number, x: number): Candidate[] {
  const h = p.length;
  const at = (v: number): number => {
    const i = Math.floor(v);
    if (i < 0 || i + 1 >= h) return 0;
    return p[i]! > p[i + 1]! ? p[i]! : p[i + 1]!;
  };
  const rows: Candidate[] = [];
  const maxS = spacings[spacings.length - 1]!;
  for (let y = 1; y + 4 * maxS + 2 < h; y++) {
    let best = 0;
    let bestS = 0;
    for (const s of spacings) {
      let teeth = 0;
      let low = Infinity;
      for (let k = 0; k < 5; k++) {
        const v = at(y + k * s);
        teeth += v;
        if (v < low) low = v;
      }
      if (low < FIND.minLineShare * (teeth / 5)) continue;
      let gaps = 0;
      for (let k = 0; k < 4; k++) gaps += at(y + (k + 0.5) * s);
      if (teeth / 5 < FIND.minContrast * (gaps / 4)) continue;
      const score = teeth - (5 / 4) * gaps;
      if (score > best) [best, bestS] = [score, s];
    }
    if (best > 0) rows.push({ strip, x, y, s: bestS, score: best });
  }
  rows.sort((a, b) => b.score - a.score);
  const kept: Candidate[] = [];
  for (const r of rows) if (!kept.some((k) => Math.abs(k.y - r.y) < 3.5 * Math.min(k.s, r.s))) kept.push(r);
  return kept;
}

/** Staves of a page (CLAHE'd gray), top to bottom (by page of an open book first). `hint`: the expected spacing. */
export function findStaves(page: Gray, hint: number | null, debug?: Record<string, unknown>): FoundStave[] {
  const s0 = hint ?? estimateSpacing(page);
  if (!s0 || s0 < 4) return [];
  const step = Math.max(3, Math.round(FIND.stripStep * s0));
  const blocks = blockProfiles(page, step, Math.max(2, Math.round(0.3 * s0)));
  const perStrip = Math.max(1, Math.round(FIND.stripWidth / FIND.stripStep));
  const spacings: number[] = [];
  for (let f = FIND.spacingRange[0]; f <= FIND.spacingRange[1] + 1e-9; f += FIND.spacingStep) spacings.push(f * s0);
  const all: Candidate[][] = [];
  for (let i = 0; i + perStrip <= blocks.length; i++) {
    const p = new Float32Array(page.height);
    for (let k = 0; k < perStrip; k++) {
      const b = blocks[i + k]!;
      for (let y = 0; y < p.length; y++) p[y]! += b[y]!;
    }
    all.push(stripCandidates(p, spacings, i, (i + perStrip / 2) * step));
  }
  const scores = all.flat().map((c) => c.score).sort((a, b) => a - b);
  if (!scores.length) return [];
  const strong = scores[Math.floor(0.95 * (scores.length - 1))]!;
  const floor = FIND.minScoreShare * strong;
  if (debug) Object.assign(debug, { s0, strong, floor, above: all.map((st) => st.filter((c) => c.score >= floor).length) });

  // Follow staves strip to strip: each candidate continues the chain it lies on (where the chain was heading).
  interface Chain {
    readonly points: Candidate[];
  }
  const chains: Chain[] = [];
  let active: Chain[] = [];
  for (const strip of all) {
    const taken = new Set<Chain>();
    for (const c of strip.filter((k) => k.score >= floor)) {
      let best: Chain | null = null;
      let bestD = Infinity;
      for (const ch of active) {
        if (taken.has(ch)) continue;
        const last = ch.points[ch.points.length - 1]!;
        if (c.strip - last.strip > FIND.maxGapStrips + 1) continue;
        const prev = ch.points[Math.max(0, ch.points.length - 6)]!;
        const slope = last.x > prev.x ? (last.y - prev.y) / (last.x - prev.x) : 0;
        const d = Math.abs(c.y - (last.y + slope * (c.x - last.x)));
        if (d > FIND.followReach * last.s || Math.abs(c.s - last.s) > FIND.followSpacing * last.s) continue;
        if (d < bestD) [best, bestD] = [ch, d];
      }
      if (best) {
        best.points.push(c);
        taken.add(best);
      } else {
        const ch = { points: [c] };
        chains.push(ch);
        active.push(ch);
        taken.add(ch);
      }
    }
    // Chains left behind for good are not looked at again.
    const now = strip[0]?.strip ?? 0;
    active = active.filter((ch) => now - ch.points[ch.points.length - 1]!.strip <= FIND.maxGapStrips + 1);
  }
  // An open book's gutter: strips near the middle where no stave is found while most strips of the page find some.
  const counts = all.map((st) => st.filter((c) => c.score >= floor).length);
  const usual = median(counts.filter((n) => n > 0));
  let gutterStrip = -1;
  for (let i = Math.round(0.3 * counts.length); i <= Math.round(0.7 * counts.length); i++) {
    if (counts[i]! <= Math.max(0, Math.floor(0.1 * usual)) && counts[i - 1]! <= Math.floor(0.1 * usual) + 1 && counts[i + 1]! <= Math.floor(0.1 * usual) + 1 && (gutterStrip < 0 || Math.abs(i - counts.length / 2) < Math.abs(gutterStrip - counts.length / 2))) gutterStrip = i;
  }
  // Pieces of one stave (broken by a beam, a word, or a stretch read a line off): joined when one runs on where the
  // other ends, its middle line within reach of where the other's is heading.
  const length = (pts: readonly Candidate[]): number => pts[pts.length - 1]!.x - pts[0]!.x + perStrip * step;
  let pieces = chains.map((ch) => ch.points).filter((pts) => pts.length >= 3);
  const midAt = (pts: readonly Candidate[], x: number): number => {
    // The piece's own line near that end: a fit through its nearest eight points.
    const near = [...pts].sort((a, b) => Math.abs(a.x - x) - Math.abs(b.x - x)).slice(0, 8);
    const n = near.length;
    const mx = near.reduce((t, p) => t + p.x, 0) / n;
    const my = near.reduce((t, p) => t + p.y + 2 * p.s, 0) / n;
    let sxx = 0;
    let sxy = 0;
    for (const p of near) {
      sxx += (p.x - mx) ** 2;
      sxy += (p.x - mx) * (p.y + 2 * p.s - my);
    }
    const b = sxx > 0 ? Math.max(-0.2, Math.min(0.2, sxy / sxx)) : 0;
    return my + b * (x - mx);
  };
  for (let merged = true; merged; ) {
    merged = false;
    pieces.sort((a, b) => length(b) - length(a));
    for (let i = 0; i < pieces.length; i++) {
      for (let j = i + 1; j < pieces.length; j++) {
        const a = pieces[i]!;
        const b = pieces[j]!;
        const [l, r] = a[0]!.x <= b[0]!.x ? [a, b] : [b, a];
        const lEnd = l[l.length - 1]!;
        const rStart = r[0]!;
        const overlap = lEnd.strip - rStart.strip;
        if (overlap > 2) continue;
        const gap = rStart.x - lEnd.x;
        const s = (lEnd.s + rStart.s) / 2;
        if (gap > FIND.maxJoinGap * s || (gutterStrip >= 0 && lEnd.strip < gutterStrip && rStart.strip > gutterStrip)) continue;
        // A first look at the ends alone (the fits below are the costly part).
        if (Math.abs(rStart.y - lEnd.y) > 1.6 * s + 0.2 * Math.abs(gap)) continue;
        const xm = (lEnd.x + rStart.x) / 2;
        const d = midAt(r, xm) - midAt(l, xm);
        // Within reach, or a whole line off (one of the two read a line up or down): the weaker is moved onto the other.
        const lines = Math.round(d / s);
        if (Math.abs(lines) > 1 || Math.abs(d - lines * s) > Math.min(0.6, 0.35 + gap / (40 * s)) * s) continue;
        const weight = (pts: readonly Candidate[]): number => pts.reduce((t, p) => t + p.score, 0);
        const move = (pts: readonly Candidate[], dy: number): Candidate[] => pts.map((p) => ({ ...p, y: p.y + dy }));
        const [l2, r2] = lines === 0 ? [l, r] : weight(l) >= weight(r) ? [l, move(r, -lines * s)] : [move(l, lines * s), r];
        pieces[i] = [...l2.filter((p) => p.strip < rStart.strip), ...r2];
        pieces.splice(j, 1);
        merged = true;
        j = i;
      }
    }
  }
  // Each stave followed once more along its fitted path, every strip, at the alignment that holds most ink: a
  // piece read a line off is put back on its lines.
  pieces = pieces.map((pts) => retrack(pts, all, page));
  const longest = Math.max(0, ...pieces.filter((pts) => pts.length >= FIND.minDensity * (pts[pts.length - 1]!.strip - pts[0]!.strip + 1)).map(length));
  // Lines that hold along the stave: found in most strips it crosses (noise strings together a few here and there).
  const dense = (pts: readonly Candidate[]): boolean => pts.length >= FIND.minDensity * (pts[pts.length - 1]!.strip - pts[0]!.strip + 1);
  let staves = pieces.filter((pts) => dense(pts) && length(pts) >= FIND.minLengthSpacings * s0 && length(pts) >= FIND.minLengthShare * longest);
  // As strong as the page's staves: noise strung along a page is weak where every stave is strong.
  const medians = staves.map((pts) => median(pts.map((p) => p.score)));
  const ref = [...medians].sort((a, b) => a - b)[Math.floor(0.8 * (medians.length - 1))] ?? 0;
  staves = staves.filter((_, i) => medians[i]! >= FIND.minStrengthShare * ref);
  // One stave found twice, or a stave and lines beside it (ledger lines, a beam): of two whose lines overlap, the
  // stronger.
  const total = (pts: readonly Candidate[]): number => pts.reduce((a, p) => a + p.score, 0);
  staves.sort((a, b) => total(b) - total(a));
  const kept: Candidate[][] = [];
  for (const pts of staves) {
    const clash = kept.some((k) => {
      const shared = pts.filter((p) => k.some((q) => Math.abs(q.strip - p.strip) <= 1 && Math.abs(q.y - p.y) < 4.5 * p.s));
      return shared.length >= 0.3 * Math.min(pts.length, k.length);
    });
    if (!clash) kept.push(pts);
  }
  staves = kept;
  if (debug) Object.assign(debug, { chains: chains.length, pieces: pieces.length, staves: staves.map((p) => `${Math.round(p[0]!.x)}-${Math.round(p[p.length - 1]!.x)}@${Math.round(p[0]!.y)} n${p.length} d${(p.length / (p[p.length - 1]!.strip - p[0]!.strip + 1)).toFixed(2)} m${Math.round(median(p.map((q) => q.score)))}`) });

  const w = page.width;
  // No stave across a fair part of the page: a close-up at an angle, a blur; nothing is read rather than guesses.
  if (!staves.length || Math.max(...staves.map(length)) < FIND.minPageShare * w) return [];
  const found = staves.map((pts) => fitChain(pts, w, step * perStrip));
  // An open book: a band of columns no stave crosses, near the middle, with staves on both sides.
  const cover = new Int32Array(w);
  for (const f of found) for (let x = Math.max(0, Math.round(f.x0)); x <= Math.min(w - 1, Math.round(f.x1)); x++) cover[x]!++;
  let gutter = -1;
  for (let x = Math.round(0.3 * w); x <= Math.round(0.7 * w); x++) if (cover[x] === 0 && (gutter < 0 || Math.abs(x - w / 2) < Math.abs(gutter - w / 2))) gutter = x;
  const columnOf = (f: { x0: number; x1: number }): number => (gutter >= 0 && (f.x0 + f.x1) / 2 > gutter ? 1 : 0);
  const two = gutter >= 0 && found.filter((f) => columnOf(f) === 0).length >= 2 && found.filter((f) => columnOf(f) === 1).length >= 2;
  return found
    .map((f) => ({ ...f, column: two ? columnOf(f) : 0 }))
    .sort((a, b) => a.column - b.column || a.lines[2]! + ((a.x0 + a.x1) / 2) * a.tilt - (b.lines[2]! + ((b.x0 + b.x1) / 2) * b.tilt));
}

/**
 * A stave's points again from every strip it crosses: the strip's candidate nearest its fitted top line, at the
 * offset (a line up, none, a line down) that the most strips agree with.
 */
function retrack(points: readonly Candidate[], strips: readonly Candidate[][], page: Gray): Candidate[] {
  const height = page.height;
  const fit = fitChain(points, page.width, 0);
  const s = fit.spacing;
  const first = points[0]!.strip;
  const last = points[points.length - 1]!.strip;
  let best: Candidate[] = [...points];
  let bestScore = -1;
  for (const offset of [-1, 0, 1]) {
    const out: Candidate[] = [];
    for (let i = first; i <= last; i++) {
      const strip = strips[i];
      if (!strip?.length) continue;
      const x = strip[0]!.x;
      const want = fit.lines[0]! + x * fit.tilt + fit.bend[Math.max(0, Math.min(page.width - 1, Math.round(x)))]! + offset * s;
      if (want < 0 || want + 4 * s >= height) continue;
      let pick: Candidate | null = null;
      for (const c of strip) if (Math.abs(c.y - want) <= 0.45 * s && Math.abs(c.s - s) <= FIND.followSpacing * s && (!pick || c.score > pick.score)) pick = c;
      if (pick) out.push(pick);
    }
    const score = out.reduce((a, p) => a + p.score, 0);
    if (score > bestScore && out.length >= 3) [best, bestScore] = [out, score];
  }
  return best;
}

/** A straight stave fitted to a chain of top-line points, its bend per column, its extent. */
function fitChain(points: readonly Candidate[], width: number, stripWidth: number): Omit<FoundStave, 'column'> {
  let use = [...points];
  let a = 0;
  let b = 0;
  for (let round = 0; round < 2; round++) {
    const n = use.length;
    const mx = use.reduce((t, p) => t + p.x, 0) / n;
    const my = use.reduce((t, p) => t + p.y, 0) / n;
    let sxx = 0;
    let sxy = 0;
    for (const p of use) {
      sxx += (p.x - mx) ** 2;
      sxy += (p.x - mx) * (p.y - my);
    }
    b = sxx > 0 ? sxy / sxx : 0;
    a = my - b * mx;
    const spacing = median(use.map((p) => p.s));
    const inliers = use.filter((p) => Math.abs(p.y - (a + b * p.x)) <= 1.5 * spacing);
    if (inliers.length >= 3) use = inliers;
  }
  const spacing = median(points.map((p) => p.s));
  const bend = new Float32Array(width);
  const xs = points.map((p) => p.x);
  const ds = points.map((p) => p.y - (a + b * p.x));
  // Smoothed: a median of five points.
  const smooth = ds.map((_, k) => median(ds.slice(Math.max(0, k - 2), k + 3)));
  for (let x = 0; x < width; x++) {
    if (x <= xs[0]!) bend[x] = smooth[0]!;
    else if (x >= xs[xs.length - 1]!) bend[x] = smooth[smooth.length - 1]!;
    else {
      let k = 1;
      while (xs[k]! < x) k++;
      const f = (x - xs[k - 1]!) / Math.max(1e-6, xs[k]! - xs[k - 1]!);
      bend[x] = smooth[k - 1]! + f * (smooth[k]! - smooth[k - 1]!);
    }
  }
  return {
    lines: [0, 1, 2, 3, 4].map((k) => a + k * spacing),
    tilt: b,
    spacing,
    x0: Math.max(0, xs[0]! - stripWidth / 2),
    x1: Math.min(width - 1, xs[xs.length - 1]! + stripWidth / 2),
    bend,
  };
}

function median(v: readonly number[]): number {
  const s = [...v].sort((p, q) => p - q);
  return s.length ? s[s.length >> 1]! : 0;
}
