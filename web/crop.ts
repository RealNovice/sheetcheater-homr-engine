/**
 * homr's staff image preparation (homr/staff_parsing.py, prepare_staff_image and the helpers it uses; homr's
 * color_adjust.apply_clahe), ported to TypeScript for the browser. AGPL-3.0, like homr (../LICENSE).
 *
 * Differences from homr: the page comes already straightened, with its staves found, by the host app; homr's own
 * staff detection and per-staff dewarping are not run. A staff is given as five line heights at x = 0 plus a tilt,
 * and its horizontal extent.
 */

/** One byte per pixel, row-major, 0 black. */
export interface Gray {
  readonly width: number;
  readonly height: number;
  readonly data: Uint8Array;
}

export interface StaffGeometry {
  /** Line ys at x = 0, top to bottom (five, or ten for a grand staff read as one). */
  readonly lines: readonly number[];
  readonly tilt: number;
  readonly spacing: number;
  readonly x0: number;
  readonly x1: number;
}

/** The transformer's input canvas (homr's TrOMR config). */
export const CANVAS = { width: 1280, height: 256, mean: 0.7931, std: 0.1738 } as const;

/** RGBA to gray as OpenCV's BGR2GRAY. */
export function toGray(rgba: { width: number; height: number; data: Uint8Array | Uint8ClampedArray }): Gray {
  const out = new Uint8Array(rgba.width * rgba.height);
  const d = rgba.data;
  for (let i = 0, j = 0; i < out.length; i++, j += 4) out[i] = Math.round(0.299 * d[j]! + 0.587 * d[j + 1]! + 0.114 * d[j + 2]!);
  return { width: rgba.width, height: rgba.height, data: out };
}

/**
 * Contrast-limited adaptive histogram equalisation as homr applies it (OpenCV CLAHE, clip limit 1.0, an 8 × 8
 * grid of tiles), with bilinear blending between the tiles' lookup tables.
 */
export function clahe(gray: Gray, clipLimit = 1.0, grid = 8): Gray {
  const { width, height, data } = gray;
  const tw = Math.ceil(width / grid);
  const th = Math.ceil(height / grid);
  const luts = new Uint8Array(grid * grid * 256);
  const hist = new Int32Array(256);
  for (let ty = 0; ty < grid; ty++) {
    for (let tx = 0; tx < grid; tx++) {
      hist.fill(0);
      const x0 = tx * tw;
      const y0 = ty * th;
      const x1 = Math.min(width, x0 + tw);
      const y1 = Math.min(height, y0 + th);
      const area = Math.max(1, (x1 - x0) * (y1 - y0));
      for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) hist[data[y * width + x]!]!++;
      const limit = Math.max(1, Math.trunc((clipLimit * area) / 256));
      let clipped = 0;
      for (let i = 0; i < 256; i++) {
        if (hist[i]! > limit) {
          clipped += hist[i]! - limit;
          hist[i] = limit;
        }
      }
      const batch = Math.trunc(clipped / 256);
      const residual = clipped - batch * 256;
      for (let i = 0; i < 256; i++) hist[i]! += batch;
      if (residual > 0) {
        const step = Math.max(1, Math.trunc(256 / residual));
        for (let i = 0, r = residual; i < 256 && r > 0; i += step, r--) hist[i]!++;
      }
      const scale = 255 / area;
      let sum = 0;
      const base = (ty * grid + tx) * 256;
      for (let i = 0; i < 256; i++) {
        sum += hist[i]!;
        luts[base + i] = Math.min(255, Math.max(0, Math.round(sum * scale)));
      }
    }
  }
  const out = new Uint8Array(width * height);
  for (let y = 0; y < height; y++) {
    const fy = (y + 0.5) / th - 0.5;
    const ty0 = Math.max(0, Math.min(grid - 1, Math.floor(fy)));
    const ty1 = Math.min(grid - 1, ty0 + 1);
    const ay = Math.max(0, Math.min(1, fy - ty0));
    for (let x = 0; x < width; x++) {
      const fx = (x + 0.5) / tw - 0.5;
      const tx0 = Math.max(0, Math.min(grid - 1, Math.floor(fx)));
      const tx1 = Math.min(grid - 1, tx0 + 1);
      const ax = Math.max(0, Math.min(1, fx - tx0));
      const v = data[y * width + x]!;
      const a = luts[(ty0 * grid + tx0) * 256 + v]!;
      const b = luts[(ty0 * grid + tx1) * 256 + v]!;
      const c = luts[(ty1 * grid + tx0) * 256 + v]!;
      const d = luts[(ty1 * grid + tx1) * 256 + v]!;
      out[y * width + x] = Math.round((1 - ay) * ((1 - ax) * a + ax * b) + ay * ((1 - ax) * c + ax * d));
    }
  }
  return { width, height, data: out };
}

/** Bilinear sample of a gray image (outside: white). */
function sample(g: Gray, x: number, y: number): number {
  if (x < 0 || y < 0 || x > g.width - 1 || y > g.height - 1) return 255;
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const x1 = Math.min(g.width - 1, x0 + 1);
  const y1 = Math.min(g.height - 1, y0 + 1);
  const fx = x - x0;
  const fy = y - y0;
  const d = g.data;
  const w = g.width;
  return (1 - fy) * ((1 - fx) * d[y0 * w + x0]! + fx * d[y0 * w + x1]!) + fy * ((1 - fx) * d[y1 * w + x0]! + fx * d[y1 * w + x1]!);
}

/**
 * How far a staff's lines really stand from where the host put them, column by column (homr dewarps every staff
 * on its own, staff_dewarping.py; this follows the lines instead): at every spacing along the staff the vertical
 * shift within reach of the last one that puts most ink under five lines, tracked out from the middle and
 * smoothed. Returns a shift per column of the page (0 outside the staff).
 */
export function lineOffsets(page: Gray, staff: StaffGeometry): Float32Array {
  const w = page.width;
  const out = new Float32Array(w);
  const s = staff.spacing;
  const five = staff.lines.slice(0, Math.min(staff.lines.length, 10));
  const ink = (x0: number, dy: number): number => {
    let sum = 0;
    let n = 0;
    for (let x = Math.max(0, x0 - Math.round(s)); x <= Math.min(w - 1, x0 + Math.round(s)); x += 2) {
      for (const l of five) {
        const y = Math.round(l + x * staff.tilt + dy);
        if (y >= 0 && y < page.height) {
          sum += 255 - page.data[y * w + x]!;
          n++;
        }
      }
    }
    return n ? sum / n : 0;
  };
  const step = Math.max(4, Math.round(s));
  const xs: number[] = [];
  for (let x = Math.max(0, Math.round(staff.x0)); x <= Math.min(w - 1, Math.round(staff.x1)); x += step) xs.push(x);
  if (!xs.length) return out;
  const dys = new Float32Array(xs.length);
  const best = (x: number, centre: number, reach: number): number => {
    let top = centre;
    let topInk = -1;
    for (let dy = centre - reach; dy <= centre + reach; dy += 0.5) {
      const v = ink(x, dy);
      if (v > topInk) {
        topInk = v;
        top = dy;
      }
    }
    return top;
  };
  const mid = Math.trunc(xs.length / 2);
  dys[mid] = best(xs[mid]!, 0, 0.6 * s);
  for (let k = mid + 1; k < xs.length; k++) dys[k] = best(xs[k]!, dys[k - 1]!, 0.2 * s);
  for (let k = mid - 1; k >= 0; k--) dys[k] = best(xs[k]!, dys[k + 1]!, 0.2 * s);
  // A median of five steps, then straight lines between the steps.
  const smooth = Array.from(dys, (_, k) => [...dys.slice(Math.max(0, k - 2), k + 3)].sort((a, b) => a - b)[Math.trunc(Math.min(5, Math.min(k + 3, xs.length) - Math.max(0, k - 2)) / 2)]!);
  for (let x = 0; x < w; x++) {
    if (x <= xs[0]!) out[x] = smooth[0]!;
    else if (x >= xs[xs.length - 1]!) out[x] = smooth[xs.length - 1]!;
    else {
      const k = Math.min(xs.length - 2, Math.floor((x - xs[0]!) / step));
      const f = (x - xs[k]!) / step;
      out[x] = smooth[k]! + f * (smooth[k + 1]! - smooth[k]!);
    }
  }
  return out;
}

/**
 * The transformer's input for one staff: the staff's region (2 spacings either side, 4 above and below, but not
 * into the staves next to it), scaled to fit 1280 × 256 keeping its shape, left-aligned and centred vertically on
 * white, normalised as homr does. `neighbours` are the line spans (top, bottom at the staff's middle) of the other
 * staves of the page. Returns the tensor data, 1 × 1 × 256 × 1280.
 */
export function staffInput(page: Gray, staff: StaffGeometry, neighbours: readonly { readonly top: number; readonly bottom: number }[], lineShift?: Float32Array | readonly [Float32Array, Float32Array]): { data: Float32Array; region: { x0: number; y0: number; x1: number; y1: number }; scale: number } {
  const u = staff.spacing;
  const xm = (staff.x0 + staff.x1) / 2;
  const top = staff.lines[0]! + xm * staff.tilt;
  const bottom = staff.lines[staff.lines.length - 1]! + xm * staff.tilt;
  const above = Math.max(0, ...neighbours.filter((n) => n.top < top).map((n) => n.bottom));
  const below = Math.min(1e12, ...neighbours.filter((n) => n.bottom > bottom).map((n) => n.top));
  const region = {
    x0: Math.max(0, staff.x0 - 2 * u),
    x1: Math.min(page.width, staff.x1 + 2 * u),
    y0: Math.max(0, Math.max(top - 4 * u, above)),
    y1: Math.min(page.height, Math.min(bottom + 4 * u, below)),
  };
  const w = Math.max(1, region.x1 - region.x0);
  const h = Math.max(1, region.y1 - region.y0);
  // get_tr_omr_canvas_size: fit the region into the canvas, keeping its shape.
  const [cw, ch] = h / w > CANVAS.height / CANVAS.width ? [Math.trunc((w / h) * CANVAS.height), CANVAS.height] : [CANVAS.width, Math.trunc((h / w) * CANVAS.width)];
  const scale = ch / h;
  const yOffset = Math.trunc((CANVAS.height - ch) / 2);
  // A grand staff (ten lines) bends as its two staves do: the upper's shift above its middle line, the lower's below
  // its own, blended between.
  const pair = Array.isArray(lineShift) ? (lineShift as readonly [Float32Array, Float32Array]) : null;
  const single = pair ? null : ((lineShift as Float32Array | undefined) ?? lineOffsets(page, staff));
  const upperMid = staff.lines[2]! + xm * staff.tilt;
  const lowerMid = (staff.lines[7] ?? staff.lines[2]!) + xm * staff.tilt;
  const shiftAt = (x: number, y: number): number => {
    const k = Math.max(0, Math.min(page.width - 1, Math.round(x)));
    if (single) return single[k]!;
    const f = Math.max(0, Math.min(1, (y - upperMid) / Math.max(1, lowerMid - upperMid)));
    return (1 - f) * pair![0][k]! + f * pair![1][k]!;
  };
  const out = new Float32Array(CANVAS.width * CANVAS.height);
  const white = (1 - CANVAS.mean) / CANVAS.std;
  out.fill(white);
  // The staff's tilt is taken out as it is sampled: the canvas row follows the lines.
  for (let y = 0; y < ch; y++) {
    for (let x = 0; x < cw; x++) {
      const px = region.x0 + (x + 0.5) / (cw / w) - 0.5;
      const py0 = region.y0 + (y + 0.5) / scale - 0.5 + (px - xm) * staff.tilt;
      const py = py0 + shiftAt(px, py0);
      out[(y + yOffset) * CANVAS.width + x] = (sample(page, px, py) / 255 - CANVAS.mean) / CANVAS.std;
    }
  }
  return { data: out, region, scale };
}

/**
 * Where a staff's lines run, left to right: the columns where its lines stand out darker than the paper between
 * them, the longest stretch of such columns (gaps under three spacings bridged). For a page whose staves come
 * without an extent.
 */
export function staffExtent(page: Gray, lines: readonly number[], tilt: number, spacing: number, shift?: Float32Array, contrast = 18): { x0: number; x1: number } {
  const w = page.width;
  const at = (x: number, y: number): number => {
    const yy = Math.round(y);
    return yy >= 0 && yy < page.height ? page.data[yy * w + x]! : 255;
  };
  const five = lines.slice(0, 5);
  const on = new Uint8Array(w);
  for (let x = 0; x < w; x++) {
    let line = 0;
    let gap = 0;
    for (const [k, l] of five.entries()) {
      const y = l + x * tilt + (shift ? shift[x]! : 0);
      line += Math.min(at(x, y - 2), at(x, y - 1), at(x, y), at(x, y + 1), at(x, y + 2));
      if (k < 4) gap += at(x, y + spacing / 2);
    }
    on[x] = gap / 4 - line / 5 > contrast ? 1 : 0;
  }
  const win = Math.max(3, Math.round(2 * spacing));
  const ok = new Uint8Array(w);
  let count = 0;
  for (let x = 0; x < w; x++) {
    count += on[x]!;
    if (x >= win) count -= on[x - win]!;
    if (x >= win - 1 && count >= 0.6 * win) for (let k = x - win + 1; k <= x; k++) ok[k] = 1;
  }
  let best = { x0: 0, x1: w - 1, len: 0 };
  let start = -1;
  let lastOn = -1;
  const bridge = 3 * spacing;
  for (let x = 0; x <= w; x++) {
    if (x < w && ok[x]) {
      if (start < 0 || x - lastOn > bridge) start = x;
      lastOn = x;
      if (lastOn - start > best.len) best = { x0: start, x1: lastOn, len: lastOn - start };
    }
  }
  return best.len > 0 ? { x0: best.x0, x1: best.x1 } : { x0: 0, x1: w - 1 };
}

/**
 * Whether two staves one above the other are joined into one system: a line drawn from the first down to the
 * second at their left ends (the system's start line or brace; homr's brace_dot_detection looks for the same).
 * Looks at the columns about the staves' common start for one that is ink nearly all the way between them.
 */
export function joined(page: Gray, upper: { bottom: number; x0: number; spacing: number }, lower: { top: number; x0: number }, threshold = 150): boolean {
  const x0 = Math.min(upper.x0, lower.x0);
  const s = upper.spacing;
  const y0 = Math.round(upper.bottom);
  const y1 = Math.round(lower.top);
  if (y1 - y0 < 2) return true;
  for (let x = Math.max(0, Math.round(x0 - 2.5 * s)); x <= Math.min(page.width - 1, Math.round(x0 + 1.5 * s)); x++) {
    let dark = 0;
    for (let y = y0; y <= y1; y++) {
      let d = false;
      for (let dx = -1; dx <= 1 && !d; dx++) {
        const xx = x + dx;
        if (xx >= 0 && xx < page.width && y >= 0 && y < page.height && page.data[y * page.width + xx]! < threshold) d = true;
      }
      if (d) dark++;
    }
    if (dark >= 0.92 * (y1 - y0 + 1)) return true;
  }
  return false;
}
