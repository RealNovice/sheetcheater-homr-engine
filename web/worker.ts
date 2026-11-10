/**
 * The homr engine's Web Worker (the host app starts it with `new Worker(new URL(...))` and talks to it only by
 * messages). AGPL-3.0, like homr (../LICENSE).
 *
 *   → { type: 'init', cdn, backend: 'wasm' | 'webgpu', threads, encoder: ArrayBuffer, decoder: ArrayBuffer }
 *   ← { type: 'ready', backend } | { type: 'error', message }
 *   → { type: 'read', id, image: { width, height, data: ArrayBuffer (RGBA) }, staves: [{ lines, tilt, spacing }] }
 *   ← { type: 'progress', id, done, total } … { type: 'result', id, readings } | { type: 'error', id, message }
 *   → { type: 'cancel', id }
 *
 * onnxruntime-web is loaded from its CDN inside the worker (as the host's seg_net worker does).
 */
import { readImage, type HostStave } from './engine';
import { withLastRows } from './lastRows';
import type { OrtLike, OrtSessionLike } from './staff2score';

interface Scope {
  onmessage: ((e: MessageEvent) => void) | null;
  postMessage(message: unknown, transfer?: Transferable[]): void;
  importScripts(...urls: string[]): void;
  ort?: OrtLike & { env: { wasm: { wasmPaths: string; numThreads: number }; logLevel: string }; InferenceSession: { create(model: Uint8Array, options: { executionProviders: string[]; enableCpuMemArena?: boolean; enableMemPattern?: boolean }): Promise<OrtSessionLike> } };
  navigator: { gpu?: unknown };
}
const scope = self as unknown as Scope;

let sessions: { ort: OrtLike; encoder: OrtSessionLike; decoder: OrtSessionLike; oneAtATime: boolean } | null = null;

/**
 * The decoder made on its first run (low-memory mode, task 0023 of the host app): the encoder has read every staff
 * of the page and been released by then, so the two models are never in memory together. The host lets this worker
 * go after the page.
 */
function laterSession(create: () => Promise<OrtSessionLike>): OrtSessionLike {
  let session: Promise<OrtSessionLike> | null = null;
  return {
    async run(feeds, fetches) {
      session ??= create();
      const s = await session;
      return fetches ? s.run(feeds, fetches) : s.run(feeds);
    },
    async release() {
      const s = await session;
      session = null;
      await s?.release?.();
    },
  };
}
const cancelled = new Set<number>();

scope.onmessage = async (e: MessageEvent) => {
  const m = e.data as { type: string; id?: number; [k: string]: unknown };
  try {
    if (m.type === 'init') {
      const backend = m.backend as 'wasm' | 'webgpu';
      if (backend === 'webgpu' && !scope.navigator.gpu) throw new Error('WebGPU is not available in this browser');
      scope.importScripts(`${m.cdn as string}${backend === 'webgpu' ? 'ort.webgpu.min.js' : 'ort.min.js'}`);
      const ort = scope.ort;
      if (!ort) throw new Error('onnxruntime-web did not register');
      ort.env.wasm.wasmPaths = m.cdn as string;
      ort.env.wasm.numThreads = m.threads as number;
      ort.env.logLevel = 'error';
      // Low-memory mode (the host decides; task 0020 of the host app): no memory arena, no planned memory pattern.
      const options = m.lowMemory ? { executionProviders: [backend], enableCpuMemArena: false, enableMemPattern: false } : { executionProviders: [backend] };
      const encoder = await ort.InferenceSession.create(new Uint8Array(m.encoder as ArrayBuffer), options);
      // The caches' last rows as outputs, so a step's caches are not copied out whole (lastRows.ts).
      let decoderModel: Uint8Array | null = withLastRows(new Uint8Array(m.decoder as ArrayBuffer));
      const makeDecoder = (): Promise<OrtSessionLike> => {
        const model = decoderModel!;
        decoderModel = null;
        return ort.InferenceSession.create(model, options);
      };
      const decoder = m.lowMemory ? laterSession(makeDecoder) : await makeDecoder();
      sessions = { ort, encoder, decoder, oneAtATime: !!m.lowMemory };
      scope.postMessage({ type: 'ready', backend });
    } else if (m.type === 'cancel') {
      cancelled.add(m.id!);
    } else if (m.type === 'read') {
      if (!sessions) throw new Error('The engine is not started.');
      const id = m.id!;
      const img = m.image as { width: number; height: number; data: ArrayBuffer };
      const readings = await readImage(sessions.ort, sessions.encoder, sessions.decoder, { width: img.width, height: img.height, data: new Uint8ClampedArray(img.data) }, m.staves as HostStave[], (done, total) => scope.postMessage({ type: 'progress', id, done, total }), () => cancelled.has(id), { oneAtATime: sessions.oneAtATime });
      // One page only in low-memory mode: its encoder is gone.
      if (sessions.oneAtATime) sessions = null;
      cancelled.delete(id);
      // The symbols stay here: the host takes the timeline.
      scope.postMessage({ type: 'result', id, readings: readings.map(({ symbols: _symbols, ...r }) => r) });
    }
  } catch (err) {
    scope.postMessage({ type: 'error', id: m.id, message: String((err as Error)?.message ?? err) });
  }
};
scope.postMessage({ type: 'loaded' });
