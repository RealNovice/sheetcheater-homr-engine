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
import { preparePage, readPage, type HostStave } from './engine';
import type { OrtLike, OrtSessionLike } from './staff2score';

interface Scope {
  onmessage: ((e: MessageEvent) => void) | null;
  postMessage(message: unknown, transfer?: Transferable[]): void;
  importScripts(...urls: string[]): void;
  ort?: OrtLike & { env: { wasm: { wasmPaths: string; numThreads: number }; logLevel: string }; InferenceSession: { create(model: Uint8Array, options: { executionProviders: string[] }): Promise<OrtSessionLike> } };
  navigator: { gpu?: unknown };
}
const scope = self as unknown as Scope;

let sessions: { ort: OrtLike; encoder: OrtSessionLike; decoder: OrtSessionLike } | null = null;
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
      const encoder = await ort.InferenceSession.create(new Uint8Array(m.encoder as ArrayBuffer), { executionProviders: [backend] });
      const decoder = await ort.InferenceSession.create(new Uint8Array(m.decoder as ArrayBuffer), { executionProviders: [backend] });
      sessions = { ort, encoder, decoder };
      scope.postMessage({ type: 'ready', backend });
    } else if (m.type === 'cancel') {
      cancelled.add(m.id!);
    } else if (m.type === 'read') {
      if (!sessions) throw new Error('The engine is not started.');
      const id = m.id!;
      const img = m.image as { width: number; height: number; data: ArrayBuffer };
      const page = preparePage({ width: img.width, height: img.height, data: new Uint8ClampedArray(img.data) });
      const readings = await readPage(sessions.ort, sessions.encoder, sessions.decoder, page, m.staves as HostStave[], (done, total) => scope.postMessage({ type: 'progress', id, done, total }), () => cancelled.has(id));
      cancelled.delete(id);
      // The symbols stay here: the host takes the timeline.
      scope.postMessage({ type: 'result', id, readings: readings.map(({ symbols: _symbols, ...r }) => r) });
    }
  } catch (err) {
    scope.postMessage({ type: 'error', id: m.id, message: String((err as Error)?.message ?? err) });
  }
};
scope.postMessage({ type: 'loaded' });
