/**
 * shared/wasm-threads.js
 * Un solo hilo WASM. GitHub Pages y `python -m http.server` no pueden
 * enviar COOP ni COEP, así que no hay crossOriginIsolated ni SharedArrayBuffer.
 * El WASM multihilo de ONNX Runtime aborta en ese caso. Un hilo arranca igual.
 */

/** onnxruntime-web importado directo (Stable Diffusion). */
export function pinOrtThreads(ort) {
  if (!ort?.env?.wasm) return;
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.proxy = false;
}

/** `env` de transformers.js (LLM, voz, tokenizador CLIP). */
export function pinTransformersThreads(env) {
  const wasm = env?.backends?.onnx?.wasm;
  if (!wasm) return;
  wasm.numThreads = 1;
}
