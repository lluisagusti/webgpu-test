/**
 * shared/webgpu.js
 * Comprueba WebGPU y la feature shader-f16, obligatoria para los modelos fp16.
 */

export function isFileProtocol() {
  return globalThis.location?.protocol === "file:";
}

/**
 * @returns {Promise<{
 *   ok: boolean,
 *   f16: boolean,
 *   reason: string,
 *   detail: string
 * }>}
 */
export async function checkWebGPU() {
  if (isFileProtocol()) {
    return {
      ok: false,
      f16: false,
      reason: "file",
      detail: "La página está en file://. Sirve el repositorio por HTTP.",
    };
  }
  if (!navigator.gpu) {
    return {
      ok: false,
      f16: false,
      reason: "no-gpu",
      detail: "Este navegador no expone navigator.gpu. Hace falta Chromium o Safari reciente, con WebGPU activado.",
    };
  }

  let adapter = null;
  try {
    adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) adapter = await navigator.gpu.requestAdapter();
  } catch (error) {
    return {
      ok: false,
      f16: false,
      reason: "adapter-error",
      detail: `requestAdapter falló: ${error instanceof Error ? error.message : String(error)}`,
    };
  }

  if (!adapter) {
    return {
      ok: false,
      f16: false,
      reason: "no-adapter",
      detail: "Hay API WebGPU, pero ningún adaptador. Revisa los controladores de la GPU o las flags del navegador.",
    };
  }

  const f16 = adapter.features?.has?.("shader-f16") === true;
  let detail = f16
    ? "WebGPU listo, con shader-f16."
    : "Hay WebGPU, pero falta shader-f16. Los modelos en fp16 no pueden correr.";
  try {
    const info = adapter.info;
    if (info && (info.vendor || info.architecture || info.device)) {
      const bits = [info.vendor, info.architecture, info.device].filter(Boolean).join(" · ");
      if (bits) detail = `${detail} Adaptador: ${bits}.`;
    }
  } catch {
    /* info es opcional */
  }

  return { ok: f16, f16, reason: f16 ? "ok" : "no-f16", detail };
}
