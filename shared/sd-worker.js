/**
 * shared/sd-worker.js
 * Worker de Stable Diffusion. La descarga y la inferencia no bloquean la UI.
 *
 * Mensajes de entrada: download, load, generate, release, clear-cache.
 * Mensajes de salida: progress, download-done, loaded, step, image, error, cache-cleared, status.
 */

import { ORT_URL, SD_MODELS, hfResolve } from "./models.js";
import { clearModelCache, fetchCached, trackSet } from "./download.js";
import { createSdRuntime, pngFromRgba } from "./sd-engine.js";

let runtime = null;
try {
  const ort = await import(ORT_URL);
  runtime = createSdRuntime(ort);
} catch (error) {
  self.postMessage({
    type: "error",
    modelKey: null,
    jobId: null,
    message: `No se pudo cargar ONNX Runtime Web: ${error instanceof Error ? error.message : String(error)}`,
  });
}

function post(message, transfer) {
  self.postMessage(message, transfer ?? []);
}

self.onmessage = async (event) => {
  const message = event.data;
  if (!runtime && message.type !== "clear-cache") {
    post({
      type: "error",
      modelKey: message.modelKey ?? null,
      jobId: message.jobId ?? null,
      message: "ONNX Runtime Web no llegó a cargarse.",
    });
    return;
  }
  try {
    if (message.type === "download") {
      const model = SD_MODELS[message.modelKey];
      const tracker = trackSet(model.files);
      for (const file of model.files) {
        const url = hfResolve(model.repo, file.path);
        await fetchCached(url, {
          expectedBytes: file.bytes,
          signal: undefined,
          onProgress: (progress) => {
            post({
              type: "progress",
              modelKey: model.key,
              ...tracker.note(file.path, progress),
            });
          },
        });
      }
      post({ type: "download-done", modelKey: model.key });
      return;
    }

    if (message.type === "load") {
      await runtime.load(message.modelKey, (text) => {
        post({ type: "status", modelKey: message.modelKey, message: text });
      });
      post({ type: "loaded", modelKey: message.modelKey });
      return;
    }

    if (message.type === "generate") {
      const image = await runtime.generate({
        modelKey: message.modelKey,
        prompt: message.prompt,
        negative: message.negative,
        steps: message.steps,
        guidance: message.guidance,
        seed: message.seed,
        onStatus: (text) => post({ type: "status", modelKey: message.modelKey, message: text }),
        onStep: (step, total) => post({ type: "step", modelKey: message.modelKey, step, total }),
      });
      const buffer = await pngFromRgba(image.rgba, image.width, image.height);
      post(
        {
          type: "image",
          modelKey: message.modelKey,
          jobId: message.jobId,
          width: image.width,
          height: image.height,
          seed: image.seed,
          buffer,
        },
        [buffer],
      );
      return;
    }

    if (message.type === "release") {
      await runtime.release();
      post({ type: "released" });
      return;
    }

    if (message.type === "clear-cache") {
      if (runtime) await runtime.release();
      await clearModelCache();
      post({ type: "cache-cleared" });
    }
  } catch (error) {
    post({
      type: "error",
      modelKey: message.modelKey ?? null,
      jobId: message.jobId ?? null,
      message: error instanceof Error ? error.stack || error.message : String(error),
    });
  }
};
