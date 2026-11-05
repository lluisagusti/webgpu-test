/**
 * demos/agent-suite/tts-worker.js
 * Voz con el pipeline text-to-speech de transformers.js (MMS, español por defecto).
 * Si WebGPU rechaza el grafo, reintenta en WASM y lo dice en el estado.
 */

import { TRANSFORMERS_URL, TTS_DTYPE, TTS_EXPECTED_BYTES, TTS_MODEL_ID } from "../../shared/models.js";
import { pinTransformersThreads } from "../../shared/wasm-threads.js";

let synthesizer = null;
let deviceUsed = null;

function post(message, transfer) {
  self.postMessage(message, transfer ?? []);
}

async function load(device) {
  const { pipeline, env } = await import(TRANSFORMERS_URL);
  env.allowLocalModels = false;
  env.useBrowserCache = true;
  pinTransformersThreads(env);
  const files = new Map();
  const started = performance.now();
  let networkLoaded = 0;
  const model = await pipeline("text-to-speech", TTS_MODEL_ID, {
    dtype: TTS_DTYPE,
    device,
    progress_callback(progress) {
      if (progress.file) {
        const previous = files.get(progress.file) || 0;
        const loadedNow = progress.loaded || previous;
        networkLoaded += Math.max(0, loadedNow - previous);
        files.set(progress.file, loadedNow);
      }
      let loaded = 0;
      for (const value of files.values()) loaded += value;
      const total = Math.max(TTS_EXPECTED_BYTES, loaded);
      const seconds = Math.max((performance.now() - started) / 1000, 0.05);
      const remain = Math.max(0, TTS_EXPECTED_BYTES - loaded);
      post({
        type: "progress",
        loaded,
        total,
        percent: Math.min(100, (loaded / total) * 100),
        etaMs: networkLoaded > 1024 * 32 && remain > 0 ? (remain / (networkLoaded / seconds)) * 1000 : 0,
        file: progress.file || progress.status || "",
        status: progress.status,
      });
    },
  });
  return model;
}

function samplesFrom(output) {
  const rate = output.sampling_rate || output.sample_rate || 16000;
  let audio = output.audio ?? output.waveform;
  if (audio && audio.data) audio = audio.data;
  const samples = audio instanceof Float32Array ? new Float32Array(audio) : Float32Array.from(audio);
  return { samples, rate };
}

self.onmessage = async (event) => {
  const message = event.data;
  try {
    if (message.type === "load") {
      post({ type: "status", message: `Descargando ${TTS_MODEL_ID}…` });
      try {
        synthesizer = await load("webgpu");
        deviceUsed = "webgpu";
      } catch (error) {
        post({
          type: "status",
          message: `WebGPU rechazó el TTS (${error instanceof Error ? error.message : error}). Reintentando en WASM…`,
        });
        synthesizer = await load("wasm");
        deviceUsed = "wasm";
      }
      post({ type: "ready", device: deviceUsed });
      return;
    }

    if (message.type === "speak") {
      if (!synthesizer) throw new Error("La voz todavía no está cargada.");
      const text = String(message.text || "").slice(0, 280);
      const output = await synthesizer(text);
      const { samples, rate } = samplesFrom(output);
      post({ type: "audio", jobId: message.jobId, sampleRate: rate, samples, device: deviceUsed }, [samples.buffer]);
    }
  } catch (error) {
    post({
      type: "error",
      jobId: message.jobId ?? null,
      message: error instanceof Error ? error.stack || error.message : String(error),
    });
  }
};
