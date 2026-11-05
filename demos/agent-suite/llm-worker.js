/**
 * demos/agent-suite/llm-worker.js
 * Carga el LLM con transformers.js y emite tokens mientras genera.
 * El id y el dtype llegan en el mensaje "load": están definidos en shared/models.js.
 */

import { SYSTEM_PROMPT, TRANSFORMERS_URL } from "../../shared/models.js";
import { pinTransformersThreads } from "../../shared/wasm-threads.js";

let generator = null;
let activeKey = null;

function post(message) {
  self.postMessage(message);
}

self.onmessage = async (event) => {
  const message = event.data;
  try {
    if (message.type === "load") {
      const preset = message.preset;
      activeKey = preset.key;
      const files = new Map();
      const started = performance.now();
      let networkLoaded = 0;
      const { pipeline, env, TextStreamer } = await import(TRANSFORMERS_URL);
      env.allowLocalModels = false;
      env.useBrowserCache = true;
      pinTransformersThreads(env);

      post({ type: "status", message: `Abriendo ${preset.modelId} (${preset.dtype})…` });
      generator = await pipeline("text-generation", preset.modelId, {
        dtype: preset.dtype,
        device: "webgpu",
        progress_callback(progress) {
          if (progress.file) {
            const previous = files.get(progress.file) || { loaded: 0 };
            const loadedNow = progress.loaded || previous.loaded;
            if (progress.status === "progress" || progress.status === "download" || progress.status === "done") {
              networkLoaded += Math.max(0, loadedNow - previous.loaded);
            }
            files.set(progress.file, { loaded: loadedNow, total: progress.total || 0 });
          }
          let loaded = 0;
          for (const file of files.values()) loaded += file.loaded;
          const seconds = Math.max((performance.now() - started) / 1000, 0.05);
          const remain = Math.max(0, preset.bytes - loaded);
          const etaMs = networkLoaded > 1024 * 256 ? (remain / (networkLoaded / seconds)) * 1000 : null;
          post({
            type: "progress",
            loaded,
            total: preset.bytes,
            percent: Math.min(99, preset.bytes ? (loaded / preset.bytes) * 100 : 0),
            etaMs,
            file: progress.file || progress.status || "",
            status: progress.status,
          });
        },
      });

      post({
        type: "progress",
        loaded: preset.bytes,
        total: preset.bytes,
        percent: 100,
        etaMs: 0,
        file: "listo",
        status: "ready",
      });
      post({ type: "ready", key: preset.key });
      // TextStreamer se crea por petición; la referencia evita que el import se sacuda.
      self.__TextStreamer = TextStreamer;
      return;
    }

    if (message.type === "generate") {
      if (!generator) throw new Error("El modelo de lenguaje todavía no está cargado.");
      const TextStreamer = self.__TextStreamer;
      let streamed = "";
      const streamer = new TextStreamer(generator.tokenizer, {
        skip_prompt: true,
        skip_special_tokens: true,
        callback_function(text) {
          streamed += text;
          post({ type: "token", jobId: message.jobId, text });
        },
      });
      const messages = [
        {
          role: "system",
          content: message.allowTools === false
            ? `${SYSTEM_PROMPT}\n\nThis follow-up turn has no tools. Do not emit a tool block. Reply in one or two short sentences.`
            : SYSTEM_PROMPT,
        },
        ...message.messages,
      ];
      const output = await generator(messages, {
        max_new_tokens: 384,
        do_sample: false,
        repetition_penalty: 1.05,
        streamer,
      });
      const finalText = output?.[0]?.generated_text?.at?.(-1)?.content ?? streamed;
      post({ type: "done", jobId: message.jobId, text: finalText || streamed });
    }
  } catch (error) {
    post({
      type: "error",
      jobId: message.jobId ?? null,
      message: error instanceof Error ? error.stack || error.message : String(error),
    });
  }
};
