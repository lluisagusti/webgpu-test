/**
 * demos/agent-suite/main.js
 * Bucle del agente: mensaje → LLM (con tokens) → generate_image y/o speak.
 * Cada modelo vive en su worker para que las barras sigan moviéndose.
 */

import {
  DEFAULT_LLM_PRESET,
  LLM_PRESETS,
  SD_MODELS,
  TRANSFORMERS_CACHE,
  TTS_EXPECTED_BYTES,
  TTS_MODEL_ID,
  llmPreset,
  sdBytes,
} from "../../shared/models.js";
import { formatBytes, formatEta, formatPercent } from "../../shared/format.js";
import { checkWebGPU, isFileProtocol } from "../../shared/webgpu.js";
import { parseToolCalls, visibleReply } from "../../shared/tools.js";
import { pcmToWavBlob } from "../../shared/wav.js";

const $ = (id) => document.getElementById(id);

const state = {
  presetKey: DEFAULT_LLM_PRESET,
  llm: null,
  sd: null,
  tts: null,
  llmReady: false,
  sdReady: false,
  ttsReady: false,
  llmBusy: false,
  history: [],
  pending: new Map(),
};

function bar(fillId, barId, bytesId, etaId, progress, fallbackTotal) {
  const percent = progress?.percent ?? 0;
  $(fillId).style.width = `${Math.max(0, Math.min(100, percent))}%`;
  $(barId).setAttribute("aria-valuenow", String(Math.round(percent)));
  const total = progress?.total || fallbackTotal;
  $(bytesId).textContent = `${formatBytes(progress?.loaded || 0)} / ${formatBytes(total)} · ${formatPercent(percent)}`;
  if (etaId) {
    $(etaId).textContent = progress?.fromCache ? "en caché" : progress ? `ETA ${formatEta(progress.etaMs)}` : "—";
  }
}

function addBubble(kind, text) {
  const node = document.createElement("div");
  node.className = `bubble ${kind}`;
  node.textContent = text;
  $("log").append(node);
  node.scrollIntoView({ block: "nearest" });
  return node;
}

function setChatEnabled(enabled) {
  $("input").disabled = !enabled;
  $("send").disabled = !enabled;
  if (enabled) $("input").placeholder = "Escribe en español o en inglés. Pide un dibujo o que lo diga en alto.";
}

function renderPresets() {
  const field = $("presets");
  field.innerHTML = "";
  for (const preset of LLM_PRESETS) {
    const label = document.createElement("label");
    label.className = `preset${preset.heavy ? " heavy" : ""}`;
    label.innerHTML = `
      <input type="radio" name="preset" value="${preset.key}">
      <span><strong>${preset.label}</strong> · ${formatBytes(preset.bytes)}<br>${preset.blurb}</span>
    `;
    const input = label.querySelector("input");
    input.checked = preset.key === state.presetKey;
    input.addEventListener("change", () => {
      state.presetKey = preset.key;
      $("llm-card").classList.toggle("heavy", Boolean(llmPreset(preset.key).heavy));
    });
    field.append(label);
  }
  $("llm-card").classList.toggle("heavy", Boolean(llmPreset(state.presetKey).heavy));
}

function lockPresets(locked) {
  for (const input of $("presets").querySelectorAll("input")) input.disabled = locked;
  $("llm-go").disabled = locked;
  $("llm-cancel").hidden = !locked || state.llmReady;
}

function waitFor(workerKind, jobId) {
  return new Promise((resolve, reject) => {
    state.pending.set(`${workerKind}:${jobId}`, { resolve, reject });
  });
}

function settle(workerKind, jobId, error, value) {
  const key = `${workerKind}:${jobId}`;
  const pending = state.pending.get(key);
  if (!pending) return false;
  state.pending.delete(key);
  if (error) pending.reject(error);
  else pending.resolve(value);
  return true;
}

function startLlm() {
  if (state.llm) state.llm.terminate();
  state.llmReady = false;
  state.llmBusy = true;
  setChatEnabled(false);
  const preset = llmPreset(state.presetKey);
  $("llm-title").textContent = preset.label;
  $("llm-status").textContent = "Resolviendo ficheros en Hugging Face…";
  $("llm-percent").textContent = "0 %";
  lockPresets(true);
  const worker = new Worker(new URL("./llm-worker.js", import.meta.url), { type: "module" });
  state.llm = worker;
  worker.onmessage = (event) => onLlm(event.data);
  worker.postMessage({ type: "load", preset });
}

function onLlm(message) {
  if (message.type === "progress") {
    const percent = message.percent || 0;
    $("llm-percent").textContent = formatPercent(percent);
    bar("llm-fill", "llm-bar", "llm-bytes", "llm-eta", message, llmPreset(state.presetKey).bytes);
    $("llm-file").textContent = message.file || "";
    if (message.status && message.status !== "progress") {
      $("llm-status").textContent = `Estado: ${message.status}${message.file ? ` · ${message.file}` : ""}`;
    }
    return;
  }
  if (message.type === "status") {
    $("llm-status").textContent = message.message;
    return;
  }
  if (message.type === "ready") {
    state.llmReady = true;
    state.llmBusy = false;
    $("llm-percent").textContent = "100 %";
    $("llm-status").textContent = "Modelo en la GPU. El chat ya acepta mensajes.";
    $("llm-cancel").hidden = true;
    setChatEnabled(true);
    $("input").focus();
    return;
  }
  if (message.type === "token") {
    const node = document.querySelector(`[data-job="${message.jobId}"]`);
    if (node) node.textContent += message.text;
    return;
  }
  if (message.type === "done") {
    settle("llm", message.jobId, null, message.text || "");
    return;
  }
  if (message.type === "error") {
    if (!settle("llm", message.jobId, new Error(message.message))) {
      state.llmBusy = false;
      $("llm-status").textContent = message.message;
      lockPresets(false);
      $("llm-cancel").hidden = true;
    }
  }
}

function startSd() {
  const worker = new Worker(new URL("../../shared/sd-worker.js", import.meta.url), { type: "module" });
  state.sd = worker;
  worker.onmessage = (event) => onSd(event.data);
  $("sd-status").textContent = "Descargando Small Stable Diffusion v0…";
  worker.postMessage({ type: "download", modelKey: "small" });
}

function onSd(message) {
  if (message.type === "progress") {
    bar("sd-fill", "sd-bar", "sd-bytes", "sd-eta", message, sdBytes(SD_MODELS.small));
    $("sd-status").textContent = message.fromCache ? "En caché." : "Descargando…";
    return;
  }
  if (message.type === "download-done") {
    $("sd-status").textContent = "Descarga lista. Creando la sesión WebGPU…";
    state.sd.postMessage({ type: "load", modelKey: "small" });
    return;
  }
  if (message.type === "loaded") {
    state.sdReady = true;
    $("sd-status").textContent = "Listo para generate_image.";
    return;
  }
  if (message.type === "status" || message.type === "step") {
    $("sd-status").textContent = message.message || `Paso ${message.step} de ${message.total}`;
    return;
  }
  if (message.type === "image") {
    settle("sd", message.jobId, null, message);
    return;
  }
  if (message.type === "error") {
    if (!settle("sd", message.jobId, new Error(message.message))) {
      $("sd-status").textContent = message.message;
    }
  }
}

function startTts() {
  const worker = new Worker(new URL("./tts-worker.js", import.meta.url), { type: "module" });
  state.tts = worker;
  worker.onmessage = (event) => onTts(event.data);
  $("tts-status").textContent = `Descargando ${TTS_MODEL_ID}…`;
  worker.postMessage({ type: "load" });
}

function onTts(message) {
  if (message.type === "progress") {
    bar("tts-fill", "tts-bar", "tts-bytes", "tts-eta", message, TTS_EXPECTED_BYTES);
    return;
  }
  if (message.type === "status") {
    $("tts-status").textContent = message.message;
    return;
  }
  if (message.type === "ready") {
    state.ttsReady = true;
    $("tts-status").textContent = message.device === "wasm"
      ? "Listo en WASM (WebGPU no aceptó este grafo)."
      : "Listo en WebGPU.";
    $("tts-demo").disabled = false;
    bar("tts-fill", "tts-bar", "tts-bytes", "tts-eta", {
      loaded: TTS_EXPECTED_BYTES,
      total: TTS_EXPECTED_BYTES,
      percent: 100,
      etaMs: 0,
    }, TTS_EXPECTED_BYTES);
    return;
  }
  if (message.type === "audio") {
    settle("tts", message.jobId, null, message);
    return;
  }
  if (message.type === "error") {
    if (!settle("tts", message.jobId, new Error(message.message))) {
      $("tts-status").textContent = message.message;
    }
  }
}

function playSamples(samples, sampleRate) {
  const blob = pcmToWavBlob(samples, sampleRate);
  const url = URL.createObjectURL(blob);
  const audio = document.createElement("audio");
  audio.controls = true;
  audio.src = url;
  audio.play().catch(() => {
    /* el navegador puede exigir otro gesto; los controles siguen ahí */
  });
  return audio;
}

async function runImage(call) {
  const prompt = String(call.arguments.prompt || "").trim();
  const node = addBubble("tool", prompt ? `generate_image · ${prompt}` : "generate_image sin prompt.");
  if (!prompt) return "generate_image no trajo prompt.";
  if (!state.sdReady) {
    node.textContent += " El modelo de imagen todavía no está listo.";
    return "generate_image could not run because the image model is still loading.";
  }
  const jobId = crypto.randomUUID();
  const done = waitFor("sd", jobId);
  state.sd.postMessage({
    type: "generate",
    modelKey: "small",
    prompt,
    negative: "blurry, low quality, watermark",
    steps: SD_MODELS.small.defaultSteps,
    guidance: SD_MODELS.small.guidance,
    seed: (Math.random() * 0xffffffff) >>> 0,
    jobId,
  });
  const image = await done;
  const blob = new Blob([image.buffer], { type: "image/png" });
  const img = document.createElement("img");
  img.alt = prompt;
  img.className = "out";
  img.src = URL.createObjectURL(blob);
  node.append(img);
  return `generate_image finished. The picture for "${prompt}" is on screen.`;
}

async function runSpeak(call) {
  const text = String(call.arguments.text || "").trim();
  const node = addBubble("tool", text ? `speak · ${text}` : "speak sin texto.");
  if (!text) return "speak no trajo texto.";
  if (!state.ttsReady) {
    node.textContent += " La voz todavía no está lista.";
    return "speak could not run because TTS is still loading.";
  }
  const jobId = crypto.randomUUID();
  const done = waitFor("tts", jobId);
  state.tts.postMessage({ type: "speak", text, jobId });
  const audio = await done;
  node.append(playSamples(audio.samples, audio.sampleRate));
  return `speak finished. The user heard: "${text}".`;
}

async function askLlm(allowTools) {
  const jobId = crypto.randomUUID();
  const bubble = addBubble("assistant", "");
  bubble.dataset.job = jobId;
  const done = waitFor("llm", jobId);
  state.llm.postMessage({
    type: "generate",
    jobId,
    allowTools,
    messages: state.history.slice(-8),
  });
  const text = await done;
  const calls = allowTools ? parseToolCalls(text) : [];
  const visible = visibleReply(text);
  bubble.textContent = visible || (calls.length ? "(llamada a herramienta)" : text);
  bubble.removeAttribute("data-job");
  state.history.push({ role: "assistant", content: text });
  return { text, calls, visible };
}

async function onSubmit(event) {
  event.preventDefault();
  const content = $("input").value.trim();
  if (!content || !state.llmReady || state.llmBusy) return;
  $("input").value = "";
  state.llmBusy = true;
  setChatEnabled(false);
  addBubble("user", content);
  state.history.push({ role: "user", content });
  try {
    const first = await askLlm(true);
    const notes = [];
    for (const call of first.calls) {
      if (call.name === "generate_image") notes.push(await runImage(call));
      else if (call.name === "speak") notes.push(await runSpeak(call));
    }
    if (notes.length) {
      state.history.push({
        role: "user",
        content: `Tool results:\n${notes.join("\n")}\nReply briefly. Do not call tools.`,
      });
      await askLlm(false);
    }
  } catch (error) {
    addBubble("system", error instanceof Error ? error.message : String(error));
  } finally {
    state.llmBusy = false;
    if (state.llmReady) setChatEnabled(true);
  }
}

async function main() {
  renderPresets();
  if (isFileProtocol()) return;
  const gpu = await checkWebGPU();
  if (!gpu.ok) {
    const banner = $("gpu-banner");
    banner.hidden = false;
    banner.innerHTML = `<strong>WebGPU no está listo.</strong> ${gpu.detail}`;
    $("sd-status").textContent = gpu.detail;
    $("tts-status").textContent = gpu.detail;
    $("llm-status").textContent = "Sin WebGPU no se descarga el modelo de lenguaje.";
    $("llm-go").disabled = true;
    return;
  }
  $("sd-status").textContent = gpu.detail;
  startSd();
  startTts();

  $("llm-go").addEventListener("click", startLlm);
  $("llm-cancel").addEventListener("click", () => {
    state.llm?.terminate();
    state.llm = null;
    state.llmReady = false;
    state.llmBusy = false;
    $("llm-status").textContent = "Descarga cancelada. Puedes elegir otro tamaño.";
    $("llm-percent").textContent = "0 %";
    lockPresets(false);
  });
  $("clear-llm").addEventListener("click", async () => {
    if (!confirm("¿Borrar la caché de transformers.js? También quita el tokenizador CLIP y la voz si comparten esa caché.")) return;
    state.llm?.terminate();
    state.llm = null;
    state.llmReady = false;
    await caches.delete(TRANSFORMERS_CACHE);
    $("llm-status").textContent = "Caché de transformers.js borrada.";
    $("llm-percent").textContent = "0 %";
    lockPresets(false);
    setChatEnabled(false);
  });
  $("tts-demo").addEventListener("click", async () => {
    const node = addBubble("tool", "Prueba de voz.");
    try {
      const jobId = crypto.randomUUID();
      const done = waitFor("tts", jobId);
      state.tts.postMessage({ type: "speak", text: "Hola. Esta voz se genera en el navegador.", jobId });
      const audio = await done;
      node.append(playSamples(audio.samples, audio.sampleRate));
    } catch (error) {
      node.textContent = error instanceof Error ? error.message : String(error);
    }
  });
  $("composer").addEventListener("submit", onSubmit);
  $("input").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      $("composer").requestSubmit();
    }
  });
}

main();
