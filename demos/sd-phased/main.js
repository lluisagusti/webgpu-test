/**
 * demos/sd-phased/main.js
 * UI de las dos fases. La inferencia vive en shared/sd-worker.js.
 */

import { SD_MODELS, sdBytes } from "../../shared/models.js";
import { formatBytes, formatEta, formatPercent } from "../../shared/format.js";
import { checkWebGPU, isFileProtocol } from "../../shared/webgpu.js";

const $ = (id) => document.getElementById(id);

const state = {
  downloaded: { small: false, large: false },
  loaded: null,
  busy: false,
  worker: null,
};

function paintBar(key, progress) {
  const suffix = key === "small" ? "a" : "b";
  const percent = progress?.percent ?? 0;
  $(`fill-${suffix}`).style.width = `${Math.max(0, Math.min(100, percent))}%`;
  $(`bar-${suffix}`).setAttribute("aria-valuenow", String(Math.round(percent)));
  const total = progress?.total || sdBytes(SD_MODELS[key]);
  $(`bytes-${suffix}`).textContent = `${formatBytes(progress?.loaded || 0)} / ${formatBytes(total)} · ${formatPercent(percent)}`;
  $(`eta-${suffix}`).textContent = progress?.fromCache
    ? "en caché"
    : progress ? `ETA ${formatEta(progress.etaMs)}` : "—";
  $(`file-${suffix}`).textContent = progress?.file
    ? `${progress.file} · ${formatBytes(progress.fileLoaded)} / ${formatBytes(progress.fileTotal)}`
    : "";
}

function setActiveButtons() {
  $("use-small").disabled = !state.downloaded.small || state.busy;
  $("use-large").disabled = !state.downloaded.large || state.busy;
  $("use-small").setAttribute("aria-pressed", String(state.loaded === "small"));
  $("use-large").setAttribute("aria-pressed", String(state.loaded === "large"));
  $("go").disabled = !state.loaded || state.busy;
  $("active").textContent = state.loaded ? SD_MODELS[state.loaded].label : "ninguno";
  const turbo = state.loaded === "large";
  $("steps").disabled = turbo || state.busy;
  $("guidance").disabled = turbo || state.busy;
  $("negative").disabled = turbo || state.busy;
  if (turbo) $("steps").value = "1";
}

function showGpu(check) {
  const banner = $("gpu-banner");
  if (check.ok) {
    banner.hidden = true;
    return;
  }
  banner.hidden = false;
  banner.innerHTML = `<strong>WebGPU no está listo.</strong> ${check.detail} Sin adaptador con shader-f16 no se descargan los modelos.`;
}

function ensureWorker() {
  if (state.worker) return state.worker;
  const worker = new Worker(new URL("../../shared/sd-worker.js", import.meta.url), { type: "module" });
  worker.onmessage = (event) => onWorker(event.data);
  state.worker = worker;
  return worker;
}

function onWorker(message) {
  if (message.type === "progress") {
    paintBar(message.modelKey, message);
    const status = message.modelKey === "small" ? $("status-a") : $("status-b");
    status.textContent = message.fromCache ? "Leyendo la caché…" : "Descargando…";
    return;
  }
  if (message.type === "download-done") {
    state.downloaded[message.modelKey] = true;
    const status = message.modelKey === "small" ? $("status-a") : $("status-b");
    status.textContent = "Descarga completa.";
    paintBar(message.modelKey, {
      loaded: sdBytes(SD_MODELS[message.modelKey]),
      total: sdBytes(SD_MODELS[message.modelKey]),
      percent: 100,
      etaMs: 0,
      fromCache: false,
    });
    if (message.modelKey === "small") {
      $("status-a").textContent = "Creando la sesión en la GPU…";
      state.worker.postMessage({ type: "load", modelKey: "small" });
    }
    setActiveButtons();
    return;
  }
  if (message.type === "status") {
    if (message.modelKey === "small" && state.loaded !== "large") $("status-a").textContent = message.message;
    if (message.modelKey === "large") $("status-b").textContent = message.message;
    $("gen-status").textContent = message.message;
    return;
  }
  if (message.type === "loaded") {
    state.loaded = message.modelKey;
    state.busy = false;
    if (message.modelKey === "small") {
      $("status-a").textContent = "Listo. Ya puedes generar. Empieza la fase B.";
      if (!state.downloaded.large) {
        $("status-b").textContent = "Descargando SD-Turbo…";
        state.worker.postMessage({ type: "download", modelKey: "large" });
      }
    } else {
      $("status-b").textContent = "SD-Turbo está en la GPU.";
    }
    setActiveButtons();
    return;
  }
  if (message.type === "step") {
    $("gen-status").textContent = `Paso ${message.step} de ${message.total}…`;
    return;
  }
  if (message.type === "image") {
    const blob = new Blob([message.buffer], { type: "image/png" });
    const url = URL.createObjectURL(blob);
    const image = new Image();
    image.onload = () => {
      const canvas = $("canvas");
      canvas.width = image.width;
      canvas.height = image.height;
      canvas.getContext("2d").drawImage(image, 0, 0);
      URL.revokeObjectURL(url);
    };
    image.src = url;
    $("result-meta").textContent = `${SD_MODELS[message.modelKey].label} · semilla ${message.seed} · ${message.width}×${message.height}`;
    $("gen-status").textContent = "Imagen lista.";
    state.busy = false;
    setActiveButtons();
    return;
  }
  if (message.type === "cache-cleared") {
    state.downloaded = { small: false, large: false };
    state.loaded = null;
    state.busy = false;
    paintBar("small", null);
    paintBar("large", null);
    $("status-a").textContent = "Caché vacía. Volviendo a bajar el modelo pequeño…";
    $("status-b").textContent = "Aún no empieza.";
    $("gen-status").textContent = "";
    setActiveButtons();
    state.worker.postMessage({ type: "download", modelKey: "small" });
    return;
  }
  if (message.type === "error") {
    state.busy = false;
    $("gen-status").textContent = message.message;
    $("gen-status").className = "status err";
    const status = message.modelKey === "large" ? $("status-b") : $("status-a");
    status.textContent = "Error. El detalle está bajo el botón Generar.";
    setActiveButtons();
  }
}

function switchTo(modelKey) {
  if (!state.downloaded[modelKey] || state.loaded === modelKey) {
    setActiveButtons();
    return;
  }
  state.busy = true;
  setActiveButtons();
  $("gen-status").className = "status";
  $("gen-status").textContent = `Cargando ${SD_MODELS[modelKey].label} desde la caché…`;
  state.worker.postMessage({ type: "load", modelKey });
}

async function main() {
  if (isFileProtocol()) return;
  const gpu = await checkWebGPU();
  showGpu(gpu);
  if (!gpu.ok) {
    $("status-a").textContent = gpu.detail;
    return;
  }
  $("status-a").textContent = `${gpu.detail} Bajando el modelo pequeño…`;
  ensureWorker();
  state.worker.postMessage({ type: "download", modelKey: "small" });

  $("use-small").addEventListener("click", () => switchTo("small"));
  $("use-large").addEventListener("click", () => switchTo("large"));
  $("clear").addEventListener("click", () => {
    if (!confirm("¿Borrar los modelos de Stable Diffusion de la Cache API?")) return;
    state.busy = true;
    state.worker.postMessage({ type: "clear-cache" });
  });
  $("form").addEventListener("submit", (event) => {
    event.preventDefault();
    if (!state.loaded || state.busy) return;
    state.busy = true;
    $("gen-status").className = "status";
    $("gen-status").textContent = "Generando…";
    setActiveButtons();
    const seedRaw = $("seed").value.trim();
    state.worker.postMessage({
      type: "generate",
      modelKey: state.loaded,
      prompt: $("prompt").value.trim(),
      negative: $("negative").value.trim(),
      steps: Number($("steps").value) || SD_MODELS[state.loaded].defaultSteps,
      guidance: Number($("guidance").value),
      seed: seedRaw === "" ? null : Number(seedRaw),
      jobId: crypto.randomUUID(),
    });
  });
}

main();
