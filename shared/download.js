/**
 * shared/download.js
 * Descarga con progreso (bytes, porcentaje, ETA) y caché en la Cache API.
 *
 * Una respuesta a medias no se guarda: la Cache API solo admite respuestas
 * completas. Si se corta el TCP, ese fichero vuelve a empezar. Los que ya
 * terminaron se reutilizan. Las peticiones Range y por qué no reanudamos a
 * medias están explicadas en el README.
 */

import { CACHE_NAME } from "./models.js";

/**
 * @param {string} url
 * @param {{
 *   cacheName?: string,
 *   expectedBytes?: number,
 *   signal?: AbortSignal,
 *   onProgress?: (event: {
 *     loaded: number,
 *     total: number,
 *     percent: number,
 *     etaMs: number | null,
 *     fromCache: boolean
 *   }) => void
 * }} [options]
 * @returns {Promise<ArrayBuffer | null>} null: solo se dejó en caché
 */
export async function fetchCached(url, options = {}) {
  const cacheName = options.cacheName ?? CACHE_NAME;
  const cache = await caches.open(cacheName);
  const hit = await cache.match(url);
  if (hit) {
    const lengthHeader = Number(hit.headers.get("Content-Length"));
    const total = Number.isFinite(lengthHeader) && lengthHeader > 0
      ? lengthHeader
      : options.expectedBytes || 0;
    options.onProgress?.({
      loaded: total,
      total,
      percent: 100,
      etaMs: 0,
      fromCache: true,
    });
    if (options.returnBuffer) return hit.arrayBuffer();
    return null;
  }

  const response = await fetch(url, { signal: options.signal });
  if (!response.ok || !response.body) {
    throw new Error(`HTTP ${response.status} al pedir ${url}`);
  }

  const headerTotal = Number(response.headers.get("Content-Length"));
  const total = Number.isFinite(headerTotal) && headerTotal > 0
    ? headerTotal
    : options.expectedBytes || 0;

  const reader = response.body.getReader();
  const chunks = [];
  let loaded = 0;
  const started = performance.now();

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.byteLength;
    const seconds = Math.max((performance.now() - started) / 1000, 0.05);
    const remain = total > loaded ? (total - loaded) / (loaded / seconds) : 0;
    options.onProgress?.({
      loaded,
      total: total || loaded,
      percent: total ? (loaded / total) * 100 : 0,
      etaMs: total > loaded ? remain * 1000 : 0,
      fromCache: false,
    });
  }

  const blob = new Blob(chunks);
  const headers = new Headers({
    "Content-Type": response.headers.get("Content-Type") || "application/octet-stream",
    "Content-Length": String(blob.size),
  });
  await cache.put(url, new Response(blob, { status: 200, headers }));
  if (options.returnBuffer) return blob.arrayBuffer();
  return null;
}

/**
 * Agrega varias descargas en una sola barra.
 * La ETA ignora los ficheros que salieron de caché, para no marcar "0 s"
 * y luego dispararse cuando empieza el fichero grande.
 */
export function trackSet(files) {
  const total = files.reduce((sum, file) => sum + file.bytes, 0);
  const perFile = new Map(files.map((file) => [file.path, { loaded: 0, total: file.bytes }]));
  const started = performance.now();
  let networkLoaded = 0;

  return {
    total,
    note(path, event) {
      const previous = perFile.get(path) ?? { loaded: 0, total: 0 };
      const delta = Math.max(0, event.loaded - previous.loaded);
      if (!event.fromCache) networkLoaded += delta;
      perFile.set(path, {
        loaded: event.fromCache ? previous.total || event.loaded : event.loaded,
        total: event.total || previous.total,
      });

      let loaded = 0;
      for (const item of perFile.values()) loaded += item.loaded;
      const seconds = Math.max((performance.now() - started) / 1000, 0.05);
      const remain = Math.max(0, total - loaded);
      const etaMs = networkLoaded > 1024 * 64 ? (remain / (networkLoaded / seconds)) * 1000 : null;
      return {
        loaded,
        total,
        percent: total ? (loaded / total) * 100 : 0,
        etaMs: loaded >= total ? 0 : etaMs,
        fromCache: networkLoaded === 0 && loaded >= total,
        file: path,
        fileLoaded: perFile.get(path)?.loaded ?? 0,
        fileTotal: perFile.get(path)?.total ?? 0,
      };
    },
  };
}

export async function clearModelCache(cacheName = CACHE_NAME) {
  await caches.delete(cacheName);
}
