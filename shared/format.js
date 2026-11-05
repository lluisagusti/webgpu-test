/**
 * shared/format.js
 * Formatea bytes y tiempos para las barras de progreso. Sin dependencias.
 */

/** Bytes en unidades decimales (1 GB = 10^9), como los lista Hugging Face. */
export function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes < 0) return "—";
  if (bytes < 1000) return `${Math.round(bytes)} B`;
  const units = ["KB", "MB", "GB", "TB"];
  let value = bytes / 1000;
  let unit = 0;
  while (value >= 1000 && unit < units.length - 1) {
    value /= 1000;
    unit += 1;
  }
  const digits = value >= 100 ? 0 : value >= 10 ? 1 : 2;
  return `${value.toFixed(digits)} ${units[unit]}`;
}

/** ETA a partir de milisegundos. null si todavía no hay tasa. */
export function formatEta(ms) {
  if (ms == null || !Number.isFinite(ms)) return "calculando…";
  if (ms < 1500) return "unos segundos";
  const totalSeconds = Math.round(ms / 1000);
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;
  if (hours > 0) return `${hours} h ${minutes} min`;
  if (minutes > 0) return `${minutes} min ${seconds} s`;
  return `${seconds} s`;
}

export function formatPercent(percent) {
  if (!Number.isFinite(percent)) return "0 %";
  return `${Math.max(0, Math.min(100, percent)).toFixed(1)} %`;
}
