/**
 * shared/tools.js
 * Extrae llamadas a herramientas del texto del LLM.
 *
 * Formato que pedimos en el prompt del sistema:
 *
 * ```tool
 * {"tool":"generate_image","arguments":{"prompt":"..."}}
 * ```
 *
 * También acepta ```json, etiquetas <tool> y un objeto suelto con
 * "tool" o "name". No lanza si el JSON está mal: devuelve lo que pudo leer.
 */

const TOOL_NAMES = new Set(["generate_image", "speak"]);

function normalizeCall(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  const name = value.tool || value.name || value.tool_name;
  if (!TOOL_NAMES.has(name)) return null;
  const rawArgs = value.arguments || value.args || value.parameters || {};
  const args = rawArgs && typeof rawArgs === "object" && !Array.isArray(rawArgs) ? rawArgs : {};
  return { name, arguments: args };
}

function tryParse(text) {
  const trimmed = text.trim();
  if (!trimmed) return [];
  try {
    const parsed = JSON.parse(trimmed);
    if (Array.isArray(parsed)) return parsed.map(normalizeCall).filter(Boolean);
    const one = normalizeCall(parsed);
    return one ? [one] : [];
  } catch {
    return [];
  }
}

/** Quita un bloque <think>…</think> para no confundirlo con una herramienta. */
export function stripThinking(text) {
  return text.replace(/<think>[\s\S]*?<\/think>/gi, " ").replace(/<\|think\|>[\s\S]*?<\|\/think\|>/g, " ");
}

/**
 * @param {string} text
 * @returns {{ name: "generate_image" | "speak", arguments: Record<string, unknown> }[]}
 */
export function parseToolCalls(text) {
  if (!text) return [];
  const source = stripThinking(text);
  const found = [];
  const fence = /```(?:tool|json)?\s*([\s\S]*?)```/gi;
  let match = fence.exec(source);
  while (match) {
    found.push(...tryParse(match[1]));
    match = fence.exec(source);
  }

  const tagged = /<tool>\s*([\s\S]*?)<\/tool>/gi;
  match = tagged.exec(source);
  while (match) {
    found.push(...tryParse(match[1]));
    match = tagged.exec(source);
  }

  if (found.length > 0) return found;

  const start = source.indexOf("{");
  const end = source.lastIndexOf("}");
  if (start >= 0 && end > start) return tryParse(source.slice(start, end + 1));
  return [];
}

/** Texto visible: fuera de los bloques de herramienta. */
export function visibleReply(text) {
  return stripThinking(text)
    .replace(/```(?:tool|json)?\s*[\s\S]*?```/gi, " ")
    .replace(/<tool>[\s\S]*?<\/tool>/gi, " ")
    .replace(/[ \t]+\n/g, "\n")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}
