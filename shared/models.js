/**
 * shared/models.js
 * Identificadores, URLs y tamaños reales (comprobados en la API de Hugging Face).
 * Para probar otro LLM, cambia LLM_PRESETS o DEFAULT_LLM_PRESET.
 * Para otra voz, cambia TTS_MODEL_ID (Xenova/mms-tts-eng es el mismo formato, en inglés).
 */

export const CACHE_NAME = "ejemplo-webgpu-models-v1";
export const TRANSFORMERS_CACHE = "transformers-cache";

export const ORT_VERSION = "1.31.1";
export const TRANSFORMERS_VERSION = "4.3.2";

export const ORT_URL = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/ort.webgpu.min.mjs`;
export const TRANSFORMERS_URL = `https://cdn.jsdelivr.net/npm/@huggingface/transformers@${TRANSFORMERS_VERSION}/+esm`;

export const CLIP_TOKENIZER_ID = "Xenova/clip-vit-base-patch16";

export function hfResolve(repo, path) {
  return `https://huggingface.co/${repo}/resolve/main/${path}`;
}

/**
 * small: OFA-Sys/small-stable-diffusion-v0 exportado a ONNX para ORT.
 *   UNet en dos piezas (model.onnx + weights.pb). Entradas float16.
 * large: SD-Turbo fp16, el mismo repositorio que el ejemplo oficial de ORT WebGPU.
 *   Un paso, sin classifier-free guidance. Entradas float32 / timestep int64.
 */
export const SD_MODELS = {
  small: {
    key: "small",
    repo: "subpixel/small-stable-diffusion-v0-onnx-ort-web",
    baseModel: "OFA-Sys/small-stable-diffusion-v0",
    label: "Small Stable Diffusion v0",
    mode: "euler",
    defaultSteps: 12,
    guidance: 7.5,
    files: [
      { path: "text_encoder/model.onnx", bytes: 246397990 },
      { path: "unet/model.onnx", bytes: 757360 },
      { path: "unet/weights.pb", bytes: 1158728960 },
      { path: "vae_decoder/model.onnx", bytes: 99093889 },
    ],
    externalWeights: "weights.pb",
    session: {
      textEncoder: { freeDimensionOverrides: { batch: 1, sequence: 77 } },
      unet: {
        freeDimensionOverrides: { batch: 1, channels: 4, height: 64, width: 64, sequence: 77 },
      },
      vae: { freeDimensionOverrides: { batch: 1, channels: 4, height: 64, width: 64 } },
    },
  },
  large: {
    key: "large",
    repo: "schmuell/sd-turbo-ort-web",
    baseModel: "stabilityai/sd-turbo",
    label: "SD-Turbo",
    mode: "turbo",
    defaultSteps: 1,
    guidance: 0,
    files: [
      { path: "text_encoder/model.onnx", bytes: 681393168 },
      { path: "unet/model.onnx", bytes: 1733430199 },
      { path: "vae_decoder/model.onnx", bytes: 99094314 },
    ],
    externalWeights: null,
    session: {
      textEncoder: { freeDimensionOverrides: { batch_size: 1 } },
      unet: {
        freeDimensionOverrides: {
          batch_size: 1,
          num_channels: 4,
          height: 64,
          width: 64,
          sequence_length: 77,
        },
      },
      vae: {
        freeDimensionOverrides: {
          batch_size: 1,
          num_channels_latent: 4,
          height_latent: 64,
          width_latent: 64,
        },
      },
    },
  },
};

export function sdBytes(model) {
  return model.files.reduce((sum, file) => sum + file.bytes, 0);
}

/** Voz por defecto en español. El cuantizado q8 pesa unos 36,6 MB. */
export const TTS_MODEL_ID = "Xenova/mms-tts-spa";
export const TTS_DTYPE = "q8";
export const TTS_EXPECTED_BYTES = 38362987;

/**
 * El objetivo de ~8 GB es el preset olmo-q8 (q8, ficheros de menos de 2 GB).
 * No es el predeterminado: en una GPU de 8 GB se queda sin VRAM.
 * Los bytes incluyen el grafo ONNX, los shards y tokenizer.json.
 */
export const LLM_PRESETS = [
  {
    key: "olmo-q4f16",
    modelId: "onnx-community/Olmo-3-7B-Instruct-ONNX",
    dtype: "q4f16",
    bytes: 414480 + 2077432032 + 1732204768 + 4119438,
    label: "Olmo 3 7B · q4f16",
    blurb: "Recomendado. Unos 3,8 GB. Es lo más grande que suele caber junto al escritorio en una GPU de 8 GB.",
    recommended: true,
  },
  {
    key: "olmo-q8",
    modelId: "onnx-community/Olmo-3-7B-Instruct-ONNX",
    dtype: "q8",
    bytes: 370552 + 2096083392 + 2057715712 + 2055954432 + 1408455104 + 4119438,
    label: "Olmo 3 7B · q8 · ~7,6 GB",
    blurb: "El objetivo cercano a 8 GB, partido en shards de menos de 2 GB. Reserva una GPU con 12–16 GB libres.",
    heavy: true,
  },
  {
    key: "qwen3-4b-q4f16",
    modelId: "onnx-community/Qwen3-4B-ONNX",
    dtype: "q4f16",
    bytes: 2832918673,
    label: "Qwen3 4B · q4f16",
    blurb: "Unos 2,8 GB. Más pequeño, y suele respetar mejor el JSON de las herramientas.",
  },
];

export const DEFAULT_LLM_PRESET = "olmo-q4f16";

export function llmPreset(key) {
  return LLM_PRESETS.find((preset) => preset.key === key) ?? LLM_PRESETS[0];
}

export const SYSTEM_PROMPT = `You are a small on-device agent running entirely in the user's browser via WebGPU. Reply in the same language the user writes (Spanish or English).

You can call two tools. When you need one, add a single fenced block in exactly this shape:

\`\`\`tool
{"tool":"generate_image","arguments":{"prompt":"English visual description"}}
\`\`\`

or

\`\`\`tool
{"tool":"speak","arguments":{"text":"short line to read aloud, same language as the user"}}
\`\`\`

Rules:
- Call generate_image when the user wants a picture, drawing, or illustration. The prompt inside the tool must be English.
- Call speak when the user wants something read aloud. Keep that text under 280 characters.
- You may write one short sentence before the fence. Emit at most one tool block.
- If no tool is needed, answer normally and do not emit a fence or JSON.
- Do not invent tool results. The application runs the tool after your message.`;
