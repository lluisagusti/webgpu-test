/**
 * shared/sd-engine.js
 * Texto → imagen con ONNX Runtime Web (WebGPU).
 *
 * El tokenizador CLIP sale de transformers.js. El UNet, el text encoder y el
 * VAE decoder se ejecutan a mano: transformers.js no ofrece pipeline text-to-image.
 *
 * SD-Turbo (modelo grande): un paso de Euler con sigma 14.6146 y timestep 999,
 * la misma receta que el ejemplo de Microsoft onnxruntime-inference-examples.
 * Small SD v0: varios pasos de Euler con guidance. Su ficha pide DPM-Solver++;
 * Euler usa la misma agenda de betas y no añade otra librería.
 */

import {
  CLIP_TOKENIZER_ID,
  ORT_VERSION,
  SD_MODELS,
  TRANSFORMERS_URL,
  hfResolve,
} from "./models.js";
import { fetchCached } from "./download.js";
import { packFloat16, unpackToFloat32 } from "./fp16.js";
import { applyCfg, eulerSchedule, eulerStep, scaleLatent } from "./scheduler.js";
import { pinOrtThreads, pinTransformersThreads } from "./wasm-threads.js";

const TURBO_SIGMA = 14.6146;
const VAE_SCALE = 0.18215;
const LATENT_SHAPE = [1, 4, 64, 64];

function configureOrt(ort) {
  ort.env.wasm.wasmPaths = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
  pinOrtThreads(ort);
  ort.env.logLevel = "warning";
}

function sessionOptions(extra) {
  return {
    executionProviders: ["webgpu"],
    enableMemPattern: false,
    enableCpuMemArena: false,
    extra: {
      session: {
        disable_prepacking: "1",
        use_device_allocator_for_initializers: "1",
        use_ort_model_bytes_directly: "1",
        use_ort_model_bytes_for_initializers: "1",
      },
    },
    ...extra,
  };
}

function mulberry32(seed) {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) | 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function randn(rng) {
  const u = Math.max(rng(), 1e-12);
  const v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function noiseLatent(sigma, seed) {
  const rng = mulberry32(seed >>> 0);
  const data = new Float32Array(LATENT_SHAPE[0] * LATENT_SHAPE[1] * LATENT_SHAPE[2] * LATENT_SHAPE[3]);
  for (let i = 0; i < data.length; i += 1) data[i] = randn(rng) * sigma;
  return data;
}

function pickOutput(outputs, name) {
  if (outputs[name]) return outputs[name];
  const key = Object.keys(outputs)[0];
  return outputs[key];
}

async function tensorFloat32(tensor) {
  let data = tensor.data;
  if (tensor.location === "gpu-buffer" && typeof tensor.getData === "function") {
    data = await tensor.getData();
  }
  return unpackToFloat32(data);
}

function flattenIds(value) {
  let data = value;
  if (data && typeof data === "object" && "data" in data && data.data) data = data.data;
  if (Array.isArray(data) && Array.isArray(data[0])) data = data[0];
  return Array.from(data, (item) => Number(item));
}

/**
 * @param {typeof import("onnxruntime-web")} ort
 */
export function createSdRuntime(ort) {
  configureOrt(ort);

  /** @type {import("@huggingface/transformers").PreTrainedTokenizer | null} */
  let tokenizer = null;
  const sessions = { key: null, text: null, unet: null, vae: null, buffers: null };

  async function ensureTokenizer(onStatus) {
    if (tokenizer) return tokenizer;
    onStatus?.("Cargando el tokenizador CLIP…");
    const { AutoTokenizer, env } = await import(TRANSFORMERS_URL);
    env.allowLocalModels = false;
    env.useBrowserCache = true;
    pinTransformersThreads(env);
    tokenizer = await AutoTokenizer.from_pretrained(CLIP_TOKENIZER_ID);
    return tokenizer;
  }

  async function tokenize(text, padId) {
    const tok = await ensureTokenizer();
    tok.pad_token_id = padId;
    const encoded = await tok(text ?? "", {
      padding: true,
      max_length: 77,
      truncation: true,
    });
    const raw = flattenIds(encoded.input_ids);
    const ids = new Int32Array(77);
    ids.fill(padId);
    ids.set(raw.slice(0, 77));
    return ids;
  }

  async function readWeights(model) {
    const buffers = {};
    for (const file of model.files) {
      const url = hfResolve(model.repo, file.path);
      buffers[file.path] = await fetchCached(url, {
        expectedBytes: file.bytes,
        returnBuffer: true,
      });
    }
    return buffers;
  }

  async function release() {
    for (const key of ["text", "unet", "vae"]) {
      const session = sessions[key];
      sessions[key] = null;
      if (session?.release) {
        try {
          await session.release();
        } catch {
          /* la sesión ya estaba suelta */
        }
      }
    }
    sessions.key = null;
    sessions.buffers = null;
  }

  async function load(modelKey, onStatus) {
    const model = SD_MODELS[modelKey];
    if (!model) throw new Error(`Modelo desconocido: ${modelKey}`);
    onStatus?.(`Creando sesiones WebGPU de ${model.label}…`);
    await release();
    const buffers = await readWeights(model);

    sessions.text = await ort.InferenceSession.create(
      buffers["text_encoder/model.onnx"],
      sessionOptions(model.session.textEncoder),
    );

    const unetOptions = sessionOptions(model.session.unet);
    if (model.externalWeights) {
      unetOptions.externalData = [
        {
          path: model.externalWeights,
          data: new Uint8Array(buffers["unet/weights.pb"]),
        },
      ];
    }
    sessions.unet = await ort.InferenceSession.create(buffers["unet/model.onnx"], unetOptions);
    sessions.vae = await ort.InferenceSession.create(
      buffers["vae_decoder/model.onnx"],
      sessionOptions(model.session.vae),
    );
    // ORT puede leer estos bytes durante toda la vida de la sesión.
    sessions.buffers = buffers;
    sessions.key = modelKey;
    onStatus?.(`${model.label} listo en la GPU.`);
    return modelKey;
  }

  async function encodeText(ids) {
    const outputs = await sessions.text.run({
      input_ids: new ort.Tensor("int32", ids, [1, ids.length]),
    });
    const hidden = pickOutput(outputs, "last_hidden_state");
    if (hidden.location === "gpu-buffer" && typeof hidden.getData === "function") {
      const data = await hidden.getData();
      hidden.dispose?.();
      return new ort.Tensor(hidden.type, data, hidden.dims);
    }
    return hidden;
  }

  function reuseHidden(hidden, float16) {
    if (float16 && hidden.type !== "float16") {
      const cpu = hidden.data instanceof Float32Array ? hidden.data : unpackToFloat32(hidden.data);
      return new ort.Tensor("float16", packFloat16(cpu), hidden.dims);
    }
    return new ort.Tensor(hidden.type || (float16 ? "float16" : "float32"), hidden.data, hidden.dims);
  }

  async function runUnet(sample, timestep, hidden, float16) {
    const sampleTensor = float16
      ? new ort.Tensor("float16", packFloat16(sample), LATENT_SHAPE)
      : new ort.Tensor("float32", sample, LATENT_SHAPE);
    const timestepTensor = float16
      ? new ort.Tensor("float16", packFloat16(Float32Array.from(timestep)), [1])
      : new ort.Tensor("int64", BigInt64Array.from(timestep.map((value) => BigInt(Math.round(value)))), [1]);
    const hiddenTensor = reuseHidden(hidden, float16);

    const outputs = await sessions.unet.run({
      sample: sampleTensor,
      timestep: timestepTensor,
      encoder_hidden_states: hiddenTensor,
    });
    return tensorFloat32(pickOutput(outputs, "out_sample"));
  }

  async function decode(latent, float16) {
    const tensor = float16
      ? new ort.Tensor("float16", packFloat16(latent), LATENT_SHAPE)
      : new ort.Tensor("float32", latent, LATENT_SHAPE);
    const outputs = await sessions.vae.run({ latent_sample: tensor });
    const image = pickOutput(outputs, "sample");
    const data = await tensorFloat32(image);
    return { data, dims: image.dims };
  }

  function rgbaFromNchw(data, dims) {
    const height = Number(dims[2]);
    const width = Number(dims[3]);
    const plane = height * width;
    const rgba = new Uint8ClampedArray(width * height * 4);
    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        const pixel = y * width + x;
        const offset = pixel * 4;
        for (let channel = 0; channel < 3; channel += 1) {
          const value = data[channel * plane + pixel] / 2 + 0.5;
          const clamped = value < 0 ? 0 : value > 1 ? 1 : value;
          rgba[offset + channel] = Math.round(clamped * 255);
        }
        rgba[offset + 3] = 255;
      }
    }
    return { rgba, width, height };
  }

  async function generateTurbo(prompt, seed) {
    const ids = await tokenize(prompt, 0);
    const hidden = await encodeText(ids);
    const latent = noiseLatent(TURBO_SIGMA, seed);
    const scaled = scaleLatent(latent, TURBO_SIGMA);
    const noise = await runUnet(scaled, [999], hidden, false);
    const sigmaHat = TURBO_SIGMA;
    const decoded = new Float32Array(latent.length);
    for (let i = 0; i < latent.length; i += 1) {
      const predOriginal = latent[i] - sigmaHat * noise[i];
      const derivative = (latent[i] - predOriginal) / sigmaHat;
      const dt = -sigmaHat;
      decoded[i] = (latent[i] + derivative * dt) / VAE_SCALE;
    }
    return decode(decoded, false);
  }

  async function generateEuler(prompt, negative, steps, guidance, seed, onStep) {
    const cond = await encodeText(await tokenize(prompt, 49407));
    const uncond = await encodeText(await tokenize(negative || "", 49407));
    const schedule = eulerSchedule(steps, { spacing: "linspace", stepsOffset: 1 });
    let latent = noiseLatent(schedule.sigmas[0], seed);
    for (let step = 0; step < schedule.timesteps.length; step += 1) {
      onStep?.(step + 1, schedule.timesteps.length);
      const sigma = schedule.sigmas[step];
      const scaled = scaleLatent(latent, sigma);
      const timestep = [schedule.timesteps[step]];
      const condNoise = await runUnet(scaled, timestep, cond, true);
      const uncondNoise = await runUnet(scaled, timestep, uncond, true);
      const guided = applyCfg(uncondNoise, condNoise, guidance);
      latent = eulerStep(latent, guided, sigma, schedule.sigmas[step + 1]);
    }
    const forVae = new Float32Array(latent.length);
    for (let i = 0; i < latent.length; i += 1) forVae[i] = latent[i] / VAE_SCALE;
    return decode(forVae, true);
  }

  async function generate(options) {
    const model = SD_MODELS[options.modelKey];
    if (!model) throw new Error("No hay modelo activo.");
    if (sessions.key !== options.modelKey) await load(options.modelKey, options.onStatus);
    const seed = Number.isFinite(options.seed) ? options.seed >>> 0 : (Math.random() * 0xffffffff) >>> 0;
    await ensureTokenizer(options.onStatus);
    const image = model.mode === "turbo"
      ? await generateTurbo(options.prompt, seed)
      : await generateEuler(
        options.prompt,
        options.negative,
        options.steps || model.defaultSteps,
        options.guidance ?? model.guidance,
        seed,
        options.onStep,
      );
    return { ...rgbaFromNchw(image.data, image.dims), seed };
  }

  return {
    load,
    generate,
    release,
    get active() {
      return sessions.key;
    },
  };
}

export async function pngFromRgba(rgba, width, height) {
  const canvas = new OffscreenCanvas(width, height);
  const context = canvas.getContext("2d");
  context.putImageData(new ImageData(rgba, width, height), 0, 0);
  const blob = await canvas.convertToBlob({ type: "image/png" });
  return blob.arrayBuffer();
}
