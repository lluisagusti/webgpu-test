/**
 * shared/scheduler.js
 * Planificador de Euler discreto para Stable Diffusion (predicción epsilon).
 *
 * El repositorio del modelo pequeño publica DPM-Solver++. Euler usa la misma
 * agenda de betas scaled_linear y cabe en el navegador sin otra dependencia.
 * SD-Turbo no pasa por aquí: su único paso está fijado en sd-engine.js para
 * coincidir con el ejemplo de ONNX Runtime Web (sigma 14.6146, timestep 999).
 */

export function trainingSigmas(betaStart = 0.00085, betaEnd = 0.012, count = 1000) {
  const start = Math.sqrt(betaStart);
  const end = Math.sqrt(betaEnd);
  const sigmas = new Float64Array(count);
  let alphaCum = 1;
  for (let i = 0; i < count; i += 1) {
    const t = count === 1 ? 0 : i / (count - 1);
    const beta = (start + (end - start) * t) ** 2;
    alphaCum *= 1 - beta;
    sigmas[i] = Math.sqrt((1 - alphaCum) / alphaCum);
  }
  return sigmas;
}

function interpolateClamp(values, x) {
  if (x <= 0) return values[0];
  const last = values.length - 1;
  if (x >= last) return values[last];
  const index = Math.floor(x);
  const fraction = x - index;
  return values[index] * (1 - fraction) + values[index + 1] * fraction;
}

/**
 * @param {number} steps
 * @param {{ spacing?: "linspace" | "trailing", stepsOffset?: number, numTrain?: number }} [options]
 * @returns {{ timesteps: number[], sigmas: number[] }} sigmas tiene un 0 final
 */
export function eulerSchedule(steps, options = {}) {
  const spacing = options.spacing ?? "linspace";
  const stepsOffset = options.stepsOffset ?? 1;
  const numTrain = options.numTrain ?? 1000;
  const trained = trainingSigmas(0.00085, 0.012, numTrain);
  const timesteps = [];

  if (spacing === "trailing") {
    // diffusers: arange(num_train, 0, -ratio) y luego restar 1. Con 1 paso sale 999.
    const ratio = numTrain / steps;
    for (let i = 0; i < steps; i += 1) {
      timesteps.push(Math.round(numTrain - i * ratio) - 1);
    }
  } else {
    for (let i = 0; i < steps; i += 1) {
      const t = steps === 1 ? 0 : i / (steps - 1);
      timesteps.push(t * (numTrain - 1));
    }
    timesteps.reverse();
    for (let i = 0; i < timesteps.length; i += 1) timesteps[i] += stepsOffset;
  }

  const sigmas = timesteps.map((timestep) => interpolateClamp(trained, timestep));
  sigmas.push(0);
  return { timesteps, sigmas };
}

/** latent_model_input = sample / sqrt(sigma^2 + 1) */
export function scaleLatent(sample, sigma) {
  const divisor = Math.sqrt(sigma * sigma + 1);
  const out = new Float32Array(sample.length);
  for (let i = 0; i < sample.length; i += 1) out[i] = sample[i] / divisor;
  return out;
}

/** Un paso de Euler. `sample` es el latente sin escalar. */
export function eulerStep(sample, noisePred, sigma, sigmaNext) {
  const out = new Float32Array(sample.length);
  const denom = sigma === 0 ? 1 : sigma;
  const dt = sigmaNext - sigma;
  for (let i = 0; i < sample.length; i += 1) {
    const predOriginal = sample[i] - sigma * noisePred[i];
    const derivative = (sample[i] - predOriginal) / denom;
    out[i] = sample[i] + derivative * dt;
  }
  return out;
}

/** guidance: uncond + escala * (cond - uncond) */
export function applyCfg(uncond, cond, guidance) {
  const out = new Float32Array(cond.length);
  for (let i = 0; i < cond.length; i += 1) {
    out[i] = uncond[i] + guidance * (cond[i] - uncond[i]);
  }
  return out;
}
