/**
 * shared/selfcheck.mjs
 * Comprueba el planificador, float16 y el parser de herramientas sin navegador.
 *   node shared/selfcheck.mjs
 */

import assert from "node:assert/strict";
import { f16BitsToF32, f32ToF16Bits } from "./fp16.js";
import { eulerSchedule, trainingSigmas } from "./scheduler.js";
import { parseToolCalls, visibleReply } from "./tools.js";

const sigmas = trainingSigmas();
assert.ok(Math.abs(sigmas[999] - 14.614641229333639) < 1e-6, `sigma 999 = ${sigmas[999]}`);

const turbo = eulerSchedule(1, { spacing: "trailing" });
assert.equal(turbo.timesteps[0], 999);
assert.ok(Math.abs(turbo.sigmas[0] - 14.614641229333639) < 1e-6);
assert.equal(turbo.sigmas.at(-1), 0);

const multi = eulerSchedule(15, { spacing: "linspace", stepsOffset: 1 });
assert.equal(multi.timesteps.length, 15);
assert.equal(multi.sigmas.length, 16);
assert.ok(multi.timesteps[0] > multi.timesteps.at(-1));
assert.ok(Math.abs(multi.sigmas[0] - sigmas[999]) < 1e-6);

for (const value of [0, 1, -1, 0.5, 999, 14.6146, -0.18215]) {
  const back = f16BitsToF32(f32ToF16Bits(value));
  const tolerance = Math.max(1e-3, Math.abs(value) * 1e-3);
  assert.ok(Math.abs(back - value) <= tolerance, `${value} -> ${back}`);
}

const fenced = parseToolCalls(
  'Aquí va.\n```tool\n{"tool":"generate_image","arguments":{"prompt":"a red door"}}\n```',
);
assert.equal(fenced.length, 1);
assert.equal(fenced[0].name, "generate_image");
assert.equal(fenced[0].arguments.prompt, "a red door");

const named = parseToolCalls('{"name":"speak","arguments":{"text":"hola"}}');
assert.equal(named[0].name, "speak");
assert.equal(named[0].arguments.text, "hola");

const ignored = parseToolCalls("Solo texto, sin herramienta.");
assert.equal(ignored.length, 0);

const visible = visibleReply('Hola.\n```tool\n{"tool":"speak","arguments":{"text":"x"}}\n```');
assert.equal(visible, "Hola.");

const thought = parseToolCalls('<think>{"tool":"speak","arguments":{"text":"no"}}</think>\nNada.');
assert.equal(thought.length, 0);

console.log("selfcheck ok");
