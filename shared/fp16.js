/**
 * shared/fp16.js
 * Conversión IEEE-754 binaria entre float32 y float16.
 * El UNet pequeño exportado por diffusers pide tensores float16; en navegadores
 * sin Float16Array usamos los bits en un Uint16Array.
 */

/** float32 -> bits float16 (uint16). Redondeo al par. */
export function f32ToF16Bits(value) {
  const f32 = new Float32Array(1);
  const u32 = new Uint32Array(f32.buffer);
  f32[0] = value;
  const x = u32[0];
  const sign = (x >>> 16) & 0x8000;
  const mantissa = x & 0x7fffff;
  const exponent = (x >>> 23) & 0xff;

  if (exponent === 255) {
    if (mantissa) return sign | 0x7e00;
    return sign | 0x7c00;
  }

  const halfExponent = exponent - 127 + 15;
  if (halfExponent >= 31) return sign | 0x7c00;
  if (halfExponent <= 0) {
    if (halfExponent < -10) return sign;
    const m = mantissa | 0x800000;
    const shift = 1 - halfExponent;
    let halfMantissa = m >>> (shift + 13);
    const roundBit = (m >>> (shift + 12)) & 1;
    const stickyMask = (1 << (shift + 12)) - 1;
    const sticky = (m & stickyMask) !== 0;
    if (roundBit && (sticky || (halfMantissa & 1))) halfMantissa += 1;
    return sign | halfMantissa;
  }

  let halfMantissa = mantissa >>> 13;
  const roundBit = (mantissa >>> 12) & 1;
  const sticky = (mantissa & 0xfff) !== 0;
  if (roundBit && (sticky || (halfMantissa & 1))) {
    halfMantissa += 1;
    if (halfMantissa === 0x400) {
      return sign | ((halfExponent + 1) << 10);
    }
  }
  return sign | (halfExponent << 10) | halfMantissa;
}

/** bits float16 -> número JS. */
export function f16BitsToF32(bits) {
  const sign = (bits & 0x8000) << 16;
  const exponent = (bits >>> 10) & 0x1f;
  const mantissa = bits & 0x3ff;
  const u32 = new Uint32Array(1);
  const f32 = new Float32Array(u32.buffer);

  if (exponent === 0) {
    if (mantissa === 0) {
      u32[0] = sign;
      return f32[0];
    }
    let e = -14;
    let m = mantissa;
    while ((m & 0x400) === 0) {
      m <<= 1;
      e -= 1;
    }
    m &= 0x3ff;
    u32[0] = sign | ((e + 127) << 23) | (m << 13);
    return f32[0];
  }

  if (exponent === 31) {
    u32[0] = sign | 0x7f800000 | (mantissa << 13);
    return f32[0];
  }

  u32[0] = sign | ((exponent - 15 + 127) << 23) | (mantissa << 13);
  return f32[0];
}

/** Float32Array -> Float16Array si existe, si no Uint16Array con los bits. */
export function packFloat16(float32) {
  if (typeof Float16Array !== "undefined") {
    const out = new Float16Array(float32.length);
    for (let i = 0; i < float32.length; i += 1) out[i] = float32[i];
    return out;
  }
  const out = new Uint16Array(float32.length);
  for (let i = 0; i < float32.length; i += 1) out[i] = f32ToF16Bits(float32[i]);
  return out;
}

/** Salida de ORT (float32, Float16Array o Uint16Array de bits) -> Float32Array. */
export function unpackToFloat32(data) {
  if (data instanceof Float32Array) return data;
  if (typeof Float16Array !== "undefined" && data instanceof Float16Array) {
    const out = new Float32Array(data.length);
    for (let i = 0; i < data.length; i += 1) out[i] = data[i];
    return out;
  }
  if (data instanceof Uint16Array) {
    const out = new Float32Array(data.length);
    for (let i = 0; i < data.length; i += 1) out[i] = f16BitsToF32(data[i]);
    return out;
  }
  return Float32Array.from(data);
}
