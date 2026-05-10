/**
 * Web Audio drum synthesizer.
 *
 * Synthesises four drum voices entirely in the browser (no audio files
 * to ship). Each voice is built from primitive nodes — oscillators for
 * pitched components, an AudioBufferSourceNode of white noise for the
 * unpitched components — shaped by a short ADSR-like envelope.
 *
 * The synthesis recipes follow the classic 808/909 sketches:
 *   - Kick:  low sine + fast pitch envelope (~150→50 Hz)
 *   - Snare: noise + 200 Hz tone
 *   - HiHat: high-pass filtered noise burst
 *   - Crash: long, lightly filtered noise tail
 *
 * Public API: `getAudioEngine()` returns a lazily constructed singleton.
 * Always call `engine.unlock()` from a user gesture (button click) before
 * any `engine.play()` — most browsers refuse to start the AudioContext
 * otherwise.
 */

export type DrumId = "kick" | "snare" | "hihat" | "crash" | "tom" | "clap";

export interface AudioEngine {
  unlock: () => Promise<void>;
  play: (drum: DrumId, velocity?: number) => void;
  playGuideTrack: (delaySeconds?: number) => void;
  stopGuideTrack: () => void;
  isReady: () => boolean;
  getContext: () => AudioContext | null;
}

let _engine: AudioEngine | null = null;

export function getAudioEngine(): AudioEngine {
  if (_engine) return _engine;

  let ctx: AudioContext | null = null;
  let masterGain: GainNode | null = null;
  let guideGain: GainNode | null = null;
  let noiseBuffer: AudioBuffer | null = null;

  const ensureContext = (): AudioContext => {
    if (ctx && ctx.state !== "closed") return ctx;
    const Ctor =
      (window as any).AudioContext || (window as any).webkitAudioContext;
    ctx = new Ctor();
    masterGain = ctx!.createGain();
    masterGain.gain.value = 0.85;
    masterGain.connect(ctx!.destination);
    noiseBuffer = makeNoiseBuffer(ctx!, 1.0);
    return ctx!;
  };

  const unlock = async () => {
    const c = ensureContext();
    if (c.state === "suspended") await c.resume();
  };

  const isReady = () => !!ctx && ctx.state === "running";

  const play = (drum: DrumId, velocity = 1.0) => {
    const c = ensureContext();
    if (c.state !== "running") {
      c.resume().catch(() => {});
    }
    const v = clamp(velocity, 0.05, 1.5);
    switch (drum) {
      case "kick":  return playKick(c, masterGain!, v);
      case "snare": return playSnare(c, masterGain!, noiseBuffer!, v);
      case "hihat": return playHiHat(c, masterGain!, noiseBuffer!, v);
      case "crash": return playCrash(c, masterGain!, noiseBuffer!, v);
      case "tom":   return playTom(c, masterGain!, v);
      case "clap":  return playClap(c, masterGain!, noiseBuffer!, v);
    }
  };

  const stopGuideTrack = () => {
    if (!guideGain || !ctx) return;
    try {
      guideGain.gain.cancelScheduledValues(ctx.currentTime);
      guideGain.gain.setTargetAtTime(0.0001, ctx.currentTime, 0.03);
      const old = guideGain;
      window.setTimeout(() => {
        try { old.disconnect(); } catch {}
      }, 180);
    } catch {}
    guideGain = null;
  };

  const playGuideTrack = (delaySeconds = 0.75) => {
    const c = ensureContext();
    if (c.state !== "running") c.resume().catch(() => {});
    stopGuideTrack();

    guideGain = c.createGain();
    guideGain.gain.setValueAtTime(0.001, c.currentTime);
    guideGain.gain.exponentialRampToValueAtTime(0.46, c.currentTime + 0.18);
    guideGain.connect(masterGain!);

    const start = c.currentTime + delaySeconds;
    scheduleMusicBed(c, guideGain, noiseBuffer!, start);
  };

  _engine = {
    unlock,
    play,
    playGuideTrack,
    stopGuideTrack,
    isReady,
    getContext: () => ctx,
  };
  return _engine;
}

// ── Voice synthesis ─────────────────────────────────────────

function playKick(ctx: AudioContext, dest: AudioNode, vel: number, at = ctx.currentTime) {
  const t0 = at;
  const osc = ctx.createOscillator();
  const gain = ctx.createGain();
  osc.type = "sine";
  // pitch envelope: 150 → 45 Hz
  osc.frequency.setValueAtTime(150, t0);
  osc.frequency.exponentialRampToValueAtTime(45, t0 + 0.12);
  // amp envelope
  gain.gain.setValueAtTime(0.001, t0);
  gain.gain.exponentialRampToValueAtTime(1.1 * vel, t0 + 0.005);
  gain.gain.exponentialRampToValueAtTime(0.001, t0 + 0.45);
  osc.connect(gain).connect(dest);
  osc.start(t0);
  osc.stop(t0 + 0.5);
}

function playSnare(
  ctx: AudioContext, dest: AudioNode, noise: AudioBuffer, vel: number,
  at = ctx.currentTime,
) {
  const t0 = at;

  // noise component
  const nSrc = ctx.createBufferSource();
  nSrc.buffer = noise;
  const nFilter = ctx.createBiquadFilter();
  nFilter.type = "highpass";
  nFilter.frequency.value = 1500;
  const nGain = ctx.createGain();
  nGain.gain.setValueAtTime(0.001, t0);
  nGain.gain.exponentialRampToValueAtTime(0.9 * vel, t0 + 0.005);
  nGain.gain.exponentialRampToValueAtTime(0.001, t0 + 0.18);
  nSrc.connect(nFilter).connect(nGain).connect(dest);
  nSrc.start(t0);
  nSrc.stop(t0 + 0.2);

  // tone (200 Hz) component
  const osc = ctx.createOscillator();
  osc.type = "triangle";
  osc.frequency.value = 200;
  const oGain = ctx.createGain();
  oGain.gain.setValueAtTime(0.001, t0);
  oGain.gain.exponentialRampToValueAtTime(0.6 * vel, t0 + 0.003);
  oGain.gain.exponentialRampToValueAtTime(0.001, t0 + 0.12);
  osc.connect(oGain).connect(dest);
  osc.start(t0);
  osc.stop(t0 + 0.13);
}

function playHiHat(
  ctx: AudioContext, dest: AudioNode, noise: AudioBuffer, vel: number,
  at = ctx.currentTime,
) {
  const t0 = at;
  const src = ctx.createBufferSource();
  src.buffer = noise;
  const hp = ctx.createBiquadFilter();
  hp.type = "highpass";
  hp.frequency.value = 7000;
  const bp = ctx.createBiquadFilter();
  bp.type = "bandpass";
  bp.frequency.value = 9000;
  bp.Q.value = 0.7;
  const gain = ctx.createGain();
  gain.gain.setValueAtTime(0.001, t0);
  gain.gain.exponentialRampToValueAtTime(0.55 * vel, t0 + 0.002);
  gain.gain.exponentialRampToValueAtTime(0.001, t0 + 0.07);
  src.connect(hp).connect(bp).connect(gain).connect(dest);
  src.start(t0);
  src.stop(t0 + 0.08);
}

function playCrash(
  ctx: AudioContext, dest: AudioNode, noise: AudioBuffer, vel: number,
  at = ctx.currentTime,
) {
  const t0 = at;
  const src = ctx.createBufferSource();
  src.buffer = noise;
  const hp = ctx.createBiquadFilter();
  hp.type = "highpass";
  hp.frequency.value = 4000;
  const peak = ctx.createBiquadFilter();
  peak.type = "peaking";
  peak.frequency.value = 6500;
  peak.Q.value = 1.2;
  peak.gain.value = 5;
  const gain = ctx.createGain();
  gain.gain.setValueAtTime(0.001, t0);
  gain.gain.exponentialRampToValueAtTime(0.7 * vel, t0 + 0.003);
  gain.gain.exponentialRampToValueAtTime(0.001, t0 + 1.4);
  src.connect(hp).connect(peak).connect(gain).connect(dest);
  src.start(t0);
  src.stop(t0 + 1.5);
}

function playTom(ctx: AudioContext, dest: AudioNode, vel: number, at = ctx.currentTime) {
  const t0 = at;
  const osc = ctx.createOscillator();
  const gain = ctx.createGain();
  osc.type = "sine";
  osc.frequency.setValueAtTime(170, t0);
  osc.frequency.exponentialRampToValueAtTime(85, t0 + 0.18);
  gain.gain.setValueAtTime(0.001, t0);
  gain.gain.exponentialRampToValueAtTime(0.9 * vel, t0 + 0.006);
  gain.gain.exponentialRampToValueAtTime(0.001, t0 + 0.34);
  osc.connect(gain).connect(dest);
  osc.start(t0);
  osc.stop(t0 + 0.36);
}

function playClap(
  ctx: AudioContext, dest: AudioNode, noise: AudioBuffer, vel: number,
  at = ctx.currentTime,
) {
  const t0 = at;
  for (let i = 0; i < 3; i++) {
    const t = t0 + i * 0.014;
    const src = ctx.createBufferSource();
    src.buffer = noise;
    const bp = ctx.createBiquadFilter();
    bp.type = "bandpass";
    bp.frequency.value = 1600;
    bp.Q.value = 0.9;
    const gain = ctx.createGain();
    gain.gain.setValueAtTime(0.001, t);
    gain.gain.exponentialRampToValueAtTime(0.38 * vel, t + 0.002);
    gain.gain.exponentialRampToValueAtTime(0.001, t + 0.055);
    src.connect(bp).connect(gain).connect(dest);
    src.start(t);
    src.stop(t + 0.065);
  }
}

function scheduleMusicBed(
  ctx: AudioContext,
  dest: AudioNode,
  noise: AudioBuffer,
  start: number,
) {
  const bpm = 96;
  const beat = 60 / bpm;
  const step = beat / 2;
  const bars = 8;
  const chords = [
    [261.63, 311.13, 392.0],
    [207.65, 246.94, 311.13],
    [233.08, 293.66, 349.23],
    [196.0, 246.94, 293.66],
  ];
  const bass = [130.81, 103.83, 116.54, 98.0];

  for (let bar = 0; bar < bars; bar++) {
    const barStart = start + bar * beat * 4;
    playChord(ctx, dest, chords[bar % chords.length], barStart, beat * 3.8);
    for (let b = 0; b < 4; b++) {
      const t = barStart + b * beat;
      playBass(ctx, dest, bass[bar % bass.length] * (b === 2 ? 1.5 : 1), t, beat * 0.46);
    }
    playKick(ctx, dest, 0.36, barStart);
    playSnare(ctx, dest, noise, 0.2, barStart + beat * 2);
    for (let s = 0; s < 8; s++) {
      playHiHat(ctx, dest, noise, s % 2 === 0 ? 0.16 : 0.08, barStart + s * step);
    }
  }

  const guideHits: Array<[DrumId, number, number]> = [
    ["kick", 0, 0.22], ["hihat", 2, 0.1], ["snare", 4, 0.18], ["hihat", 6, 0.1],
    ["kick", 8, 0.18], ["tom", 10, 0.16], ["snare", 12, 0.2], ["crash", 14, 0.16],
    ["kick", 16, 0.18], ["hihat", 18, 0.1], ["clap", 20, 0.16], ["hihat", 22, 0.1],
    ["snare", 24, 0.18], ["tom", 26, 0.16], ["kick", 28, 0.2], ["crash", 30, 0.16],
    ["kick", 32, 0.18], ["hihat", 34, 0.1], ["snare", 36, 0.2], ["hihat", 38, 0.1],
    ["kick", 40, 0.18], ["tom", 42, 0.16], ["clap", 44, 0.16], ["hihat", 46, 0.1],
    ["snare", 48, 0.18], ["crash", 50, 0.22],
  ];

  for (const [drum, stepIndex, velocity] of guideHits) {
    scheduleDrum(ctx, dest, noise, drum, start + stepIndex * step, velocity);
  }
}

function scheduleDrum(
  ctx: AudioContext,
  dest: AudioNode,
  noise: AudioBuffer,
  drum: DrumId,
  at: number,
  velocity: number,
) {
  switch (drum) {
    case "kick": return playKick(ctx, dest, velocity, at);
    case "snare": return playSnare(ctx, dest, noise, velocity, at);
    case "hihat": return playHiHat(ctx, dest, noise, velocity, at);
    case "crash": return playCrash(ctx, dest, noise, velocity, at);
    case "tom": return playTom(ctx, dest, velocity, at);
    case "clap": return playClap(ctx, dest, noise, velocity, at);
  }
}

function playBass(
  ctx: AudioContext,
  dest: AudioNode,
  freq: number,
  at: number,
  dur: number,
) {
  const osc = ctx.createOscillator();
  const gain = ctx.createGain();
  const filter = ctx.createBiquadFilter();
  osc.type = "sawtooth";
  osc.frequency.setValueAtTime(freq, at);
  filter.type = "lowpass";
  filter.frequency.setValueAtTime(420, at);
  gain.gain.setValueAtTime(0.001, at);
  gain.gain.exponentialRampToValueAtTime(0.12, at + 0.018);
  gain.gain.exponentialRampToValueAtTime(0.001, at + dur);
  osc.connect(filter).connect(gain).connect(dest);
  osc.start(at);
  osc.stop(at + dur + 0.02);
}

function playChord(
  ctx: AudioContext,
  dest: AudioNode,
  freqs: number[],
  at: number,
  dur: number,
) {
  const filter = ctx.createBiquadFilter();
  filter.type = "lowpass";
  filter.frequency.setValueAtTime(1100, at);
  const gain = ctx.createGain();
  gain.gain.setValueAtTime(0.001, at);
  gain.gain.exponentialRampToValueAtTime(0.055, at + 0.08);
  gain.gain.exponentialRampToValueAtTime(0.001, at + dur);
  filter.connect(gain).connect(dest);
  for (const freq of freqs) {
    const osc = ctx.createOscillator();
    osc.type = "triangle";
    osc.frequency.setValueAtTime(freq, at);
    osc.connect(filter);
    osc.start(at);
    osc.stop(at + dur + 0.04);
  }
}

// ── helpers ────────────────────────────────────────────────

function makeNoiseBuffer(ctx: AudioContext, seconds: number): AudioBuffer {
  const sampleRate = ctx.sampleRate;
  const buffer = ctx.createBuffer(1, Math.floor(sampleRate * seconds), sampleRate);
  const data = buffer.getChannelData(0);
  for (let i = 0; i < data.length; i++) data[i] = Math.random() * 2 - 1;
  return buffer;
}

function clamp(x: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, x));
}
