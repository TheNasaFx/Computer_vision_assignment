/**
 * Predictive Hit Detection engine.
 *
 * Fed with fingertip samples (from MediaPipe Hands), this state machine
 * decides when a downward strike has occurred and which drum pad it
 * landed on. Two refinements keep the timing tight:
 *
 *   1. **Velocity-reversal trigger.** A real strike has the kinematic
 *      signature of fast downward motion that suddenly decelerates (the
 *      hand "bouncing" off the imaginary drum head). We watch for the
 *      first frame where the y-velocity collapses, not for "fingertip
 *      inside pad polygon" — that test would also fire on horizontal
 *      sweeps, and would miss fast hits that pass through the pad in
 *      one frame.
 *
 *   2. **Forward extrapolation.** The fingertip sample we observe is
 *      already ~30 ms old (camera shutter → MediaPipe inference →
 *      callback). When a strike fires, we predict the fingertip's
 *      position PREDICT_AHEAD_MS into the future and test *that* point
 *      against the pad polygons. The audio fires at the moment the
 *      fingertip will hit, not the moment we noticed.
 *
 * The class is a pure JS state machine — no DOM, no audio — so it is
 * trivial to unit-test and trivial to swap audio backends.
 */

export interface Vec2 {
  x: number;
  y: number;
}

export interface Sample {
  x: number;
  y: number;
  t: number;
}

export type DrumPadId = "kick" | "snare" | "hihat" | "crash" | "tom" | "clap";

export interface DrumPad {
  id: DrumPadId;
  polygon: number[][];
  label: string;
  color: string;
}

export interface HitEvent {
  pad: DrumPad;
  sourceKey: string;
  velocity: number;
  predictedAt: Vec2;
  timestamp: number;
}

// ── tunable parameters ─────────────────────────────────────
//
// Defaults are tuned for *index-fingertip* tracking from MediaPipe Hands
// at ~30–60 fps. Full-body wrist pose at 5–10 fps would need looser
// thresholds and a larger PREDICT_AHEAD_MS.

export const DRUM_CONFIG = {
  /** How many recent samples to keep in the ring buffer. */
  HISTORY: 6,
  /** Forward-extrapolation horizon for the predicted hit point (ms). */
  PREDICT_AHEAD_MS: 25,
  /** Minimum downward speed (px/ms) of the *previous* frame to count
   *  the current frame as a strike candidate. */
  VELOCITY_THRESHOLD: 0.7,
  /** Refractory period between hits on the same source (ms). */
  COOLDOWN_MS: 90,
  /** Strike confirmed when v_now.y < REVERSAL_RATIO × v_prev.y. */
  REVERSAL_RATIO: 0.55,
  /** A strike is "predominantly vertical" when |vy| ≥ this × |vx|. */
  VERTICAL_DOMINANCE: 0.85,
  /** Velocity (px/ms) that maps to maximum gain. */
  VELOCITY_FOR_MAX_GAIN: 3.0,
  GAIN_MIN: 0.35,
  GAIN_MAX: 1.1,
};

// ── per-source state ───────────────────────────────────────

interface SourceState {
  buffer: Sample[];
  lastHitTime: number;
  lastSampleTime: number;
}

function newSourceState(): SourceState {
  return { buffer: [], lastHitTime: -Infinity, lastSampleTime: -Infinity };
}

// ── kinematics ─────────────────────────────────────────────

interface Kinematics {
  v: Vec2;
  a: Vec2;
  current: Sample;
  prevV: Vec2;
}

function computeKinematics(buf: Sample[]): Kinematics | null {
  if (buf.length < 3) return null;
  const p0 = buf[buf.length - 3];
  const p1 = buf[buf.length - 2];
  const p2 = buf[buf.length - 1];
  const dt1 = Math.max(1, p1.t - p0.t);
  const dt2 = Math.max(1, p2.t - p1.t);
  const v1: Vec2 = { x: (p1.x - p0.x) / dt1, y: (p1.y - p0.y) / dt1 };
  const v2: Vec2 = { x: (p2.x - p1.x) / dt2, y: (p2.y - p1.y) / dt2 };
  const a: Vec2 = { x: (v2.x - v1.x) / dt2, y: (v2.y - v1.y) / dt2 };
  return { v: v2, a, current: p2, prevV: v1 };
}

// ── point-in-polygon (even-odd) ────────────────────────────

export function pointInPolygon(p: Vec2, poly: number[][]): boolean {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i];
    const [xj, yj] = poly[j];
    const intersect =
      yi > p.y !== yj > p.y &&
      p.x < ((xj - xi) * (p.y - yi)) / (yj - yi + 1e-9) + xi;
    if (intersect) inside = !inside;
  }
  return inside;
}

// ── public engine ──────────────────────────────────────────

export class DrumEngine {
  private states: Map<string, SourceState> = new Map();
  private pads: DrumPad[] = [];

  setPads(pads: DrumPad[]) {
    this.pads = pads;
  }

  pads_(): DrumPad[] {
    return this.pads;
  }

  reset() {
    this.states.clear();
  }

  /** Push a new sample for a named source ("Left", "Right", "hand_0", …)
   *  and return a HitEvent if the sample completes a strike pattern. */
  ingest(sourceKey: string, x: number, y: number, t: number): HitEvent | null {
    let state = this.states.get(sourceKey);
    if (!state) {
      state = newSourceState();
      this.states.set(sourceKey, state);
    }

    if (t <= state.lastSampleTime) return null;
    state.lastSampleTime = t;

    state.buffer.push({ x, y, t });
    if (state.buffer.length > DRUM_CONFIG.HISTORY) state.buffer.shift();

    if (t - state.lastHitTime < DRUM_CONFIG.COOLDOWN_MS) return null;

    const k = computeKinematics(state.buffer);
    if (!k) return null;

    // Strike detection runs on the *previous* frame's velocity, since by
    // the time we observe the strike the fingertip has already begun to
    // decelerate. A real strike has |prev v| above threshold and the
    // current frame collapses |v.y|.
    const prevSpeed = Math.hypot(k.prevV.x, k.prevV.y);
    if (prevSpeed < DRUM_CONFIG.VELOCITY_THRESHOLD) return null;

    const wasGoingDown = k.prevV.y > DRUM_CONFIG.VELOCITY_THRESHOLD * 0.5;
    const decelerating = k.v.y < k.prevV.y * DRUM_CONFIG.REVERSAL_RATIO;
    if (!(wasGoingDown && decelerating)) return null;

    // Vertical-dominance check — guards against horizontal sweeps.
    if (
      Math.abs(k.prevV.y) <
      Math.abs(k.prevV.x) * DRUM_CONFIG.VERTICAL_DOMINANCE
    ) {
      return null;
    }

    // Forward-extrapolate the *strike* (using prevV, the velocity into
    // the impact — the post-impact v is essentially noise).
    const dt = DRUM_CONFIG.PREDICT_AHEAD_MS;
    const px = k.current.x + k.prevV.x * dt + 0.5 * k.a.x * dt * dt;
    const py = k.current.y + k.prevV.y * dt + 0.5 * k.a.y * dt * dt;

    // Pad collision: try the current point first (most likely correct
    // for slow strikes), then the predicted point as a fallback.
    const pad =
      this.findPad(k.current) ?? this.findPad({ x: px, y: py });
    if (!pad) return null;

    state.lastHitTime = t;
    return {
      pad,
      sourceKey,
      velocity: prevSpeed,
      predictedAt: { x: px, y: py },
      timestamp: t,
    };
  }

  private findPad(p: Vec2): DrumPad | null {
    for (const pad of this.pads) {
      if (pointInPolygon(p, pad.polygon)) return pad;
    }
    return null;
  }
}

// ── helpers ────────────────────────────────────────────────

export function velocityToGain(speed: number): number {
  const f = Math.min(1, speed / DRUM_CONFIG.VELOCITY_FOR_MAX_GAIN);
  return DRUM_CONFIG.GAIN_MIN +
         (DRUM_CONFIG.GAIN_MAX - DRUM_CONFIG.GAIN_MIN) * f;
}

export function estimateBpm(timestamps: number[]): number {
  if (timestamps.length < 4) return 0;
  const recent = timestamps.slice(-8);
  const intervals: number[] = [];
  for (let i = 1; i < recent.length; i++) {
    intervals.push(recent[i] - recent[i - 1]);
  }
  intervals.sort((a, b) => a - b);
  const trimmed = intervals.slice(1, -1);
  if (trimmed.length === 0) return 0;
  const avg = trimmed.reduce((a, b) => a + b, 0) / trimmed.length;
  if (avg <= 0) return 0;
  return Math.round(60000 / avg);
}

/**
 * Exponential moving average smoother.
 *
 * Used only for the *displayed* cursor — the predictive engine consumes
 * raw samples so its velocity calculations stay accurate. Damps sub-
 * pixel jitter so the dot looks calm even when the underlying landmarks
 * shake by a pixel or two.
 */
export class EmaSmoother {
  private state: { x: number; y: number } | null = null;

  constructor(private readonly alpha: number = 0.45) {}

  /** Mix a new sample in. Returns the smoothed point. */
  update(x: number, y: number): { x: number; y: number } {
    if (!this.state) {
      this.state = { x, y };
    } else {
      this.state.x = this.state.x * (1 - this.alpha) + x * this.alpha;
      this.state.y = this.state.y * (1 - this.alpha) + y * this.alpha;
    }
    return { x: this.state.x, y: this.state.y };
  }

  reset() {
    this.state = null;
  }
}
