/**
 * Hand-landmark wrapper built on MediaPipe Tasks-Vision.
 *
 * Why this replaces the previous Solutions-API wrapper
 * ----------------------------------------------------
 * The legacy `@mediapipe/hands` package shipped Emscripten output that
 * relied on a global `Module` variable being live in the same realm as
 * the WASM loader scripts. Next.js's bundler isolates module scope and
 * its dev server wraps `XMLHttpRequest`, so under that environment the
 * loader's `xhr.onprogress` callback finds `Module.dataFileDownloads`
 * undefined and throws — even with self-hosted assets and even with
 * <script>-tag injection.
 *
 * Tasks-Vision is MediaPipe's modern API. It ships as an ES module,
 * loads its WASM via `FilesetResolver`, and has no dependence on a
 * global `Module`. It plays nicely with bundlers and the Next.js dev
 * server.
 *
 * Public surface kept intentionally identical to the previous wrapper
 * so the page consumers don't need to change.
 */

import {
  HandLandmarker,
  FilesetResolver,
} from "@mediapipe/tasks-vision";
import type { DrumEngine, HitEvent } from "./predictive";

const WASM_BASE = "/mediapipe/tasks/wasm";
const MODEL_PATH = "/mediapipe/tasks/models/hand_landmarker.task";

export type HandLandmark = { x: number; y: number; z: number };
export type HandLandmarks = HandLandmark[];
export type Handedness = "Left" | "Right";

export interface Fingertip {
  hand: Handedness;
  index: number;
  x: number;
  y: number;
  z: number;
  wristX: number;
  wristY: number;
  confidence: number;
  t: number;
}

export interface HandsHandle {
  send: (video: HTMLVideoElement) => Promise<void>;
  fingertips: Fingertip[];
  inferenceMs: number;
  close: () => void;
}

export interface HandsOptions {
  displayWidth: () => number;
  displayHeight: () => number;
  mirror?: boolean;
  drumEngine?: DrumEngine;
  onHit?: (hit: HitEvent) => void;
  minConfidence?: number;
  /** "GPU" runs through WebGL and is materially faster. Falls back to
   *  "CPU" automatically if the browser lacks WebGL support. */
  delegate?: "GPU" | "CPU";
}

export async function createHands(opts: HandsOptions): Promise<HandsHandle> {
  // FilesetResolver figures out which WASM to use (SIMD vs. nosimd) and
  // patches the model path resolution.
  const fileset = await FilesetResolver.forVisionTasks(WASM_BASE);
  const landmarker = await HandLandmarker.createFromOptions(fileset, {
    baseOptions: {
      modelAssetPath: MODEL_PATH,
      delegate: opts.delegate ?? "GPU",
    },
    runningMode: "VIDEO",
    numHands: 2,
    minHandDetectionConfidence: 0.5,
    minHandPresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });

  const minConfidence = opts.minConfidence ?? 0.4;
  const mirror = opts.mirror ?? true;

  let fingertips: Fingertip[] = [];
  let inferenceMs = 0;
  let lastSendTs = 0;

  // detectForVideo requires a strictly-increasing timestamp (ms).
  let lastVideoTs = 0;

  const send = async (video: HTMLVideoElement) => {
    if (video.readyState < 2) return;
    if (typeof video.currentTime !== "number") return;

    const t0 = performance.now();
    let videoTs = Math.round(t0);
    if (videoTs <= lastVideoTs) videoTs = lastVideoTs + 1;
    lastVideoTs = videoTs;

    let result;
    try {
      result = landmarker.detectForVideo(video, videoTs);
    } catch (err) {
      // Tasks-Vision throws if called too quickly back-to-back; benign.
      return;
    }
    inferenceMs = performance.now() - t0;
    lastSendTs = t0;

    const W = opts.displayWidth();
    const H = opts.displayHeight();
    const tips: Fingertip[] = [];

    const lmList = result.landmarks ?? [];
    const handed = result.handedness ?? [];

    for (let i = 0; i < lmList.length; i++) {
      const lm = lmList[i];
      const handMeta = handed[i]?.[0];
      const conf = handMeta?.score ?? 1;
      if (conf < minConfidence) continue;
      const handLabel = (handMeta?.categoryName ?? "Right") as Handedness;

      const tip = lm[HANDS_KP.INDEX_TIP];
      const wrist = lm[HANDS_KP.WRIST];
      const tx = mirror ? (1 - tip.x) * W : tip.x * W;
      const ty = tip.y * H;
      const wx = mirror ? (1 - wrist.x) * W : wrist.x * W;
      const wy = wrist.y * H;

      const ft: Fingertip = {
        hand: handLabel,
        index: i,
        x: tx, y: ty, z: tip.z,
        wristX: wx, wristY: wy,
        confidence: conf,
        t: performance.now(),
      };
      tips.push(ft);

      if (opts.drumEngine && opts.onHit) {
        const hit = opts.drumEngine.ingest(handLabel, tx, ty, ft.t);
        if (hit) opts.onHit(hit);
      }
    }
    fingertips = tips;
  };

  const close = () => {
    try { landmarker.close(); } catch {}
  };

  const handle: HandsHandle = Object.create({ send, close });
  Object.defineProperty(handle, "fingertips", {
    get: () => fingertips, enumerable: true,
  });
  Object.defineProperty(handle, "inferenceMs", {
    get: () => inferenceMs, enumerable: true,
  });
  return handle;
}

// ── canonical landmark indices (MediaPipe Hands) ──────────

export const HANDS_KP = {
  WRIST: 0,
  THUMB_CMC: 1, THUMB_MCP: 2, THUMB_IP: 3, THUMB_TIP: 4,
  INDEX_MCP: 5, INDEX_PIP: 6, INDEX_DIP: 7, INDEX_TIP: 8,
  MIDDLE_MCP: 9, MIDDLE_PIP: 10, MIDDLE_DIP: 11, MIDDLE_TIP: 12,
  RING_MCP: 13, RING_PIP: 14, RING_DIP: 15, RING_TIP: 16,
  PINKY_MCP: 17, PINKY_PIP: 18, PINKY_DIP: 19, PINKY_TIP: 20,
} as const;
