"use client";

import Link from "next/link";
import {
  useCallback, useEffect, useRef, useState,
} from "react";
import {
  DRUM_CONFIG,
  DrumEngine,
  DrumPad,
  DrumPadId,
  EmaSmoother,
  estimateBpm,
  HitEvent,
  velocityToGain,
} from "../../lib/predictive";
import { getAudioEngine } from "../../lib/audio";
import {
  drawFilter, FilterId, FILTER_LIST,
} from "../../lib/filters";
import type { Fingertip, HandsHandle } from "../../lib/hands";
import type { FaceMeshHandle } from "../../lib/face-mesh";

type Mode = "drum" | "filter";
type ChallengeStatus = "idle" | "playing" | "finished";
type HitRating = "perfect" | "great" | "good" | "miss";

interface SmoothedTip extends Fingertip {
  smoothed: { x: number; y: number };
}

interface ChallengeNote {
  id: number;
  pad: DrumPadId;
  time: number;
  rating?: HitRating;
}

interface CelebrationParticle {
  x: number;
  y: number;
  vx: number;
  vy: number;
  born: number;
  life: number;
  color: string;
  size: number;
}

const MAX_TRAIL = 18;
const HIT_WINDOW_MS = 220;
const PERFECT_WINDOW_MS = 75;
const GREAT_WINDOW_MS = 140;
const CHALLENGE_LOOKAHEAD_MS = 2600;
const CHALLENGE_READY_MS = 900;
const CHALLENGE_BPM = 96;
const CHALLENGE_STEP_MS = 60000 / CHALLENGE_BPM / 2;
const CHALLENGE_PATTERN: { pad: DrumPadId; step: number; time: number }[] = [
  ["kick", 0], ["hihat", 2], ["snare", 4], ["hihat", 6],
  ["kick", 8], ["tom", 10], ["snare", 12], ["crash", 14],
  ["kick", 16], ["hihat", 18], ["clap", 20], ["hihat", 22],
  ["snare", 24], ["tom", 26], ["kick", 28], ["crash", 30],
  ["kick", 32], ["hihat", 34], ["snare", 36], ["hihat", 38],
  ["kick", 40], ["tom", 42], ["clap", 44], ["hihat", 46],
  ["snare", 48], ["crash", 50],
].map(([pad, step]) => ({
  pad: pad as DrumPadId,
  step: step as number,
  time: (step as number) * CHALLENGE_STEP_MS,
}));
const CHALLENGE_DURATION_MS =
  CHALLENGE_PATTERN[CHALLENGE_PATTERN.length - 1].time + 1400;

export default function MagicMirrorPage() {
  // ── DOM refs ──────────────────────────────────────────────
  const videoRef = useRef<HTMLVideoElement>(null);
  const displayRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);

  // ── shared ────────────────────────────────────────────────
  const [mode, setMode] = useState<Mode>("drum");
  const modeRef = useRef<Mode>("drum");
  const [active, setActive] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [renderFps, setRenderFps] = useState(0);
  const [handsLoading, setHandsLoading] = useState(false);

  // ── drum ──────────────────────────────────────────────────
  const drumEngineRef = useRef<DrumEngine>(new DrumEngine());
  const padsRef = useRef<DrumPad[]>([]);
  const handsRef = useRef<HandsHandle | null>(null);
  const fingertipsRef = useRef<SmoothedTip[]>([]);
  const trailsRef = useRef<Map<string, { x: number; y: number; t: number }[]>>(
    new Map(),
  );
  const smoothersRef = useRef<Map<string, EmaSmoother>>(new Map());
  const padFlashRef = useRef<Record<DrumPadId, number>>({
    kick: 0, snare: 0, hihat: 0, crash: 0, tom: 0, clap: 0,
  });
  const splashesRef = useRef<{ x: number; y: number; t: number; color: string }[]>([]);
  const celebrationRef = useRef<CelebrationParticle[]>([]);
  const hitTimesRef = useRef<number[]>([]);
  const [bpm, setBpm] = useState(0);
  const [hitCount, setHitCount] = useState(0);
  const [handInferenceMs, setHandInferenceMs] = useState(0);
  const challengeStatusRef = useRef<ChallengeStatus>("idle");
  const challengeStartRef = useRef(0);
  const challengeNotesRef = useRef<ChallengeNote[]>([]);
  const challengeScoreRef = useRef(0);
  const challengeComboRef = useRef(0);
  const challengeHitsRef = useRef(0);
  const challengeBestComboRef = useRef(0);
  const lastChallengeUiRef = useRef(0);
  const [challengeStatus, setChallengeStatus] = useState<ChallengeStatus>("idle");
  const [challengeScore, setChallengeScore] = useState(0);
  const [challengeCombo, setChallengeCombo] = useState(0);
  const [challengeBestCombo, setChallengeBestCombo] = useState(0);
  const [challengeAccuracy, setChallengeAccuracy] = useState(100);
  const [challengeMessage, setChallengeMessage] = useState("Ready for a beat run");
  const [challengeTime, setChallengeTime] = useState(0);

  // ── filter ────────────────────────────────────────────────
  const meshRef = useRef<FaceMeshHandle | null>(null);
  const [filter, setFilter] = useState<FilterId>("sunglasses");
  const filterRef = useRef<FilterId>("sunglasses");

  // ── render state ──────────────────────────────────────────
  const animRef = useRef(0);
  const runningRef = useRef(false);
  const lastFrameTimeRef = useRef(0);
  const lastHandsSendTimeRef = useRef(0);
  const lastMeshSendTimeRef = useRef(0);

  // ── pad layout ────────────────────────────────────────────
  const recomputePads = useCallback((width: number, height: number) => {
    const size = clamp(Math.min(width, height) * 0.17, 70, 140);
    const wide = size * 1.18;
    const short = size * 0.86;
    const layout: {
      id: DrumPadId;
      label: string;
      color: string;
      cx: number;
      cy: number;
      w: number;
      h: number;
    }[] = [
      { id: "hihat", label: "Hi-Hat", color: "#0ea5e9", cx: 0.14, cy: 0.61, w: wide, h: short },
      { id: "snare", label: "Snare", color: "#a855f7", cx: 0.32, cy: 0.72, w: wide, h: size },
      { id: "kick", label: "Kick", color: "#f43f5e", cx: 0.49, cy: 0.82, w: wide, h: short },
      { id: "tom", label: "Tom", color: "#14b8a6", cx: 0.51, cy: 0.61, w: size, h: short },
      { id: "clap", label: "Clap", color: "#fb923c", cx: 0.67, cy: 0.73, w: size, h: short },
      { id: "crash", label: "Crash", color: "#facc15", cx: 0.84, cy: 0.60, w: wide, h: short },
    ];
    padsRef.current = layout.map((p) => {
      const padW = p.w;
      const padH = p.h;
      const x0 = clamp(width * p.cx - padW / 2, 12, width - padW - 12);
      const y0 = clamp(height * p.cy - padH / 2, 78, height - padH - 12);
      return {
        id: p.id, label: p.label, color: p.color,
        polygon: [
          [x0,        y0],
          [x0 + padW, y0],
          [x0 + padW, y0 + padH],
          [x0,        y0 + padH],
        ],
      };
    });
    drumEngineRef.current.setPads(padsRef.current);
  }, []);

  // ── init both MediaPipe models lazily ─────────────────────
  const ensureHands = useCallback(async () => {
    if (handsRef.current) return handsRef.current;
    setHandsLoading(true);
    const { createHands } = await import("../../lib/hands");
    const display = displayRef.current!;
    const handle = await createHands({
      displayWidth: () => display.width || 640,
      displayHeight: () => display.height || 480,
      mirror: true,
      drumEngine: drumEngineRef.current,
      onHit: (hit) => onHit(hit),
      delegate: "GPU",
      minConfidence: 0.4,
    });
    handsRef.current = handle;
    setHandsLoading(false);
    return handle;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const ensureMesh = useCallback(async () => {
    if (meshRef.current) return meshRef.current;
    const { createFaceMesh } = await import("../../lib/face-mesh");
    const handle = await createFaceMesh();
    meshRef.current = handle;
    return handle;
  }, []);

  // ── camera lifecycle ──────────────────────────────────────
  const startCamera = useCallback(async () => {
    setError(null);
    try {
      await getAudioEngine().unlock();
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 }, height: { ideal: 720 },
          facingMode: "user",
        },
        audio: false,
      });
      streamRef.current = stream;
      const video = videoRef.current!;
      video.srcObject = stream;
      await video.play();

      runningRef.current = true;
      drumEngineRef.current.reset();
      smoothersRef.current.clear();
      trailsRef.current.clear();
      fingertipsRef.current = [];
      splashesRef.current = [];
      celebrationRef.current = [];
      hitTimesRef.current = [];
      setBpm(0);
      setHitCount(0);
      setActive(true);

      // Filter mode keeps the drum pads live, so hands are always needed.
      if (modeRef.current === "filter") {
        await Promise.all([ensureHands(), ensureMesh()]);
      } else {
        await ensureHands();
      }

      animRef.current = requestAnimationFrame(displayLoop);
    } catch (e: any) {
      console.error(e);
      setError("Camera access denied or unavailable.");
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ensureHands, ensureMesh]);

  const stopAll = useCallback(() => {
    runningRef.current = false;
    cancelAnimationFrame(animRef.current);
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    handsRef.current?.close();
    handsRef.current = null;
    meshRef.current?.close();
    meshRef.current = null;
    getAudioEngine().stopGuideTrack();
    challengeStatusRef.current = "idle";
    setChallengeStatus("idle");
    setActive(false);
  }, []);

  // ── hit handler ───────────────────────────────────────────
  function beginChallenge() {
    if (!active) {
      setChallengeMessage("Start the camera first");
      return;
    }
    modeRef.current = "drum";
    setMode("drum");
    challengeStatusRef.current = "playing";
    challengeStartRef.current = performance.now() + CHALLENGE_READY_MS;
    challengeNotesRef.current = CHALLENGE_PATTERN.map((n, id) => ({ ...n, id }));
    celebrationRef.current = [];
    challengeScoreRef.current = 0;
    challengeComboRef.current = 0;
    challengeHitsRef.current = 0;
    challengeBestComboRef.current = 0;
    lastChallengeUiRef.current = 0;
    setChallengeStatus("playing");
    setChallengeScore(0);
    setChallengeCombo(0);
    setChallengeBestCombo(0);
    setChallengeAccuracy(100);
    setChallengeTime(0);
    setChallengeMessage("Get ready");
    getAudioEngine().stopGuideTrack();
    getAudioEngine().playGuideTrack(CHALLENGE_READY_MS / 1000);
  }

  function finishChallenge() {
    if (challengeStatusRef.current !== "playing") return;
    challengeStatusRef.current = "finished";
    const accuracy = Math.round(
      (challengeHitsRef.current / CHALLENGE_PATTERN.length) * 100,
    );
    setChallengeStatus("finished");
    setChallengeScore(challengeScoreRef.current);
    setChallengeCombo(0);
    setChallengeBestCombo(challengeBestComboRef.current);
    setChallengeAccuracy(accuracy);
    setChallengeTime(CHALLENGE_DURATION_MS);
    setChallengeMessage(
      accuracy >= 85 ? "Encore" :
      accuracy >= 60 ? "Groove" :
      "Retry",
    );
    if (accuracy >= 50) spawnCelebration(performance.now(), accuracy);
  }

  function spawnCelebration(now: number, accuracy: number) {
    const W = displayRef.current?.width || 1280;
    const H = displayRef.current?.height || 720;
    const palette = accuracy >= 85
      ? ["#facc15", "#fb7185", "#38bdf8", "#a78bfa", "#ffffff"]
      : ["#38bdf8", "#22c55e", "#facc15", "#ffffff"];
    const bursts = accuracy >= 85 ? 5 : 3;
    const particles: CelebrationParticle[] = [];
    for (let b = 0; b < bursts; b++) {
      const cx = W * (0.22 + Math.random() * 0.56);
      const cy = H * (0.2 + Math.random() * 0.26);
      const count = accuracy >= 85 ? 34 : 24;
      for (let i = 0; i < count; i++) {
        const angle = (Math.PI * 2 * i) / count + Math.random() * 0.28;
        const speed = 1.4 + Math.random() * (accuracy >= 85 ? 4.6 : 3.4);
        particles.push({
          x: cx,
          y: cy,
          vx: Math.cos(angle) * speed,
          vy: Math.sin(angle) * speed - 1.2,
          born: now + b * 180,
          life: 950 + Math.random() * 700,
          color: palette[i % palette.length],
          size: 2 + Math.random() * 3,
        });
      }
    }
    celebrationRef.current = particles;
  }

  function judgeChallengeHit(hit: HitEvent) {
    if (challengeStatusRef.current !== "playing") return;
    const elapsed = performance.now() - challengeStartRef.current;
    if (elapsed < -HIT_WINDOW_MS || elapsed > CHALLENGE_DURATION_MS) return;

    let best: ChallengeNote | null = null;
    let bestDelta = Infinity;
    for (const note of challengeNotesRef.current) {
      if (note.rating || note.pad !== hit.pad.id) continue;
      const delta = Math.abs(note.time - elapsed);
      if (delta < bestDelta) {
        best = note;
        bestDelta = delta;
      }
    }

    if (!best || bestDelta > HIT_WINDOW_MS) {
      challengeComboRef.current = 0;
      setChallengeCombo(0);
      setChallengeMessage("Off beat");
      return;
    }

    const rating: HitRating =
      bestDelta <= PERFECT_WINDOW_MS ? "perfect" :
      bestDelta <= GREAT_WINDOW_MS ? "great" : "good";
    best.rating = rating;

    const base = rating === "perfect" ? 1000 : rating === "great" ? 700 : 450;
    challengeComboRef.current += 1;
    challengeBestComboRef.current = Math.max(
      challengeBestComboRef.current,
      challengeComboRef.current,
    );
    challengeHitsRef.current += 1;
    challengeScoreRef.current += Math.round(
      base * (1 + Math.min(12, challengeComboRef.current) * 0.035),
    );

    const accuracy = Math.round(
      (challengeHitsRef.current / CHALLENGE_PATTERN.length) * 100,
    );
    setChallengeScore(challengeScoreRef.current);
    setChallengeCombo(challengeComboRef.current);
    setChallengeBestCombo(challengeBestComboRef.current);
    setChallengeAccuracy(accuracy);
    setChallengeMessage(rating.toUpperCase());
  }

  function updateChallengeFrame(now: number) {
    if (challengeStatusRef.current !== "playing") return;
    const elapsed = now - challengeStartRef.current;
    if (elapsed < 0) {
      if (now - lastChallengeUiRef.current > 120) {
        lastChallengeUiRef.current = now;
        setChallengeTime(0);
      }
      return;
    }

    let missed = false;
    for (const note of challengeNotesRef.current) {
      if (!note.rating && elapsed - note.time > HIT_WINDOW_MS) {
        note.rating = "miss";
        missed = true;
      }
    }
    if (missed) {
      challengeComboRef.current = 0;
      setChallengeCombo(0);
      setChallengeMessage("MISS");
    }

    if (now - lastChallengeUiRef.current > 120) {
      lastChallengeUiRef.current = now;
      setChallengeTime(Math.min(CHALLENGE_DURATION_MS, Math.max(0, elapsed)));
      setChallengeScore(challengeScoreRef.current);
      setChallengeCombo(challengeComboRef.current);
      setChallengeBestCombo(challengeBestComboRef.current);
      setChallengeAccuracy(Math.round(
        (challengeHitsRef.current / CHALLENGE_PATTERN.length) * 100,
      ));
    }

    if (elapsed >= CHALLENGE_DURATION_MS) finishChallenge();
  }

  const onHit = (hit: HitEvent) => {
    const gain = velocityToGain(hit.velocity);
    getAudioEngine().play(hit.pad.id, gain);
    padFlashRef.current[hit.pad.id] = performance.now();
    splashesRef.current.push({
      x: hit.predictedAt.x, y: hit.predictedAt.y,
      t: performance.now(), color: hit.pad.color,
    });
    if (splashesRef.current.length > 12) splashesRef.current.shift();
    hitTimesRef.current.push(hit.timestamp);
    if (hitTimesRef.current.length > 16) hitTimesRef.current.shift();
    setBpm(estimateBpm(hitTimesRef.current));
    setHitCount((c) => c + 1);
    judgeChallengeHit(hit);
  };

  // ── mode switch (live) ────────────────────────────────────
  const switchMode = useCallback(async (next: Mode) => {
    if (next === mode) return;
    modeRef.current = next;
    setMode(next);
    if (!active) return;
    if (next === "drum")  await ensureHands();
    if (next === "filter") await Promise.all([ensureHands(), ensureMesh()]);
  }, [mode, active, ensureHands, ensureMesh]);

  // ── display + inference loop ──────────────────────────────
  const displayLoop = useCallback(() => {
    if (!runningRef.current) return;

    const video = videoRef.current;
    const display = displayRef.current;
    if (!video || !display || video.readyState < 2) {
      animRef.current = requestAnimationFrame(displayLoop);
      return;
    }
    if (display.width !== video.videoWidth)  display.width  = video.videoWidth;
    if (display.height !== video.videoHeight) display.height = video.videoHeight;
    const ctx = display.getContext("2d")!;
    const W = display.width, H = display.height;

    if (padsRef.current.length === 0) recomputePads(W, H);

    // Mirror the video horizontally — selfie view feels natural
    ctx.save();
    ctx.translate(W, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(video, 0, 0, W, H);
    ctx.restore();

    // Throttle ML inference to leave headroom on weaker GPUs.
    const now = performance.now();
    const currentMode = modeRef.current;
    updateChallengeFrame(now);

    if (handsRef.current && now - lastHandsSendTimeRef.current > 22) {
      lastHandsSendTimeRef.current = now;
      handsRef.current.send(video);
    }
    if (
      currentMode === "filter" &&
      meshRef.current &&
      now - lastMeshSendTimeRef.current > 33
    ) {
      lastMeshSendTimeRef.current = now;
      meshRef.current.send(video);
    }

    // Pull latest fingertips (always read fresh — handle exposes a getter)
    if (handsRef.current) {
      const raw = handsRef.current.fingertips;
      const smoothed: SmoothedTip[] = raw.map((tip) => {
        let s = smoothersRef.current.get(tip.hand);
        if (!s) {
          s = new EmaSmoother(0.5);
          smoothersRef.current.set(tip.hand, s);
        }
        return { ...tip, smoothed: s.update(tip.x, tip.y) };
      });
      // Drop smoothers for hands no longer visible (so they re-snap on
      // re-appearance instead of tweening from a stale position).
      const present = new Set(raw.map((r) => r.hand));
      for (const key of Array.from(smoothersRef.current.keys())) {
        if (!present.has(key as any)) smoothersRef.current.delete(key);
      }
      fingertipsRef.current = smoothed;
      setHandInferenceMs(handsRef.current.inferenceMs);
      pushTrails(smoothed, now);
    }

    if (lastFrameTimeRef.current) {
      const dt = now - lastFrameTimeRef.current;
      if (dt > 0) {
        const inst = 1000 / dt;
        setRenderFps((p) => Math.round(p * 0.85 + inst * 0.15));
      }
    }
    lastFrameTimeRef.current = now;

    if (currentMode === "filter") {
      drawFilterOverlay(ctx);
      drawDrumOverlay(ctx, W, H, now);
    } else {
      drawDrumOverlay(ctx, W, H, now);
    }

    animRef.current = requestAnimationFrame(displayLoop);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [recomputePads]);

  // ── trail bookkeeping ─────────────────────────────────────
  const pushTrails = (tips: SmoothedTip[], now: number) => {
    const present = new Set<string>();
    for (const tip of tips) {
      present.add(tip.hand);
      const arr = trailsRef.current.get(tip.hand) ?? [];
      arr.push({ x: tip.smoothed.x, y: tip.smoothed.y, t: now });
      while (arr.length > MAX_TRAIL) arr.shift();
      trailsRef.current.set(tip.hand, arr);
    }
    for (const key of Array.from(trailsRef.current.keys())) {
      if (!present.has(key)) trailsRef.current.delete(key);
    }
  };

  // ── overlay drawing ───────────────────────────────────────
  const drawDrumOverlay = (
    ctx: CanvasRenderingContext2D, W: number, H: number, now: number,
  ) => {
    // Trails (under the pads so pads stay legible)
    for (const [hand, trail] of trailsRef.current) {
      const color = hand === "Left" ? "#22d3ee" : "#fb923c";
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      for (let i = 1; i < trail.length; i++) {
        const a = trail[i - 1];
        const b = trail[i];
        const alpha = i / trail.length;
        ctx.strokeStyle = hexToRgba(color, alpha * 0.7);
        ctx.lineWidth = 4 + alpha * 6;
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.stroke();
      }
    }

    // Pads
    for (const pad of padsRef.current) {
      const flashAge = now - padFlashRef.current[pad.id];
      const flash = Math.max(0, 1 - flashAge / 260);
      const baseAlpha = 0.18;
      const alpha = baseAlpha + flash * 0.6;
      ctx.fillStyle = hexToRgba(pad.color, alpha);
      ctx.strokeStyle = hexToRgba(pad.color, 0.95);
      ctx.lineWidth = 3 + flash * 6;
      drawPolygon(ctx, pad.polygon);
      ctx.fill();
      ctx.stroke();

      const padW = pad.polygon[1][0] - pad.polygon[0][0];
      const padH = pad.polygon[2][1] - pad.polygon[1][1];
      const cx = pad.polygon[0][0] + padW / 2;
      const cy = pad.polygon[0][1] + padH / 2;
      ctx.fillStyle = "#fff";
      ctx.font = `700 ${Math.round(padH * 0.18)}px Inter, system-ui, sans-serif`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.shadowColor = "rgba(0,0,0,0.8)";
      ctx.shadowBlur = 12;
      ctx.fillText(pad.label, cx, cy);
      ctx.shadowBlur = 0;

      if (flash > 0.05) {
        ctx.strokeStyle = hexToRgba("#ffffff", flash);
        ctx.lineWidth = 2 + flash * 10;
        drawPolygon(ctx, pad.polygon);
        ctx.stroke();
      }
    }
    ctx.textAlign = "left";
    ctx.textBaseline = "alphabetic";

    // Hit splashes
    for (const s of splashesRef.current) {
      const age = (now - s.t) / 320;
      if (age >= 1) continue;
      const r = 18 + age * 70;
      ctx.strokeStyle = hexToRgba(s.color, 1 - age);
      ctx.lineWidth = 5 * (1 - age);
      ctx.beginPath();
      ctx.arc(s.x, s.y, r, 0, Math.PI * 2);
      ctx.stroke();
    }

    // Fingertip cursors (drawn last, on top of everything)
    for (const tip of fingertipsRef.current) {
      const color = tip.hand === "Left" ? "#22d3ee" : "#fb923c";
      const sx = tip.smoothed.x;
      const sy = tip.smoothed.y;

      // Draw arm line from wrist to fingertip (subtle)
      ctx.strokeStyle = hexToRgba(color, 0.5);
      ctx.lineWidth = 2;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(tip.wristX, tip.wristY);
      ctx.lineTo(sx, sy);
      ctx.stroke();
      ctx.setLineDash([]);

      // Outer ring
      ctx.strokeStyle = "rgba(255,255,255,0.95)";
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.arc(sx, sy, 16, 0, Math.PI * 2);
      ctx.stroke();
      // Filled core
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(sx, sy, 11, 0, Math.PI * 2);
      ctx.fill();
      // Centre highlight
      ctx.fillStyle = "rgba(255,255,255,0.85)";
      ctx.beginPath();
      ctx.arc(sx - 3, sy - 3, 4, 0, Math.PI * 2);
      ctx.fill();

      // Hand label
      ctx.fillStyle = "rgba(0,0,0,0.6)";
      ctx.fillRect(sx + 18, sy - 14, 50, 18);
      ctx.fillStyle = "#fff";
      ctx.font = "11px Inter, system-ui";
      ctx.fillText(tip.hand === "Left" ? "Left" : "Right", sx + 24, sy);
    }

    // HUD
    ctx.font = "12px Inter, system-ui";
    ctx.fillStyle = "rgba(0,0,0,0.6)";
    ctx.fillRect(W - 240, 10, 230, 96);
    ctx.fillStyle = "#fff";
    ctx.fillText(`Hands inference: ${handInferenceMs.toFixed(0)} ms`, W - 230, 30);
    ctx.fillText(`Render: ${renderFps} FPS`, W - 230, 48);
    ctx.fillText(`Hits: ${hitCount}`, W - 230, 66);
    ctx.fillText(bpm > 0 ? `BPM: ${bpm}` : "BPM: —", W - 230, 84);
    if (fingertipsRef.current.length === 0) {
      ctx.fillStyle = "rgba(0,0,0,0.55)";
      ctx.fillRect(W / 2 - 130, 14, 260, 30);
      ctx.fillStyle = "#fff";
      ctx.textAlign = "center";
      ctx.fillText("Show your hands to the camera", W / 2, 33);
      ctx.textAlign = "left";
    }
    drawChallengeOverlay(ctx, W, H, now);
    drawCelebration(ctx, W, H, now);
  };

  const drawCelebration = (
    ctx: CanvasRenderingContext2D, W: number, H: number, now: number,
  ) => {
    if (
      challengeStatusRef.current === "finished" &&
      challengeScoreRef.current > 0
    ) {
      ctx.save();
      ctx.fillStyle = "rgba(0,0,0,0.58)";
      roundRect(ctx, W / 2 - 170, H * 0.18, 340, 96, 16);
      ctx.fill();
      ctx.textAlign = "center";
      ctx.fillStyle = "#fff";
      ctx.font = "800 26px Inter, system-ui, sans-serif";
      ctx.fillText(challengeMessage, W / 2, H * 0.18 + 38);
      ctx.font = "700 15px Inter, system-ui, sans-serif";
      ctx.fillStyle = "rgba(255,255,255,0.86)";
      ctx.fillText(
        `Score ${challengeScoreRef.current.toLocaleString()} | Accuracy ${challengeAccuracy}%`,
        W / 2,
        H * 0.18 + 68,
      );
      ctx.restore();
    }

    if (celebrationRef.current.length === 0) return;
    const alive: CelebrationParticle[] = [];
    ctx.save();
    ctx.globalCompositeOperation = "lighter";
    for (const p of celebrationRef.current) {
      const age = now - p.born;
      if (age < 0) {
        alive.push(p);
        continue;
      }
      if (age > p.life) continue;
      const t = age / p.life;
      const x = p.x + p.vx * age * 0.075;
      const y = p.y + p.vy * age * 0.075 + 110 * t * t;
      const alpha = Math.max(0, 1 - t);
      ctx.fillStyle = hexToRgba(p.color, alpha);
      ctx.beginPath();
      ctx.arc(x, y, p.size * (1 + t * 1.2), 0, Math.PI * 2);
      ctx.fill();
      alive.push(p);
    }
    ctx.restore();
    celebrationRef.current = alive;
  };

  const drawChallengeOverlay = (
    ctx: CanvasRenderingContext2D, W: number, H: number, now: number,
  ) => {
    const status = challengeStatusRef.current;
    if (status === "idle") return;

    const pads = padsRef.current;
    if (pads.length === 0) return;
    const padById = new Map(pads.map((p) => [p.id, p]));
    const elapsed = now - challengeStartRef.current;
    const targetX = W * 0.18;
    const endX = W * 0.88;
    const top = Math.max(118, H * 0.16);
    const laneH = Math.max(19, H * 0.033);
    const laneGap = Math.max(5, H * 0.008);
    const panelW = endX - targetX + 130;
    const order: DrumPadId[] = ["hihat", "snare", "kick", "tom", "clap", "crash"];
    const panelH = laneH * order.length + laneGap * (order.length - 1) + 48;
    const panelX = targetX - 86;

    ctx.save();
    ctx.fillStyle = "rgba(0,0,0,0.56)";
    roundRect(ctx, panelX, top - 14, panelW, panelH, 14);
    ctx.fill();

    ctx.fillStyle = "#fff";
    ctx.font = "700 14px Inter, system-ui, sans-serif";
    ctx.fillText(
      status === "playing" && elapsed < 0 ? "Get ready" :
      status === "finished" ? "Beat run complete" : "Beat run",
      panelX + 14,
      top + 2,
    );
    ctx.font = "12px Inter, system-ui, sans-serif";
    ctx.fillStyle = "rgba(255,255,255,0.8)";
    ctx.fillText(
      `${challengeMessage}  |  Score ${challengeScoreRef.current}`,
      panelX + 118,
      top + 2,
    );

    const laneTop = top + 24;
    for (let i = 0; i < order.length; i++) {
      const pad = padById.get(order[i]);
      if (!pad) continue;
      const y = laneTop + i * (laneH + laneGap);
      ctx.fillStyle = "rgba(255,255,255,0.08)";
      roundRect(ctx, targetX, y, endX - targetX, laneH, laneH / 2);
      ctx.fill();
      ctx.fillStyle = hexToRgba(pad.color, 0.35);
      roundRect(ctx, targetX - 74, y, 62, laneH, laneH / 2);
      ctx.fill();
      ctx.fillStyle = "#fff";
      ctx.font = "700 11px Inter, system-ui, sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(pad.label, targetX - 43, y + laneH / 2);
    }

    ctx.strokeStyle = "rgba(255,255,255,0.9)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(targetX, laneTop - 2);
    ctx.lineTo(targetX, laneTop + order.length * laneH + (order.length - 1) * laneGap + 2);
    ctx.stroke();

    if (status === "playing") {
      for (const note of challengeNotesRef.current) {
        if (note.rating && note.rating !== "miss") continue;
        const dt = note.time - Math.max(0, elapsed);
        if (dt < -HIT_WINDOW_MS || dt > CHALLENGE_LOOKAHEAD_MS) continue;
        const pad = padById.get(note.pad);
        if (!pad) continue;
        const laneIndex = order.indexOf(note.pad);
        const x = targetX + (dt / CHALLENGE_LOOKAHEAD_MS) * (endX - targetX);
        const y = laneTop + laneIndex * (laneH + laneGap) + laneH / 2;
        const r = Math.max(9, laneH * 0.36);
        ctx.fillStyle =
          note.rating === "miss" ? "rgba(255,255,255,0.18)" : hexToRgba(pad.color, 0.94);
        ctx.strokeStyle = "#fff";
        ctx.lineWidth = dt <= HIT_WINDOW_MS ? 4 : 2;
        ctx.beginPath();
        ctx.arc(x, y, r, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
      }
    }

    ctx.textAlign = "left";
    ctx.textBaseline = "alphabetic";
    ctx.restore();
  };

  const drawFilterOverlay = (ctx: CanvasRenderingContext2D) => {
    const faces = meshRef.current?.faces ?? [];
    if (faces.length === 0) return;
    // FaceMesh receives the original (un-mirrored) video, so we mirror x.
    for (const lm of faces) {
      const mirrored = lm.map((p) => ({ x: 1 - p.x, y: p.y, z: p.z }));
      drawFilter(ctx, mirrored, filterRef.current, ctx.canvas.width, ctx.canvas.height);
    }
  };

  // ── lifecycle ─────────────────────────────────────────────
  useEffect(() => {
    modeRef.current = mode;
  }, [mode]);

  useEffect(() => {
    filterRef.current = filter;
  }, [filter]);

  useEffect(() => () => stopAll(), [stopAll]);

  // Reload models when the active mode changes mid-session
  useEffect(() => {
    if (!active) return;
    if (mode === "drum")  ensureHands();
    if (mode === "filter") {
      ensureHands();
      ensureMesh();
    }
  }, [mode, active, ensureHands, ensureMesh]);

  // ── render ────────────────────────────────────────────────
  return (
    <main className="min-h-screen bg-gradient-to-b from-slate-50 to-white">
      <nav className="fixed top-0 w-full z-50 bg-white/80 backdrop-blur-md border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
          <Link href="/" className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-fuchsia-500 to-cyan-500 flex items-center justify-center text-white font-bold">M</div>
            <span className="font-semibold text-gray-900 text-lg">
              AI <span className="text-fuchsia-600">Magic Mirror</span>
            </span>
          </Link>
          <div className="flex items-center gap-4 text-sm text-gray-500">
            <Link href="/camera" className="hover:text-gray-900">Camera</Link>
            <Link href="/demo"   className="hover:text-gray-900">Video Demo</Link>
            <Link href="/"       className="hover:text-gray-900">&larr; Home</Link>
          </div>
        </div>
      </nav>

      <div className="pt-24 pb-12 px-6 max-w-6xl mx-auto">
        <div className="mb-6 flex flex-col md:flex-row gap-4 md:items-end justify-between">
          <div>
            <h1 className="text-3xl md:text-4xl font-bold text-gray-900 mb-2">
              <span className="bg-gradient-to-r from-fuchsia-600 to-cyan-600 bg-clip-text text-transparent">
                AI Magic Mirror
              </span>
            </h1>
            <p className="text-gray-500 max-w-xl">
              {mode === "drum"
                ? "Aim your index fingertip at the pads. MediaPipe Hands tracks 21 keypoints per hand, in your browser, at 30–60 fps."
                : "Snapchat-style AR filters anchored by 468 facial landmarks, with the drum pads still live."}
            </p>
          </div>
          {!active ? (
            <button
              onClick={startCamera}
              className="px-6 py-3 rounded-xl bg-gradient-to-r from-fuchsia-600 to-cyan-600 text-white font-semibold hover:shadow-lg transition-all"
            >
              Start Camera
            </button>
          ) : (
            <button
              onClick={stopAll}
              className="px-6 py-3 rounded-xl bg-gray-900 text-white font-semibold hover:bg-black transition-all"
            >
              Stop
            </button>
          )}
        </div>

        <div className="flex items-center gap-2 mb-5">
          <ModeButton label="Drum Mode"   emoji="🥁" current={mode} target="drum"   onClick={() => switchMode("drum")} />
          <ModeButton label="Filter Mode" emoji="🕶️" current={mode} target="filter" onClick={() => switchMode("filter")} />
          {handsLoading && (
            <span className="text-xs text-gray-400 ml-2 animate-pulse">
              loading hand model…
            </span>
          )}
        </div>

        {error && (
          <div className="rounded-xl p-4 border border-red-200 bg-red-50 mb-5">
            <p className="text-red-600 text-sm">⚠ {error}</p>
          </div>
        )}

        <div className="rounded-2xl overflow-hidden bg-black relative aspect-video shadow-xl">
          <video ref={videoRef} className="hidden" playsInline muted />
          <canvas
            ref={displayRef}
            className="w-full h-full object-contain bg-black"
          />
          {!active && (
            <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
              <div className="text-center text-white/80">
                <div className="text-6xl mb-3 opacity-30">
                  {mode === "drum" ? "🥁" : "🕶️"}
                </div>
                <p className="text-sm tracking-wide opacity-70">
                  Click <span className="font-semibold">Start Camera</span> to begin
                </p>
              </div>
            </div>
          )}
        </div>

        {mode === "filter" && (
          <div className="mt-5 flex flex-wrap items-center gap-2">
            {FILTER_LIST.map((f) => (
              <button
                key={f.id}
                onClick={() => setFilter(f.id)}
                className={
                  "px-4 py-2 rounded-full text-sm font-medium transition-all " +
                  (filter === f.id
                    ? "bg-fuchsia-600 text-white shadow"
                    : "bg-white border border-gray-200 text-gray-700 hover:bg-gray-50")
                }
              >
                <span className="mr-1.5">{f.emoji}</span>
                {f.label}
              </button>
            ))}
          </div>
        )}

        {mode === "drum" && (
          <>
          <div className="mt-5 grid grid-cols-2 md:grid-cols-4 gap-3">
            <Stat label="Hand inference" value={`${handInferenceMs.toFixed(0)} ms`} />
            <Stat label="Render" value={`${renderFps} FPS`} />
            <Stat label="Hits"   value={`${hitCount}`} />
            <Stat label="BPM"    value={bpm > 0 ? `${bpm}` : "—"} />
          </div>

          <div className="mt-4 bg-white border border-gray-100 rounded-xl p-4">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-3">
              <div>
                <div className="text-sm font-semibold text-gray-900">
                  Beat Challenge
                </div>
                <div className="text-xs text-gray-500 mt-1">
                  Follow the moving notes, hit the matching pad on the target line, and build combo.
                </div>
              </div>
              <button
                onClick={beginChallenge}
                disabled={!active}
                className="px-4 py-2 rounded-lg bg-gray-900 text-white text-sm font-semibold disabled:opacity-40 disabled:cursor-not-allowed hover:bg-black transition"
              >
                {challengeStatus === "playing" ? "Restart Run" : "Start Run"}
              </button>
            </div>
            <div className="mt-4 grid grid-cols-2 md:grid-cols-5 gap-3">
              <Stat label="Score" value={challengeScore.toLocaleString()} />
              <Stat label="Combo" value={`${challengeCombo}x`} />
              <Stat label="Best" value={`${challengeBestCombo}x`} />
              <Stat label="Accuracy" value={`${challengeAccuracy}%`} />
              <Stat
                label={challengeStatus === "finished" ? "Result" : "Time"}
                value={
                  challengeStatus === "finished"
                    ? challengeMessage
                    : `${Math.ceil(Math.max(0, CHALLENGE_DURATION_MS - challengeTime) / 1000)}s`
                }
              />
            </div>
          </div>
          </>
        )}

        <div className="mt-8 text-xs text-gray-400 leading-relaxed">
          <p>
            <span className="font-semibold text-gray-500">How it works:</span>{" "}
            Both modes run entirely in your browser via MediaPipe (Hands for
            drumming, FaceMesh for filters). Drum mode uses the index-fingertip
            landmark as the cursor, and a velocity-reversal hit detector that
            extrapolates {DRUM_CONFIG.PREDICT_AHEAD_MS} ms forward to compensate
            inference latency. Audio is synthesised via Web Audio — no asset files.
          </p>
        </div>
      </div>
    </main>
  );
}

// ── small components ──────────────────────────────────────

function ModeButton(props: {
  label: string; emoji: string;
  current: Mode; target: Mode;
  onClick: () => void;
}) {
  const isActive = props.current === props.target;
  return (
    <button
      onClick={props.onClick}
      className={
        "px-5 py-2.5 rounded-xl text-sm font-semibold transition-all " +
        (isActive
          ? "bg-gray-900 text-white shadow"
          : "bg-white border border-gray-200 text-gray-700 hover:bg-gray-50")
      }
    >
      <span className="mr-1.5">{props.emoji}</span>
      {props.label}
    </button>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-white rounded-xl border border-gray-100 p-3 text-center">
      <div className="text-xs text-gray-400 uppercase tracking-wide">{label}</div>
      <div className="text-xl font-bold text-gray-900 mt-0.5 font-mono">
        {value}
      </div>
    </div>
  );
}

// ── helpers ───────────────────────────────────────────────

function drawPolygon(ctx: CanvasRenderingContext2D, poly: number[][]) {
  ctx.beginPath();
  ctx.moveTo(poly[0][0], poly[0][1]);
  for (let i = 1; i < poly.length; i++) ctx.lineTo(poly[i][0], poly[i][1]);
  ctx.closePath();
}

function roundRect(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  r: number,
) {
  const radius = Math.min(r, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + radius, y);
  ctx.lineTo(x + w - radius, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + radius);
  ctx.lineTo(x + w, y + h - radius);
  ctx.quadraticCurveTo(x + w, y + h, x + w - radius, y + h);
  ctx.lineTo(x + radius, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - radius);
  ctx.lineTo(x, y + radius);
  ctx.quadraticCurveTo(x, y, x + radius, y);
  ctx.closePath();
}

function hexToRgba(hex: string, alpha: number): string {
  const h = hex.replace("#", "");
  const bigint = parseInt(h.length === 3
    ? h.split("").map((c) => c + c).join("")
    : h, 16);
  const r = (bigint >> 16) & 255;
  const g = (bigint >> 8) & 255;
  const b = bigint & 255;
  return `rgba(${r},${g},${b},${alpha})`;
}

function clamp(x: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, x));
}
