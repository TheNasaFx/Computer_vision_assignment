"use client";

import Link from "next/link";
import { useState, useRef, useEffect, useCallback } from "react";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

type OwnershipItem = {
  class_name: string;
  confidence: number;
  distance: number;
  bbox: number[];
};

type PersonInventory = {
  track_id: number | null;
  person_bbox: number[];
  num_items: number;
  items: OwnershipItem[];
};

type AlertData = {
  alert_id: string;
  alert_type: string;
  severity: string;
  message: string;
  timestamp: number;
};

type ZoneData = {
  zone_id: string;
  name: string;
  occupied: boolean;
  occupant_id: number | null;
  occupied_seconds: number;
};

type StudySpaceData = {
  inference_ms: number;
  num_persons: number;
  num_objects: number;
  zones: {
    total: number;
    occupied: number;
    available: number;
    zones: ZoneData[];
  };
  inventories: PersonInventory[];
  unowned_objects: { class_name: string; bbox: number[] }[];
  active_alerts: AlertData[];
};

/* ── Person colors ── */
const PERSON_COLORS = [
  "#0891b2", "#7c3aed", "#dc2626", "#ea580c", "#16a34a",
  "#2563eb", "#c026d3", "#ca8a04", "#0d9488", "#4f46e5",
];

export default function StudySpacePage() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const captureRef = useRef<HTMLCanvasElement>(null);
  const displayRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const runningRef = useRef(false);
  const busyRef = useRef(false);
  const animRef = useRef(0);
  const ssDataRef = useRef<StudySpaceData | null>(null);
  const captureScaleRef = useRef(1);

  const [mode, setMode] = useState<"none" | "camera" | "video">("none");
  const [active, setActive] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [fps, setFps] = useState(0);
  const [inferenceMs, setInferenceMs] = useState(0);
  const [ssData, setSsData] = useState<StudySpaceData | null>(null);
  const [totalFrames, setTotalFrames] = useState(0);
  const [file, setFile] = useState<File | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [videoEnded, setVideoEnded] = useState(false);

  /* ── Draw study-space boxes on canvas ── */
  const drawOverlay = (
    ctx: CanvasRenderingContext2D,
    data: StudySpaceData | null,
    vw: number
  ) => {
    if (!data) return;
    const s = captureScaleRef.current;
    const lineW = Math.max(2, Math.round(vw / 400));
    const fontSize = Math.max(11, Math.round(vw / 55));
    ctx.font = `bold ${fontSize}px Inter, system-ui, sans-serif`;

    /* Draw ownership lines + person boxes */
    for (const inv of data.inventories) {
      const pid = inv.track_id ?? 0;
      const color = PERSON_COLORS[pid % PERSON_COLORS.length];
      const [px1, py1, px2, py2] = inv.person_bbox.map((v) => v / s);
      const pcx = (px1 + px2) / 2;
      const pcy = (py1 + py2) / 2;

      /* Person box */
      ctx.strokeStyle = color;
      ctx.lineWidth = lineW + 1;
      ctx.strokeRect(px1, py1, px2 - px1, py2 - py1);

      /* Person label */
      const pLabel = `Person #${pid} (${inv.num_items} items)`;
      const ptw = ctx.measureText(pLabel).width;
      ctx.fillStyle = color;
      ctx.globalAlpha = 0.85;
      ctx.fillRect(px1, py1 - fontSize - 10, ptw + 12, fontSize + 8);
      ctx.globalAlpha = 1.0;
      ctx.fillStyle = "#fff";
      ctx.fillText(pLabel, px1 + 6, py1 - 6);

      /* Ownership lines + object boxes */
      for (const item of inv.items) {
        const [ox1, oy1, ox2, oy2] = item.bbox.map((v) => v / s);
        const ocx = (ox1 + ox2) / 2;
        const ocy = (oy1 + oy2) / 2;

        /* Dashed line */
        ctx.setLineDash([8, 6]);
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(pcx, pcy);
        ctx.lineTo(ocx, ocy);
        ctx.stroke();
        ctx.setLineDash([]);

        /* Object box */
        ctx.strokeStyle = color;
        ctx.lineWidth = lineW;
        ctx.strokeRect(ox1, oy1, ox2 - ox1, oy2 - oy1);

        /* Object label */
        const oLabel = `${item.class_name} ${(item.confidence * 100).toFixed(0)}%`;
        const otw = ctx.measureText(oLabel).width;
        ctx.fillStyle = color;
        ctx.globalAlpha = 0.8;
        ctx.fillRect(ox1, oy1 - fontSize - 8, otw + 10, fontSize + 6);
        ctx.globalAlpha = 1.0;
        ctx.fillStyle = "#fff";
        ctx.fillText(oLabel, ox1 + 5, oy1 - 5);
      }
    }

    /* Unattended items — pulsing red */
    const pulse = Math.abs(Math.sin(Date.now() / 300)) * 0.5 + 0.5;
    for (const obj of data.unowned_objects) {
      const [x1, y1, x2, y2] = obj.bbox.map((v) => v / s);
      ctx.strokeStyle = `rgba(255, 0, 0, ${0.5 + pulse * 0.5})`;
      ctx.lineWidth = lineW + 1;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      const uLabel = `⚠ ${obj.class_name}`;
      ctx.fillStyle = "#dc2626";
      ctx.fillText(uLabel, x1, y1 - 6);
    }
  };

  /* ── 60fps display loop ── */
  const displayLoop = useCallback(() => {
    const video = videoRef.current;
    const display = displayRef.current;
    if (!video || !display || video.readyState < 2) {
      if (runningRef.current) animRef.current = requestAnimationFrame(displayLoop);
      return;
    }
    if (display.width !== video.videoWidth) display.width = video.videoWidth;
    if (display.height !== video.videoHeight) display.height = video.videoHeight;
    const ctx = display.getContext("2d")!;
    ctx.drawImage(video, 0, 0);
    drawOverlay(ctx, ssDataRef.current, video.videoWidth);
    if (runningRef.current) animRef.current = requestAnimationFrame(displayLoop);
  }, []);

  /* ── Background detection loop ── */
  const detectionLoop = useCallback(async () => {
    if (!runningRef.current) return;
    if (busyRef.current) { setTimeout(detectionLoop, 50); return; }
    const video = videoRef.current;
    const capture = captureRef.current;
    if (!video || !capture || video.readyState < 2) {
      if (runningRef.current) setTimeout(detectionLoop, 100);
      return;
    }
    if (video.paused && !video.ended) {
      if (runningRef.current) setTimeout(detectionLoop, 200);
      return;
    }
    if (video.ended) return;

    busyRef.current = true;
    const t0 = performance.now();
    try {
      const MAX_W = 640;
      const scale = Math.min(1, MAX_W / video.videoWidth);
      captureScaleRef.current = scale;
      capture.width = Math.round(video.videoWidth * scale);
      capture.height = Math.round(video.videoHeight * scale);
      capture.getContext("2d")!.drawImage(video, 0, 0, capture.width, capture.height);
      const blob = await new Promise<Blob>((r) =>
        capture.toBlob((b) => r(b!), "image/jpeg", 0.7)
      );
      const form = new FormData();
      form.append("file", blob, "frame.jpg");
      const res = await fetch(`${API}/study-space/detect-frame`, {
        method: "POST",
        body: form,
      });
      if (res.ok) {
        const infMs = parseFloat(res.headers.get("X-Inference-Ms") || "0");
        setInferenceMs(infMs);
        setTotalFrames((p) => p + 1);
        const rawData = res.headers.get("X-Study-Space-Data");
        if (rawData) {
          try {
            const parsed: StudySpaceData = JSON.parse(rawData);
            setSsData(parsed);
            ssDataRef.current = parsed;
          } catch {}
        }
        setFps(Math.round(1000 / (performance.now() - t0)));
      }
    } catch (err: any) {
      if (err.message === "Failed to fetch") {
        setError("Backend not reachable. Is the API server running?");
        runningRef.current = false;
        setActive(false);
        busyRef.current = false;
        return;
      }
    }
    busyRef.current = false;
    if (runningRef.current) setTimeout(detectionLoop, 0);
  }, []);

  /* ── Start camera ── */
  const startCamera = useCallback(async () => {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "environment" },
        audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      setMode("camera");
      setActive(true);
      runningRef.current = true;
      busyRef.current = false;
      ssDataRef.current = null;
      animRef.current = requestAnimationFrame(displayLoop);
      setTimeout(detectionLoop, 200);
    } catch {
      setError("Camera access denied.");
    }
  }, [displayLoop, detectionLoop]);

  /* ── Start video file ── */
  const startVideo = useCallback(() => {
    const video = videoRef.current;
    if (!video || !videoUrl) return;
    setError(null);
    setVideoEnded(false);
    setTotalFrames(0);
    ssDataRef.current = null;
    setSsData(null);
    busyRef.current = false;
    video.currentTime = 0;
    video.play();
    setMode("video");
    setActive(true);
    runningRef.current = true;
    animRef.current = requestAnimationFrame(displayLoop);
    setTimeout(detectionLoop, 200);
  }, [videoUrl, displayLoop, detectionLoop]);

  /* ── Stop ── */
  const stop = useCallback(() => {
    runningRef.current = false;
    cancelAnimationFrame(animRef.current);
    setActive(false);
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    videoRef.current?.pause();
  }, []);

  /* ── File handler ── */
  const handleFile = useCallback((f: File) => {
    if (!f.type.startsWith("video/")) {
      setError("Please upload a video file");
      return;
    }
    stop();
    setFile(f);
    setVideoUrl(URL.createObjectURL(f));
    setVideoEnded(false);
    setSsData(null);
    ssDataRef.current = null;
    setError(null);
  }, [stop]);

  useEffect(() => () => {
    runningRef.current = false;
    cancelAnimationFrame(animRef.current);
    if (streamRef.current) streamRef.current.getTracks().forEach((t) => t.stop());
  }, []);

  /* ── Derived stats ── */
  const numPersons = ssData?.num_persons ?? 0;
  const numObjects = ssData?.num_objects ?? 0;
  const numUnattended = ssData?.unowned_objects?.length ?? 0;
  const numAlerts = ssData?.active_alerts?.length ?? 0;
  const zoneInfo = ssData?.zones;

  return (
    <main className="min-h-screen">
      {/* Nav */}
      <nav className="fixed top-0 w-full z-50 bg-white/80 backdrop-blur-md border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
          <Link href="/" className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-cyan-600 to-violet-600 flex items-center justify-center text-white font-bold text-sm">Y</div>
            <span className="font-semibold text-gray-900 text-lg">YOLO<span className="text-cyan-600">Detect</span></span>
          </Link>
          <div className="flex items-center gap-4">
            <Link href="/demo" className="text-sm text-gray-500 hover:text-gray-900 transition">Video Demo</Link>
            <Link href="/camera" className="text-sm text-gray-500 hover:text-gray-900 transition">Camera</Link>
            <Link href="/" className="text-sm text-gray-500 hover:text-gray-900 transition">&larr; Home</Link>
          </div>
        </div>
      </nav>

      <div className="pt-24 pb-16 px-6 max-w-7xl mx-auto">
        {/* Header */}
        <div className="flex flex-col md:flex-row items-start md:items-center justify-between mb-8 gap-4">
          <div>
            <h1 className="text-3xl md:text-4xl font-bold text-gray-900 mb-2">
              Smart <span className="gradient-text">Study Space</span> Monitor
            </h1>
            <p className="text-gray-500">
              Real-time person-object ownership, seat occupancy, and unattended item detection.
            </p>
          </div>
          <div className="flex gap-3">
            {!active ? (
              <>
                <button
                  onClick={startCamera}
                  className="px-5 py-2.5 rounded-xl bg-gradient-to-r from-cyan-600 to-violet-600 text-white font-semibold hover:shadow-lg transition-all text-sm"
                >
                  &#128247; Live Camera
                </button>
                <button
                  onClick={() => inputRef.current?.click()}
                  className="px-5 py-2.5 rounded-xl border-2 border-gray-200 text-gray-700 font-semibold hover:border-cyan-300 transition-all text-sm"
                >
                  &#128193; Upload Video
                </button>
                <input
                  ref={inputRef}
                  type="file"
                  accept="video/*"
                  className="hidden"
                  onChange={(e) => { const f = e.target.files?.[0]; if (f) handleFile(f); }}
                />
              </>
            ) : (
              <button
                onClick={stop}
                className="px-5 py-2.5 rounded-xl bg-red-500 text-white font-semibold hover:bg-red-600 transition-all text-sm"
              >
                &#9209; Stop
              </button>
            )}
          </div>
        </div>

        {/* Start video button if file loaded but not running */}
        {videoUrl && !active && !videoEnded && (
          <div className="mb-6">
            <button onClick={startVideo}
              className="px-6 py-3 rounded-xl bg-gradient-to-r from-cyan-600 to-violet-600 text-white font-semibold hover:shadow-lg transition-all">
              &#9654; Start Detection on {file?.name}
            </button>
          </div>
        )}

        {error && (
          <div className="rounded-xl p-4 border border-red-200 bg-red-50 mb-6">
            <p className="text-red-600 text-sm">&#9888; {error}</p>
          </div>
        )}

        <div className="grid lg:grid-cols-3 gap-6">
          {/* ── Video panel ── */}
          <div className="lg:col-span-2">
            <div className="glass rounded-2xl overflow-hidden">
              <div className="px-5 py-3 border-b border-gray-100 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className={`w-2.5 h-2.5 rounded-full ${active ? "bg-red-500 animate-pulse" : videoEnded ? "bg-green-500" : "bg-gray-300"}`} />
                  <span className="text-sm font-medium text-gray-900">
                    {active ? "Analyzing..." : videoEnded ? "Complete" : "Ready"}
                  </span>
                </div>
                {(active || videoEnded) && (
                  <div className="flex items-center gap-4 text-xs font-mono">
                    <span className="text-cyan-600">{fps} FPS</span>
                    <span className="text-gray-400">|</span>
                    <span className="text-violet-600">{inferenceMs.toFixed(0)}ms</span>
                    <span className="text-gray-400">|</span>
                    <span className="text-gray-600">{numPersons} persons</span>
                  </div>
                )}
              </div>

              <div className="relative bg-gray-50 aspect-video flex items-center justify-center">
                {videoUrl && <video ref={videoRef} src={videoUrl} className="hidden" playsInline muted
                  onEnded={() => { runningRef.current = false; setActive(false); setVideoEnded(true); }} />}
                {!videoUrl && <video ref={videoRef} className="hidden" playsInline muted />}
                <canvas ref={captureRef} className="hidden" />

                {active || videoEnded ? (
                  <canvas ref={displayRef} className="w-full h-full" style={{ objectFit: "contain" }} />
                ) : (
                  <div className="text-center p-8">
                    <div className="text-6xl mb-4 opacity-20">&#128218;</div>
                    <p className="text-gray-400 text-lg">Smart Study Space Monitor</p>
                    <p className="text-gray-300 text-sm mt-2">Use camera or upload a video to start</p>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* ── Side panel ── */}
          <div className="space-y-5">
            {/* Stats cards */}
            <div className="grid grid-cols-2 gap-3">
              <div className="glass rounded-xl p-4 text-center">
                <div className="text-2xl font-bold text-cyan-600">{numPersons}</div>
                <div className="text-xs text-gray-500 mt-1">People</div>
              </div>
              <div className="glass rounded-xl p-4 text-center">
                <div className="text-2xl font-bold text-violet-600">{numObjects}</div>
                <div className="text-xs text-gray-500 mt-1">Objects</div>
              </div>
              <div className="glass rounded-xl p-4 text-center">
                <div className={`text-2xl font-bold ${numUnattended > 0 ? "text-red-500" : "text-green-500"}`}>{numUnattended}</div>
                <div className="text-xs text-gray-500 mt-1">Unattended</div>
              </div>
              <div className="glass rounded-xl p-4 text-center">
                <div className="text-2xl font-bold text-gray-700">{totalFrames}</div>
                <div className="text-xs text-gray-500 mt-1">Frames</div>
              </div>
            </div>

            {/* Zone occupancy */}
            {zoneInfo && zoneInfo.total > 0 && (
              <div className="glass rounded-xl p-5">
                <h3 className="font-semibold text-gray-900 mb-3 text-sm">&#128186; Seat Occupancy</h3>
                <div className="flex items-center gap-2 mb-3">
                  <div className="flex-1 h-3 bg-gray-100 rounded-full overflow-hidden">
                    <div
                      className="h-full rounded-full progress-bar transition-all"
                      style={{ width: `${(zoneInfo.occupied / Math.max(zoneInfo.total, 1)) * 100}%` }}
                    />
                  </div>
                  <span className="text-xs font-mono text-gray-600">{zoneInfo.occupied}/{zoneInfo.total}</span>
                </div>
                <div className="space-y-1.5">
                  {zoneInfo.zones.map((z) => (
                    <div key={z.zone_id} className="flex items-center justify-between text-xs">
                      <div className="flex items-center gap-2">
                        <span className={`w-2 h-2 rounded-full ${z.occupied ? "bg-red-400" : "bg-green-400"}`} />
                        <span className="text-gray-700">{z.name}</span>
                      </div>
                      <span className={`font-medium ${z.occupied ? "text-red-500" : "text-green-500"}`}>
                        {z.occupied ? "Occupied" : "Vacant"}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Person inventories */}
            {ssData && ssData.inventories.length > 0 && (
              <div className="glass rounded-xl p-5">
                <h3 className="font-semibold text-gray-900 mb-3 text-sm">&#128100; Person Inventory</h3>
                <div className="space-y-3 max-h-64 overflow-y-auto">
                  {ssData.inventories.map((inv, i) => {
                    const color = PERSON_COLORS[(inv.track_id ?? i) % PERSON_COLORS.length];
                    return (
                      <div key={inv.track_id ?? i} className="border border-gray-100 rounded-lg p-3">
                        <div className="flex items-center gap-2 mb-2">
                          <div className="w-3 h-3 rounded-full" style={{ backgroundColor: color }} />
                          <span className="text-sm font-medium text-gray-800">Person #{inv.track_id ?? "?"}</span>
                          <span className="ml-auto text-xs text-gray-400">{inv.num_items} items</span>
                        </div>
                        {inv.items.length > 0 ? (
                          <div className="flex flex-wrap gap-1.5">
                            {inv.items.map((item, j) => (
                              <span key={j} className="px-2 py-0.5 rounded-full text-xs font-medium"
                                style={{ backgroundColor: `${color}18`, color, border: `1px solid ${color}40` }}>
                                {item.class_name}
                              </span>
                            ))}
                          </div>
                        ) : (
                          <p className="text-xs text-gray-400 italic">No items nearby</p>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {/* Alerts */}
            {ssData && ssData.active_alerts.length > 0 && (
              <div className="glass rounded-xl p-5 border-l-4 border-red-400">
                <h3 className="font-semibold text-red-600 mb-3 text-sm">&#9888;&#65039; Alerts</h3>
                <div className="space-y-2 max-h-48 overflow-y-auto">
                  {ssData.active_alerts.slice(-5).reverse().map((a) => (
                    <div key={a.alert_id} className="text-xs p-2 rounded-lg bg-red-50 border border-red-100">
                      <span className={`font-bold ${a.severity === "critical" ? "text-red-700" : "text-orange-600"}`}>
                        {a.severity === "critical" ? "!!" : "!"}
                      </span>
                      {" "}{a.message}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </main>
  );
}
