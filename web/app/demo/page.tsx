"use client";

import Link from "next/link";
import { useState, useRef, useCallback, useEffect } from "react";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

type Detection = {
  bbox: number[];
  confidence: number;
  class_id: number;
  class_name: string;
  track_id: number | null;
};

const BOX_COLORS = [
  "#0891b2", "#7c3aed", "#dc2626", "#ea580c", "#16a34a",
  "#2563eb", "#c026d3", "#ca8a04", "#0d9488", "#4f46e5",
];

export default function DemoPage() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const captureRef = useRef<HTMLCanvasElement>(null);
  const displayRef = useRef<HTMLCanvasElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const runningRef = useRef(false);
  const busyRef = useRef(false);
  const detectionsRef = useRef<Detection[]>([]);
  const animRef = useRef(0);

  const [file, setFile] = useState<File | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [detecting, setDetecting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [fps, setFps] = useState(0);
  const [inferenceMs, setInferenceMs] = useState(0);
  const [detections, setDetections] = useState<Detection[]>([]);
  const [detCount, setDetCount] = useState(0);
  const [totalFrames, setTotalFrames] = useState(0);
  const [totalDetections, setTotalDetections] = useState(0);
  const [videoEnded, setVideoEnded] = useState(false);
  const captureScaleRef = useRef(1);

  /* -- Draw detection boxes on canvas -- */
  const drawBoxes = (ctx: CanvasRenderingContext2D, dets: Detection[], vw: number) => {
    const s = captureScaleRef.current;
    const lineW = Math.max(2, Math.round(vw / 350));
    const fontSize = Math.max(12, Math.round(vw / 50));
    ctx.font = `bold ${fontSize}px Inter, system-ui, sans-serif`;
    for (const det of dets) {
      const [x1, y1, x2, y2] = det.bbox.map((v) => v / s);
      const color = BOX_COLORS[det.class_id % BOX_COLORS.length];
      ctx.strokeStyle = color;
      ctx.lineWidth = lineW;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      const label = `${det.class_name} ${(det.confidence * 100).toFixed(0)}%`;
      const tw = ctx.measureText(label).width;
      const lh = fontSize + 8;
      ctx.fillStyle = color;
      ctx.globalAlpha = 0.85;
      ctx.fillRect(x1, y1 - lh, tw + 10, lh);
      ctx.globalAlpha = 1.0;
      ctx.fillStyle = "#fff";
      ctx.fillText(label, x1 + 5, y1 - 6);
    }
  };

  /* -- File selection -- */
  const handleFile = useCallback((f: File) => {
    if (!f.type.startsWith("video/")) {
      setError("Please upload a video file (MP4, AVI, MOV, WebM)");
      return;
    }
    if (f.size > 200 * 1024 * 1024) {
      setError("File too large. Maximum 200MB.");
      return;
    }
    runningRef.current = false;
    cancelAnimationFrame(animRef.current);
    setDetecting(false);
    setVideoEnded(false);
    setTotalFrames(0);
    setTotalDetections(0);
    setDetections([]);
    detectionsRef.current = [];
    setDetCount(0);
    setFps(0);
    setInferenceMs(0);
    setFile(f);
    setVideoUrl(URL.createObjectURL(f));
    setError(null);
  }, []);

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      const f = e.dataTransfer.files[0];
      if (f) handleFile(f);
    },
    [handleFile]
  );

  /* -- 60fps display loop: draws video frame + detection boxes -- */
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
    drawBoxes(ctx, detectionsRef.current, video.videoWidth);
    if (runningRef.current) {
      animRef.current = requestAnimationFrame(displayLoop);
    }
  }, []);

  /* -- Background detection loop: sends frames, stores results -- */
  const detectionLoop = useCallback(async () => {
    if (!runningRef.current) return;
    if (busyRef.current) {
      setTimeout(detectionLoop, 50);
      return;
    }
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
      const res = await fetch(`${API}/detect-frame`, {
        method: "POST",
        body: form,
      });
      if (res.ok) {
        const infMs = parseFloat(res.headers.get("X-Inference-Ms") || "0");
        const nDet = parseInt(res.headers.get("X-Detections") || "0", 10);
        const detData = res.headers.get("X-Detection-Data");
        setInferenceMs(infMs);
        setDetCount(nDet);
        setTotalFrames((p) => p + 1);
        setTotalDetections((p) => p + nDet);
        if (detData) {
          try {
            const parsed = JSON.parse(detData);
            const dets = parsed.detections || [];
            setDetections(dets);
            detectionsRef.current = dets;
          } catch {}
        }
        setFps(Math.round(1000 / (performance.now() - t0)));
      }
    } catch (err: any) {
      if (err.message === "Failed to fetch") {
        setError("Backend not reachable. Is the API server running?");
        runningRef.current = false;
        setDetecting(false);
        busyRef.current = false;
        return;
      }
    }
    busyRef.current = false;
    if (runningRef.current) setTimeout(detectionLoop, 0);
  }, []);

  /* -- Start / Stop / Reset -- */
  const startDetection = useCallback(() => {
    const video = videoRef.current;
    if (!video || !videoUrl) return;
    setError(null);
    setVideoEnded(false);
    setTotalFrames(0);
    setTotalDetections(0);
    setDetections([]);
    detectionsRef.current = [];
    busyRef.current = false;
    video.currentTime = 0;
    video.play();
    runningRef.current = true;
    setDetecting(true);
    animRef.current = requestAnimationFrame(displayLoop);
    setTimeout(detectionLoop, 200);
  }, [videoUrl, displayLoop, detectionLoop]);

  const stopDetection = useCallback(() => {
    runningRef.current = false;
    cancelAnimationFrame(animRef.current);
    setDetecting(false);
    videoRef.current?.pause();
  }, []);

  const handleReset = () => {
    runningRef.current = false;
    cancelAnimationFrame(animRef.current);
    setDetecting(false);
    setFile(null);
    if (videoUrl) URL.revokeObjectURL(videoUrl);
    setVideoUrl(null);
    setError(null);
    setVideoEnded(false);
    setTotalFrames(0);
    setTotalDetections(0);
    setDetections([]);
    detectionsRef.current = [];
    setDetCount(0);
    setFps(0);
    setInferenceMs(0);
    if (inputRef.current) inputRef.current.value = "";
  };

  useEffect(
    () => () => {
      runningRef.current = false;
      cancelAnimationFrame(animRef.current);
    },
    []
  );

  /* -- Class summary -- */
  const classCounts: Record<string, number> = {};
  detections.forEach((d) => {
    classCounts[d.class_name] = (classCounts[d.class_name] || 0) + 1;
  });
  const sortedClasses = Object.entries(classCounts).sort((a, b) => b[1] - a[1]);

  return (
    <main className="min-h-screen">
      <nav className="fixed top-0 w-full z-50 bg-white/80 backdrop-blur-md border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
          <Link href="/" className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-cyan-600 to-violet-600 flex items-center justify-center text-white font-bold text-sm">
              Y
            </div>
            <span className="font-semibold text-gray-900 text-lg">
              YOLO<span className="text-cyan-600">Detect</span>
            </span>
          </Link>
          <div className="flex items-center gap-4">
            <Link
              href="/camera"
              className="text-sm text-gray-500 hover:text-gray-900 transition"
            >
              Live Camera
            </Link>
            <Link
              href="/"
              className="text-sm text-gray-500 hover:text-gray-900 transition"
            >
              &larr; Home
            </Link>
          </div>
        </div>
      </nav>

      <div className="pt-24 pb-16 px-6 max-w-6xl mx-auto">
        <h1 className="text-3xl md:text-4xl font-bold text-gray-900 mb-2">
          Video <span className="gradient-text">Object Detection</span>
        </h1>
        <p className="text-gray-500 mb-10">
          Upload a video and watch YOLO26 detect objects in real-time &mdash;
          video plays at full speed while detection runs in the background.
        </p>

        <div className="grid lg:grid-cols-3 gap-6">
          {/* -- Video display -- */}
          <div className="lg:col-span-2">
            <div className="glass rounded-2xl overflow-hidden">
              <div className="px-5 py-3 border-b border-gray-100 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span
                    className={`w-2.5 h-2.5 rounded-full ${
                      detecting
                        ? "bg-red-500 animate-pulse"
                        : videoEnded
                        ? "bg-green-500"
                        : "bg-gray-300"
                    }`}
                  />
                  <span className="text-sm font-medium text-gray-900">
                    {detecting
                      ? "Detecting..."
                      : videoEnded
                      ? "Detection Complete"
                      : "Ready"}
                  </span>
                </div>
                {(detecting || videoEnded) && (
                  <div className="flex items-center gap-4 text-xs font-mono">
                    <span className="text-cyan-600">{fps} FPS</span>
                    <span className="text-gray-400">|</span>
                    <span className="text-violet-600">{inferenceMs}ms</span>
                    <span className="text-gray-400">|</span>
                    <span className="text-gray-600">{detCount} objects</span>
                  </div>
                )}
              </div>

              <div className="relative bg-gray-50 aspect-video flex items-center justify-center">
                {/* Hidden video for playback + hidden canvas for frame capture */}
                {videoUrl && (
                  <video
                    ref={videoRef}
                    src={videoUrl}
                    className="hidden"
                    playsInline
                    muted
                    onEnded={() => {
                      runningRef.current = false;
                      setDetecting(false);
                      setVideoEnded(true);
                    }}
                  />
                )}
                <canvas ref={captureRef} className="hidden" />

                {/* Display canvas: video frame + detection boxes at 60fps */}
                {detecting || videoEnded ? (
                  <canvas
                    ref={displayRef}
                    className="w-full h-full"
                    style={{ objectFit: "contain" }}
                  />
                ) : videoUrl ? (
                  <video
                    controls
                    className="w-full h-full"
                    style={{ objectFit: "contain" }}
                    src={videoUrl}
                  />
                ) : (
                  <div className="text-center p-8">
                    <div className="text-6xl mb-4 opacity-20">&#127916;</div>
                    <p className="text-gray-400 text-lg">
                      Upload a video to get started
                    </p>
                    <p className="text-gray-300 text-sm mt-2">
                      Real-time detection results will appear here
                    </p>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* -- Side panel -- */}
          <div className="space-y-5">
            {/* Upload zone */}
            <div
              className={`upload-zone bg-white rounded-2xl p-6 border-2 border-dashed transition-all cursor-pointer ${
                dragOver
                  ? "drag-over border-cyan-400"
                  : "border-gray-300 hover:border-cyan-400"
              }`}
              onDragOver={(e) => {
                e.preventDefault();
                setDragOver(true);
              }}
              onDragLeave={() => setDragOver(false)}
              onDrop={onDrop}
              onClick={() => inputRef.current?.click()}
            >
              <input
                ref={inputRef}
                type="file"
                accept="video/*"
                className="hidden"
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f) handleFile(f);
                }}
              />
              {!file ? (
                <div className="text-center">
                  <div className="text-4xl mb-3">&#128193;</div>
                  <p className="text-gray-900 font-medium mb-1 text-sm">
                    Drag &amp; drop video
                  </p>
                  <p className="text-xs text-gray-400">
                    MP4, AVI, MOV &middot; max 200MB
                  </p>
                </div>
              ) : (
                <div className="text-center">
                  <div className="text-2xl mb-2">&#9989;</div>
                  <p className="text-gray-900 font-medium text-sm truncate max-w-full">
                    {file.name}
                  </p>
                  <p className="text-xs text-gray-400 mt-1">
                    {(file.size / (1024 * 1024)).toFixed(1)} MB
                  </p>
                </div>
              )}
            </div>

            {/* Actions */}
            <div className="flex gap-3">
              {!detecting ? (
                <button
                  onClick={startDetection}
                  disabled={!file}
                  className="flex-1 py-3 rounded-xl bg-gradient-to-r from-cyan-600 to-violet-600 text-white font-semibold disabled:opacity-40 disabled:cursor-not-allowed hover:shadow-lg hover:shadow-cyan-500/25 transition-all text-sm"
                >
                  &#128640; Start Detection
                </button>
              ) : (
                <button
                  onClick={stopDetection}
                  className="flex-1 py-3 rounded-xl bg-red-500 hover:bg-red-600 text-white font-semibold transition-all text-sm"
                >
                  &#9209; Stop
                </button>
              )}
              {file && !detecting && (
                <button
                  onClick={handleReset}
                  className="px-4 py-3 rounded-xl border border-gray-200 text-gray-600 hover:bg-gray-50 transition text-sm"
                >
                  Reset
                </button>
              )}
            </div>

            {/* Error */}
            {error && (
              <div className="rounded-xl p-4 border border-red-200 bg-red-50">
                <p className="text-red-600 text-sm">&#9888; {error}</p>
              </div>
            )}

            {/* Stats */}
            <div className="glass rounded-xl p-5">
              <h3 className="text-sm font-semibold text-gray-900 mb-4">
                Live Statistics
              </h3>
              <div className="grid grid-cols-2 gap-3">
                <MiniStat
                  label="Detection FPS"
                  value={String(fps)}
                  color="text-cyan-600"
                />
                <MiniStat
                  label="Inference"
                  value={`${inferenceMs}ms`}
                  color="text-violet-600"
                />
                <MiniStat
                  label="Objects"
                  value={String(detCount)}
                  color="text-orange-500"
                />
                <MiniStat
                  label="Frames Analyzed"
                  value={totalFrames.toLocaleString()}
                  color="text-gray-600"
                />
              </div>
              <div className="mt-3 pt-3 border-t border-gray-100">
                <div className="flex justify-between text-xs">
                  <span className="text-gray-400">Total Detections</span>
                  <span className="text-cyan-600 font-mono font-medium">
                    {totalDetections.toLocaleString()}
                  </span>
                </div>
              </div>
            </div>

            {/* Current detections */}
            <div className="glass rounded-xl p-5">
              <h3 className="text-sm font-semibold text-gray-900 mb-3">
                Current Detections
              </h3>
              {sortedClasses.length === 0 ? (
                <p className="text-xs text-gray-400">No objects detected</p>
              ) : (
                <div className="space-y-2 max-h-60 overflow-y-auto">
                  {sortedClasses.map(([cls, count]) => (
                    <div
                      key={cls}
                      className="flex items-center justify-between"
                    >
                      <div className="flex items-center gap-2">
                        <span className="w-2 h-2 rounded-full bg-gradient-to-r from-cyan-500 to-violet-500" />
                        <span className="text-sm text-gray-700">{cls}</span>
                      </div>
                      <span className="text-sm font-mono text-cyan-600 font-medium">
                        {count}
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Detection details */}
            <div className="glass rounded-xl p-5">
              <h3 className="text-sm font-semibold text-gray-900 mb-3">
                Detection Details
              </h3>
              <div className="space-y-1.5 max-h-48 overflow-y-auto">
                {detections.length === 0 ? (
                  <p className="text-xs text-gray-400">&mdash;</p>
                ) : (
                  detections.slice(0, 15).map((d, i) => (
                    <div
                      key={i}
                      className="flex items-center justify-between text-xs py-1 border-b border-gray-50"
                    >
                      <span className="text-gray-700 font-medium">
                        {d.class_name}
                      </span>
                      <span className="text-gray-400 font-mono">
                        {(d.confidence * 100).toFixed(0)}%
                      </span>
                    </div>
                  ))
                )}
              </div>
            </div>

            <div className="glass rounded-xl p-4">
              <p className="text-xs text-gray-400">
                <span className="text-cyan-600">Backend:</span>{" "}
                <code className="font-mono text-gray-500">{API}</code>
              </p>
              <p className="text-xs text-gray-400 mt-1">
                Video plays at full speed &bull; detection runs in background
              </p>
            </div>
          </div>
        </div>
      </div>
    </main>
  );
}

function MiniStat({
  label,
  value,
  color,
}: {
  label: string;
  value: string;
  color: string;
}) {
  return (
    <div className="bg-gray-50 rounded-lg p-3 text-center">
      <p className="text-[10px] text-gray-400 uppercase tracking-wide">
        {label}
      </p>
      <p className={`text-lg font-bold ${color}`}>{value}</p>
    </div>
  );
}
