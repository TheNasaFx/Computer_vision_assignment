import Link from "next/link";

/* ── Feature data ────────────────────────────────────────── */
const features = [
  {
    icon: "⚡",
    title: "Real-Time Inference",
    desc: "YOLO26 with CUDA + FP16 delivers <30ms latency per frame on modern GPUs.",
  },
  {
    icon: "🎯",
    title: "Multi-Object Tracking",
    desc: "ByteTrack / BoTSORT assigns persistent IDs and draws motion trails.",
  },
  {
    icon: "📹",
    title: "Any Video Source",
    desc: "Webcam, video files, RTSP streams, or multi-camera setups — all supported.",
  },
  {
    icon: "🔄",
    title: "Hot-Swap Models",
    desc: "Switch between YOLO26 nano/small/medium/large at runtime via API.",
  },
  {
    icon: "📊",
    title: "Event Detection",
    desc: "ROI zone enter/exit and line-crossing events with configurable regions.",
  },
  {
    icon: "🌐",
    title: "REST API",
    desc: "FastAPI backend with image upload, video processing, and stream control.",
  },
];

const stats = [
  { value: "80+", label: "Object Classes" },
  { value: "<30ms", label: "Inference Time" },
  { value: "30+", label: "FPS Real-Time" },
  { value: "YOLO26", label: "Latest Model" },
];

/* ── Page ─────────────────────────────────────────────────── */
export default function Home() {
  return (
    <main className="min-h-screen">
      {/* ── Navigation ── */}
      <nav className="fixed top-0 w-full z-50 bg-white/80 backdrop-blur-md border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-cyan-600 to-violet-600 flex items-center justify-center text-white font-bold text-sm">
              Y
            </div>
            <span className="font-semibold text-gray-900 text-lg">
              YOLO<span className="text-cyan-600">Detect</span>
            </span>
          </div>
          <div className="flex items-center gap-6">
            <a href="#features" className="text-sm text-gray-500 hover:text-gray-900 transition">
              Features
            </a>
            <a href="#how" className="text-sm text-gray-500 hover:text-gray-900 transition">
              How It Works
            </a>
            <Link
              href="/camera"
              className="px-4 py-2 rounded-lg border border-gray-300 text-gray-700 text-sm font-medium hover:bg-gray-50 transition"
            >
              Live Camera
            </Link>
            <Link
              href="/demo"
              className="px-4 py-2 rounded-lg bg-gradient-to-r from-cyan-600 to-violet-600 text-white text-sm font-medium hover:opacity-90 transition"
            >
              Try Demo
            </Link>
          </div>
        </div>
      </nav>

      {/* ── Hero ── */}
      <section className="pt-32 pb-20 px-6">
        <div className="max-w-5xl mx-auto text-center">
          <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-cyan-50 border border-cyan-200 text-xs text-cyan-700 mb-8">
            <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
            Powered by YOLO26 &amp; Ultralytics
          </div>

          <h1 className="text-5xl md:text-7xl font-extrabold tracking-tight mb-6">
            <span className="text-gray-900">AI-Powered</span>
            <br />
            <span className="gradient-text">Object Detection</span>
          </h1>

          <p className="text-lg md:text-xl text-gray-500 max-w-2xl mx-auto mb-10 leading-relaxed">
            Upload any video and instantly detect, classify, and track objects
            using state-of-the-art deep learning — directly from your browser.
          </p>

          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Link
              href="/demo"
              className="px-8 py-3.5 rounded-xl bg-gradient-to-r from-cyan-600 to-violet-600 text-white font-semibold text-lg hover:shadow-lg hover:shadow-cyan-500/25 transition-all"
            >
              Upload &amp; Detect →
            </Link>
            <Link
              href="/camera"
              className="px-8 py-3.5 rounded-xl border-2 border-gray-200 text-gray-700 font-semibold text-lg hover:border-cyan-300 hover:bg-cyan-50/50 transition-all"
            >
              📷 Live Camera
            </Link>
            <a
              href="https://github.com/TheNasaFx/Computer_vision_assignment"
              target="_blank"
              rel="noopener noreferrer"
              className="px-8 py-3.5 rounded-xl border-2 border-gray-200 text-gray-700 font-semibold text-lg hover:bg-gray-50 transition-all"
            >
              View on GitHub
            </a>
          </div>
        </div>

        {/* Stats */}
        <div className="max-w-4xl mx-auto mt-20 grid grid-cols-2 md:grid-cols-4 gap-4">
          {stats.map((s) => (
            <div key={s.label} className="glass rounded-xl p-5 text-center stat-card">
              <div className="text-2xl md:text-3xl font-bold gradient-text">{s.value}</div>
              <div className="text-sm text-gray-500 mt-1">{s.label}</div>
            </div>
          ))}
        </div>
      </section>

      {/* ── Features ── */}
      <section id="features" className="py-20 px-6 bg-gray-50/50">
        <div className="max-w-6xl mx-auto">
          <h2 className="text-3xl md:text-4xl font-bold text-center text-gray-900 mb-4">
            Powerful Features
          </h2>
          <p className="text-gray-500 text-center mb-14 max-w-xl mx-auto">
            Production-grade detection pipeline with everything you need
          </p>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-5">
            {features.map((f) => (
              <div
                key={f.title}
                className="bg-white rounded-2xl p-6 border border-gray-200 hover:border-cyan-300 transition-all stat-card"
              >
                <div className="text-3xl mb-4">{f.icon}</div>
                <h3 className="text-lg font-semibold text-gray-900 mb-2">{f.title}</h3>
                <p className="text-sm text-gray-500 leading-relaxed">{f.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── How It Works ── */}
      <section id="how" className="py-20 px-6">
        <div className="max-w-4xl mx-auto">
          <h2 className="text-3xl md:text-4xl font-bold text-center text-gray-900 mb-14">
            How It Works
          </h2>

          <div className="grid md:grid-cols-3 gap-8">
            {[
              {
                step: "01",
                title: "Upload Video",
                desc: "Drag & drop or browse to upload any video file (MP4, AVI, MOV).",
              },
              {
                step: "02",
                title: "AI Processing",
                desc: "YOLO26 processes every frame — detecting and tracking all objects.",
              },
              {
                step: "03",
                title: "View Results",
                desc: "Watch the annotated video with bounding boxes, labels, and statistics.",
              },
            ].map((item) => (
              <div key={item.step} className="text-center">
                <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-cyan-100 to-violet-100 border border-cyan-200 flex items-center justify-center mx-auto mb-5">
                  <span className="font-mono font-bold text-cyan-700">{item.step}</span>
                </div>
                <h3 className="text-lg font-semibold text-gray-900 mb-2">{item.title}</h3>
                <p className="text-sm text-gray-500">{item.desc}</p>
              </div>
            ))}
          </div>

          <div className="text-center mt-14">
            <Link
              href="/demo"
              className="px-8 py-3.5 rounded-xl bg-gradient-to-r from-cyan-600 to-violet-600 text-white font-semibold text-lg hover:shadow-lg hover:shadow-cyan-500/25 transition-all inline-block"
            >
              Try It Now →
            </Link>
          </div>
        </div>
      </section>

      {/* ── Architecture ── */}
      <section className="py-20 px-6 bg-gray-50/50">
        <div className="max-w-4xl mx-auto">
          <h2 className="text-3xl md:text-4xl font-bold text-center text-gray-900 mb-10">
            System Architecture
          </h2>
          <div className="bg-gray-900 rounded-2xl p-8 glow-cyan">
            <pre className="text-xs md:text-sm text-cyan-300 font-mono overflow-x-auto leading-relaxed">
{`┌──────────────────────────────────────────────────────────┐
│                    Web Frontend (Vercel)                  │
│  ┌─────────┐  ┌──────────────┐  ┌─────────────────────┐  │
│  │ Upload  │→ │  Processing  │→ │  Results + Video    │  │
│  │ Video   │  │  Progress    │  │  + Statistics       │  │
│  └─────────┘  └──────┬───────┘  └─────────────────────┘  │
└─────────────────────┬────────────────────────────────────┘
                      │ REST API
┌─────────────────────▼────────────────────────────────────┐
│                 FastAPI Backend (GPU Server)              │
│  ┌──────────┐  ┌───────────┐  ┌────────┐  ┌──────────┐  │
│  │ YOLO26   │  │ ByteTrack │  │ Events │  │ Video    │  │
│  │ Detector │→ │ Tracker   │→ │ Engine │→ │ Writer   │  │
│  └──────────┘  └───────────┘  └────────┘  └──────────┘  │
└──────────────────────────────────────────────────────────┘`}
            </pre>
          </div>
        </div>
      </section>

      {/* ── Footer ── */}
      <footer className="py-10 px-6 border-t border-gray-200">
        <div className="max-w-6xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4">
          <div className="text-sm text-gray-400">
            Built with YOLO26 · FastAPI · Next.js · OpenCV
          </div>
          <a
            href="https://github.com/TheNasaFx/Computer_vision_assignment"
            target="_blank"
            rel="noopener noreferrer"
            className="text-sm text-gray-500 hover:text-cyan-600 transition"
          >
            GitHub Repository →
          </a>
        </div>
      </footer>
    </main>
  );
}
