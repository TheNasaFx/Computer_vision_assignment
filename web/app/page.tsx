import Link from "next/link";

/* ── Stats ────────────────────────────────────────────────── */
const stats = [
  { value: "80+", label: "Object Classes" },
  { value: "<30ms", label: "Inference Time" },
  { value: "30+", label: "FPS Real-Time" },
  { value: "Smart", label: "Study Space" },
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
            <Link
              href="/study-space"
              className="px-4 py-2 rounded-lg border border-cyan-300 text-cyan-700 text-sm font-medium hover:bg-cyan-50 transition"
            >
              &#128218; Study Space
            </Link>
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
            Smart Study Space Monitor &mdash; detect, classify, and associate
            objects with nearby people in real-time. Track seat occupancy and
            unattended items using state-of-the-art YOLO26 deep learning.
          </p>

          <div className="flex flex-col sm:flex-row gap-4 justify-center flex-wrap">
            <Link
              href="/study-space"
              className="px-8 py-3.5 rounded-xl bg-gradient-to-r from-cyan-600 to-violet-600 text-white font-semibold text-lg hover:shadow-lg hover:shadow-cyan-500/25 transition-all"
            >
              &#128218; Smart Study Space →
            </Link>
            <Link
              href="/demo"
              className="px-8 py-3.5 rounded-xl border-2 border-gray-200 text-gray-700 font-semibold text-lg hover:border-cyan-300 hover:bg-cyan-50/50 transition-all"
            >
              Upload &amp; Detect
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
