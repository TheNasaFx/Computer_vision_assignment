import Link from "next/link";

const stats = [
  { value: "60 FPS", label: "Render Loop" },
  { value: "+50ms",  label: "Forward Predict" },
  { value: "468",    label: "Face Landmarks" },
  { value: "0",      label: "Audio Files Shipped" },
];

const features = [
  {
    emoji: "🥁",
    title: "Predictive Drum Engine",
    body:
      "Velocity-reversal hit detection extrapolates the wrist 50 ms ahead so the drum sound fires at the predicted strike, not after the network round-trip.",
  },
  {
    emoji: "🕶️",
    title: "Browser-Side AR Filters",
    body:
      "468-point MediaPipe FaceMesh runs entirely in the browser. Sunglasses, hats, mustaches, and dog filters track the face with sub-frame latency.",
  },
  {
    emoji: "🎛️",
    title: "Synthesised Audio",
    body:
      "Every drum voice is built from primitive Web Audio nodes — no WAV files, no asset budget, perfect crispness at any volume.",
  },
];

export default function Home() {
  return (
    <main className="min-h-screen">
      {/* Nav */}
      <nav className="fixed top-0 w-full z-50 bg-white/80 backdrop-blur-md border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-fuchsia-500 to-cyan-500 flex items-center justify-center text-white font-bold text-sm">
              M
            </div>
            <span className="font-semibold text-gray-900 text-lg">
              AI <span className="text-fuchsia-600">Magic Mirror</span>
            </span>
          </div>
          <div className="flex items-center gap-4">
            <Link
              href="/camera"
              className="hidden md:inline px-3 py-1.5 rounded-lg text-gray-500 text-sm hover:text-gray-900 transition"
            >
              Camera
            </Link>
            <Link
              href="/demo"
              className="hidden md:inline px-3 py-1.5 rounded-lg text-gray-500 text-sm hover:text-gray-900 transition"
            >
              Video Demo
            </Link>
            <Link
              href="/magic-mirror"
              className="px-4 py-2 rounded-lg bg-gradient-to-r from-fuchsia-600 to-cyan-600 text-white text-sm font-semibold shadow-sm hover:opacity-90 transition"
            >
              Launch Mirror →
            </Link>
          </div>
        </div>
      </nav>

      {/* Hero */}
      <section className="pt-32 pb-16 px-6">
        <div className="max-w-5xl mx-auto text-center">
          <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-fuchsia-50 border border-fuchsia-200 text-xs text-fuchsia-700 mb-8">
            <span className="w-2 h-2 rounded-full bg-fuchsia-500 animate-pulse" />
            Pose + FaceMesh · Real-time · Zero asset bundle
          </div>

          <h1 className="text-5xl md:text-7xl font-extrabold tracking-tight mb-6">
            <span className="text-gray-900">Play drums in the air.</span>
            <br />
            <span className="bg-gradient-to-r from-fuchsia-600 to-cyan-600 bg-clip-text text-transparent">
              Wear filters with your camera.
            </span>
          </h1>

          <p className="text-lg md:text-xl text-gray-500 max-w-2xl mx-auto mb-10 leading-relaxed">
            A pose-driven web mirror with two modes. <b className="text-gray-700">Drum</b> mode
            uses a predictive hit engine to compensate inference latency.{" "}
            <b className="text-gray-700">Filter</b> mode runs MediaPipe FaceMesh entirely in
            the browser for Snapchat-quality AR overlays.
          </p>

          <div className="flex flex-col sm:flex-row gap-4 justify-center flex-wrap">
            <Link
              href="/magic-mirror"
              className="px-8 py-3.5 rounded-xl bg-gradient-to-r from-fuchsia-600 to-cyan-600 text-white font-semibold text-lg hover:shadow-lg hover:shadow-fuchsia-500/25 transition-all"
            >
              🪞 Open the Magic Mirror →
            </Link>
            <Link
              href="/camera"
              className="px-8 py-3.5 rounded-xl border-2 border-gray-200 text-gray-700 font-semibold text-lg hover:border-fuchsia-300 hover:bg-fuchsia-50/50 transition-all"
            >
              📷 Object Detection Demo
            </Link>
          </div>
        </div>

        {/* Stats */}
        <div className="max-w-4xl mx-auto mt-20 grid grid-cols-2 md:grid-cols-4 gap-4">
          {stats.map((s) => (
            <div
              key={s.label}
              className="glass rounded-xl p-5 text-center stat-card"
            >
              <div className="text-2xl md:text-3xl font-bold gradient-text">
                {s.value}
              </div>
              <div className="text-sm text-gray-500 mt-1">{s.label}</div>
            </div>
          ))}
        </div>
      </section>

      {/* Features */}
      <section className="px-6 pb-24">
        <div className="max-w-5xl mx-auto grid md:grid-cols-3 gap-5">
          {features.map((f) => (
            <div
              key={f.title}
              className="glass rounded-2xl p-6 border border-gray-100"
            >
              <div className="text-3xl mb-3">{f.emoji}</div>
              <h3 className="font-bold text-gray-900 mb-2">{f.title}</h3>
              <p className="text-sm text-gray-500 leading-relaxed">{f.body}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Footer */}
      <footer className="py-10 px-6 border-t border-gray-200">
        <div className="max-w-6xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4">
          <div className="text-sm text-gray-400">
            Built with YOLO11-pose · FastAPI · Next.js · MediaPipe · Web Audio API
          </div>
          <div className="flex items-center gap-4 text-sm text-gray-500">
            <Link href="/magic-mirror" className="hover:text-fuchsia-600 transition">
              Magic Mirror
            </Link>
            <Link href="/study-space" className="hover:text-fuchsia-600 transition">
              Study Space (legacy)
            </Link>
          </div>
        </div>
      </footer>
    </main>
  );
}
