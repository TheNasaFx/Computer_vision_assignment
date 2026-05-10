/**
 * Install MediaPipe Tasks-Vision assets into public/mediapipe/tasks/.
 *
 * Why this exists
 * ---------------
 * The MediaPipe Tasks-Vision SDK loads two kinds of files at runtime:
 *
 *   1. WASM blobs that ship inside the npm package
 *      (`@mediapipe/tasks-vision/wasm/*`).
 *   2. `.task` model archives that are NOT in the npm package — Google
 *      hosts them on cloud storage. We download them once and serve them
 *      from our own origin so the page works offline-after-first-load
 *      and doesn't depend on a third-party CDN at runtime.
 *
 * Hooked into npm scripts as `postinstall`, `predev`, and `prebuild`, so
 * the assets are always present before the app starts.
 */

import { promises as fs } from "node:fs";
import { createWriteStream } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import https from "node:https";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, "..");
const TASKS_SRC = path.join(ROOT, "node_modules", "@mediapipe", "tasks-vision", "wasm");
const TASKS_DEST = path.join(ROOT, "public", "mediapipe", "tasks");
const MODELS_DEST = path.join(TASKS_DEST, "models");
const WASM_DEST = path.join(TASKS_DEST, "wasm");

const MODELS = [
  {
    name: "hand_landmarker.task",
    url: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
  },
  {
    name: "face_landmarker.task",
    url: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
  },
];

async function copyDir(src, dest) {
  await fs.mkdir(dest, { recursive: true });
  const entries = await fs.readdir(src, { withFileTypes: true });
  for (const entry of entries) {
    const s = path.join(src, entry.name);
    const d = path.join(dest, entry.name);
    if (entry.isDirectory()) await copyDir(s, d);
    else if (entry.isFile()) await fs.copyFile(s, d);
  }
}

function downloadOnce(url, destPath) {
  return new Promise((resolve, reject) => {
    const file = createWriteStream(destPath);
    const req = https.get(url, (res) => {
      // Follow one redirect (Google Storage occasionally 302s)
      if ([301, 302, 303, 307, 308].includes(res.statusCode)) {
        file.close();
        fs.unlink(destPath).catch(() => {});
        const next = res.headers.location;
        if (!next) return reject(new Error(`Redirect without Location: ${url}`));
        return downloadOnce(next, destPath).then(resolve, reject);
      }
      if (res.statusCode !== 200) {
        file.close();
        fs.unlink(destPath).catch(() => {});
        return reject(new Error(`HTTP ${res.statusCode} for ${url}`));
      }
      res.pipe(file);
      file.on("finish", () => file.close(resolve));
    });
    req.on("error", (err) => {
      file.close();
      fs.unlink(destPath).catch(() => {});
      reject(err);
    });
  });
}

async function ensureModel(name, url) {
  const dest = path.join(MODELS_DEST, name);
  try {
    const stat = await fs.stat(dest);
    if (stat.size > 1024) return;     // already downloaded
  } catch {}
  console.log(`[install-mediapipe] downloading ${name} …`);
  await fs.mkdir(MODELS_DEST, { recursive: true });
  await downloadOnce(url, dest);
  console.log(`[install-mediapipe] ✓ ${name}`);
}

async function main() {
  // 1. WASM bundle
  try {
    await fs.access(TASKS_SRC);
  } catch {
    console.warn("[install-mediapipe] tasks-vision not in node_modules yet — skipping");
    return;
  }
  await copyDir(TASKS_SRC, WASM_DEST);
  console.log("[install-mediapipe] ✓ tasks-vision/wasm copied to public/mediapipe/tasks/wasm/");

  // 2. .task models — downloaded once, then cached
  for (const m of MODELS) {
    try {
      await ensureModel(m.name, m.url);
    } catch (e) {
      console.warn(`[install-mediapipe] could not fetch ${m.name}: ${e.message}`);
      console.warn("  → drum / filter mode will fail at runtime until this is resolved");
    }
  }
}

main().catch((e) => {
  console.error("[install-mediapipe] failed:", e);
  process.exit(1);
});
