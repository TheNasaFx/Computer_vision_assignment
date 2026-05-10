/**
 * Face-landmark wrapper built on MediaPipe Tasks-Vision.
 *
 * FaceLandmarker returns 478 normalised 3D landmarks per face — the
 * same canonical set the legacy `FaceMesh` exposed (468 base + 10 iris
 * landmarks when `outputFaceBlendshapes` / `outputFacialTransformation`
 * features are enabled). For our AR filters we only need a handful of
 * indices defined in `FM`.
 *
 * The Tasks-Vision API replaces the legacy Solutions API for the same
 * reasons documented in `lib/hands.ts`: it ships as an ES module with
 * its own WASM resolution and avoids the global-`Module` foot-gun that
 * breaks under Next.js bundling.
 */

import {
  FaceLandmarker,
  FilesetResolver,
} from "@mediapipe/tasks-vision";

const WASM_BASE = "/mediapipe/tasks/wasm";
const MODEL_PATH = "/mediapipe/tasks/models/face_landmarker.task";

export type Landmark = { x: number; y: number; z: number };
export type FaceLandmarks = Landmark[];

export interface FaceMeshHandle {
  send: (video: HTMLVideoElement) => Promise<void>;
  landmarks: FaceLandmarks | null;
  faces: FaceLandmarks[];
  inferenceMs: number;
  close: () => void;
}

export async function createFaceMesh(): Promise<FaceMeshHandle> {
  const fileset = await FilesetResolver.forVisionTasks(WASM_BASE);
  const landmarker = await FaceLandmarker.createFromOptions(fileset, {
    baseOptions: {
      modelAssetPath: MODEL_PATH,
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numFaces: 2,
    outputFaceBlendshapes: false,
    outputFacialTransformationMatrixes: false,
  });

  let landmarks: FaceLandmarks | null = null;
  let faces: FaceLandmarks[] = [];
  let inferenceMs = 0;
  let lastVideoTs = 0;

  const send = async (video: HTMLVideoElement) => {
    if (video.readyState < 2) return;

    const t0 = performance.now();
    let videoTs = Math.round(t0);
    if (videoTs <= lastVideoTs) videoTs = lastVideoTs + 1;
    lastVideoTs = videoTs;

    let result;
    try {
      result = landmarker.detectForVideo(video, videoTs);
    } catch {
      return;
    }
    inferenceMs = performance.now() - t0;

    if (result.faceLandmarks && result.faceLandmarks.length > 0) {
      faces = result.faceLandmarks as FaceLandmarks[];
      landmarks = faces[0];
    } else {
      faces = [];
      landmarks = null;
    }
  };

  const close = () => {
    try { landmarker.close(); } catch {}
  };

  const handle: FaceMeshHandle = Object.create({ send, close });
  Object.defineProperty(handle, "landmarks", {
    get: () => landmarks, enumerable: true,
  });
  Object.defineProperty(handle, "faces", {
    get: () => faces, enumerable: true,
  });
  Object.defineProperty(handle, "inferenceMs", {
    get: () => inferenceMs, enumerable: true,
  });
  return handle;
}

// ── canonical landmark indices ─────────────────────────────
// FaceLandmarker uses the same 468 mesh layout as FaceMesh, with iris
// landmarks 468..477 always present.

export const FM = {
  NOSE_TIP: 1,
  LEFT_EYE_OUTER: 33,
  LEFT_EYE_INNER: 133,
  RIGHT_EYE_OUTER: 263,
  RIGHT_EYE_INNER: 362,
  LEFT_IRIS_CENTER: 468,
  RIGHT_IRIS_CENTER: 473,
  LEFT_EAR_TOP: 234,
  RIGHT_EAR_TOP: 454,
  CHIN_BOTTOM: 152,
  FOREHEAD_TOP: 10,
  MOUTH_LEFT: 61,
  MOUTH_RIGHT: 291,
  UPPER_LIP_TOP: 13,
  LOWER_LIP_BOTTOM: 14,
};

export function lmToPixel(
  lm: Landmark, canvasWidth: number, canvasHeight: number,
): { x: number; y: number } {
  return { x: lm.x * canvasWidth, y: lm.y * canvasHeight };
}
