/**
 * AR filter renderer.
 *
 * Each filter is a pure 2D-canvas drawing routine that takes the FaceMesh
 * landmarks and draws an accessory aligned to the face (sunglasses,
 * party hat, mustache, devil horns, dog nose). No image assets — every
 * filter is hand-drawn with shapes — so the bundle stays tiny and the
 * filters scale crisply at any resolution.
 *
 * Coordinates: filters are drawn directly onto the *display* canvas,
 * which is the same size as the source video. The FaceMesh landmarks
 * are normalised to [0, 1]; we scale by the canvas dimensions on the
 * fly.
 */

import type { FaceLandmarks } from "./face-mesh";
import { FM } from "./face-mesh";

export type FilterId =
  | "none"
  | "sunglasses"
  | "party-hat"
  | "mustache"
  | "devil-horns"
  | "dog";

export interface FilterMeta {
  id: FilterId;
  label: string;
  emoji: string;
}

export const FILTER_LIST: FilterMeta[] = [
  { id: "none",        label: "None",         emoji: "🚫" },
  { id: "sunglasses",  label: "Cool Shades",  emoji: "🕶️" },
  { id: "party-hat",   label: "Party Hat",    emoji: "🎉" },
  { id: "mustache",    label: "Gentleman",    emoji: "👨" },
  { id: "devil-horns", label: "Devil Horns",  emoji: "😈" },
  { id: "dog",         label: "Pup Filter",   emoji: "🐶" },
];

// ── public API ─────────────────────────────────────────────

export function drawFilter(
  ctx: CanvasRenderingContext2D,
  lm: FaceLandmarks,
  filter: FilterId,
  width: number,
  height: number,
) {
  if (filter === "none") return;
  const g = buildGeometry(lm, width, height);
  switch (filter) {
    case "sunglasses":  return drawSunglasses(ctx, g);
    case "party-hat":   return drawPartyHat(ctx, g);
    case "mustache":    return drawMustache(ctx, g);
    case "devil-horns": return drawDevilHorns(ctx, g);
    case "dog":         return drawDog(ctx, g);
  }
}

// ── geometry derived from landmarks ─────────────────────────

interface FaceGeometry {
  // Eye centres (between iris if available, else outer/inner mid-points)
  leftEye: { x: number; y: number };
  rightEye: { x: number; y: number };
  eyeMid: { x: number; y: number };
  /** Pixel distance between the two eye centres — our scale unit. */
  eyeDist: number;
  /** Roll angle (radians) from the eye line, for rotating accessories. */
  angle: number;
  noseTip: { x: number; y: number };
  upperLip: { x: number; y: number };
  forehead: { x: number; y: number };
  chin: { x: number; y: number };
}

function buildGeometry(
  lm: FaceLandmarks, w: number, h: number,
): FaceGeometry {
  const px = (i: number) => ({ x: lm[i].x * w, y: lm[i].y * h });

  const haveIris = lm.length > FM.RIGHT_IRIS_CENTER;
  const leftEye = haveIris
    ? px(FM.LEFT_IRIS_CENTER)
    : midpoint(px(FM.LEFT_EYE_OUTER), px(FM.LEFT_EYE_INNER));
  const rightEye = haveIris
    ? px(FM.RIGHT_IRIS_CENTER)
    : midpoint(px(FM.RIGHT_EYE_OUTER), px(FM.RIGHT_EYE_INNER));

  const eyeMid = midpoint(leftEye, rightEye);
  const eyeDist = Math.hypot(rightEye.x - leftEye.x, rightEye.y - leftEye.y);
  const screenLeftEye = leftEye.x <= rightEye.x ? leftEye : rightEye;
  const screenRightEye = leftEye.x <= rightEye.x ? rightEye : leftEye;
  const angle = Math.atan2(
    screenRightEye.y - screenLeftEye.y,
    screenRightEye.x - screenLeftEye.x,
  );

  return {
    leftEye, rightEye, eyeMid, eyeDist, angle,
    noseTip: px(FM.NOSE_TIP),
    upperLip: px(FM.UPPER_LIP_TOP),
    forehead: px(FM.FOREHEAD_TOP),
    chin: px(FM.CHIN_BOTTOM),
  };
}

function midpoint(a: { x: number; y: number }, b: { x: number; y: number }) {
  return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
}

// ── filter implementations ─────────────────────────────────

function drawSunglasses(ctx: CanvasRenderingContext2D, g: FaceGeometry) {
  ctx.save();
  ctx.translate(g.eyeMid.x, g.eyeMid.y);
  ctx.rotate(g.angle);

  const lensR = g.eyeDist * 0.55;     // each lens radius
  const halfBridge = g.eyeDist / 2;   // half the inter-eye distance

  // Lenses
  ctx.fillStyle = "rgba(15, 15, 30, 0.88)";
  ctx.strokeStyle = "#1a1a1a";
  ctx.lineWidth = lensR * 0.18;

  ctx.beginPath();
  ctx.ellipse(-halfBridge, 0, lensR, lensR * 0.85, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();

  ctx.beginPath();
  ctx.ellipse(halfBridge, 0, lensR, lensR * 0.85, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();

  // Bridge
  ctx.lineWidth = lensR * 0.15;
  ctx.beginPath();
  ctx.moveTo(-halfBridge + lensR * 0.6, 0);
  ctx.lineTo(halfBridge - lensR * 0.6, 0);
  ctx.stroke();

  // Highlight glints
  ctx.fillStyle = "rgba(255, 255, 255, 0.35)";
  ctx.beginPath();
  ctx.ellipse(-halfBridge - lensR * 0.3, -lensR * 0.35,
              lensR * 0.18, lensR * 0.1, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.beginPath();
  ctx.ellipse(halfBridge - lensR * 0.3, -lensR * 0.35,
              lensR * 0.18, lensR * 0.1, 0, 0, Math.PI * 2);
  ctx.fill();

  ctx.restore();
}

function drawPartyHat(ctx: CanvasRenderingContext2D, g: FaceGeometry) {
  ctx.save();
  // Anchor at the forehead, then translate up by 10% of face height
  const faceHeight = g.chin.y - g.forehead.y;
  ctx.translate(g.forehead.x, g.forehead.y - faceHeight * 0.05);
  ctx.rotate(g.angle);

  const hatBase = g.eyeDist * 1.4;
  const hatHeight = g.eyeDist * 2.1;

  // Cone
  const grad = ctx.createLinearGradient(0, -hatHeight, 0, 0);
  grad.addColorStop(0, "#ec4899");
  grad.addColorStop(0.5, "#a855f7");
  grad.addColorStop(1, "#3b82f6");
  ctx.fillStyle = grad;
  ctx.strokeStyle = "#1a1a1a";
  ctx.lineWidth = hatBase * 0.04;
  ctx.beginPath();
  ctx.moveTo(-hatBase / 2, 0);
  ctx.lineTo(hatBase / 2, 0);
  ctx.lineTo(0, -hatHeight);
  ctx.closePath();
  ctx.fill();
  ctx.stroke();

  // Polka dots
  const dots = [
    { x: -hatBase * 0.18, y: -hatHeight * 0.25, r: hatBase * 0.07 },
    { x: hatBase * 0.15,  y: -hatHeight * 0.45, r: hatBase * 0.06 },
    { x: -hatBase * 0.05, y: -hatHeight * 0.65, r: hatBase * 0.05 },
  ];
  ctx.fillStyle = "rgba(255, 255, 255, 0.9)";
  for (const d of dots) {
    ctx.beginPath();
    ctx.arc(d.x, d.y, d.r, 0, Math.PI * 2);
    ctx.fill();
  }

  // Pom-pom on top
  ctx.fillStyle = "#fde047";
  ctx.beginPath();
  ctx.arc(0, -hatHeight, hatBase * 0.16, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = "#ca8a04";
  ctx.lineWidth = 2;
  ctx.stroke();

  ctx.restore();
}

function drawMustache(ctx: CanvasRenderingContext2D, g: FaceGeometry) {
  ctx.save();
  // Anchor between the upper lip and the nose tip
  const ax = (g.upperLip.x + g.noseTip.x) / 2;
  const ay = g.upperLip.y - (g.upperLip.y - g.noseTip.y) * 0.25;
  ctx.translate(ax, ay);
  ctx.rotate(g.angle);

  const w = g.eyeDist * 1.3;
  const h = g.eyeDist * 0.45;

  ctx.fillStyle = "#1f2937";
  // Two curling halves
  ctx.beginPath();
  ctx.moveTo(0, 0);
  ctx.bezierCurveTo(-w * 0.1, -h * 0.6, -w * 0.45, -h * 0.5, -w * 0.5, h * 0.1);
  ctx.bezierCurveTo(-w * 0.55, h * 0.6, -w * 0.35, h * 0.7, -w * 0.2, h * 0.4);
  ctx.bezierCurveTo(-w * 0.1, h * 0.2, -w * 0.05, h * 0.2, 0, h * 0.1);
  ctx.closePath();
  ctx.fill();

  ctx.beginPath();
  ctx.moveTo(0, 0);
  ctx.bezierCurveTo(w * 0.1, -h * 0.6, w * 0.45, -h * 0.5, w * 0.5, h * 0.1);
  ctx.bezierCurveTo(w * 0.55, h * 0.6, w * 0.35, h * 0.7, w * 0.2, h * 0.4);
  ctx.bezierCurveTo(w * 0.1, h * 0.2, w * 0.05, h * 0.2, 0, h * 0.1);
  ctx.closePath();
  ctx.fill();

  ctx.restore();
}

function drawDevilHorns(ctx: CanvasRenderingContext2D, g: FaceGeometry) {
  ctx.save();
  ctx.translate(g.forehead.x, g.forehead.y);
  ctx.rotate(g.angle);

  const offset = g.eyeDist * 0.6;
  const hornH = g.eyeDist * 0.9;
  const hornW = g.eyeDist * 0.35;

  drawHorn(ctx, -offset, 0, hornW, hornH, -1);
  drawHorn(ctx,  offset, 0, hornW, hornH,  1);

  ctx.restore();
}

function drawHorn(
  ctx: CanvasRenderingContext2D,
  cx: number, cy: number, w: number, h: number, dir: number,
) {
  const grad = ctx.createLinearGradient(cx, cy, cx, cy - h);
  grad.addColorStop(0, "#7f1d1d");
  grad.addColorStop(1, "#dc2626");
  ctx.fillStyle = grad;
  ctx.strokeStyle = "#3b0a0a";
  ctx.lineWidth = w * 0.12;
  ctx.beginPath();
  ctx.moveTo(cx - w / 2, cy);
  ctx.quadraticCurveTo(cx + dir * w * 0.4, cy - h * 0.5,
                        cx + dir * w * 0.05, cy - h);
  ctx.quadraticCurveTo(cx - dir * w * 0.1, cy - h * 0.6, cx + w / 2, cy);
  ctx.closePath();
  ctx.fill();
  ctx.stroke();
}

function drawDog(ctx: CanvasRenderingContext2D, g: FaceGeometry) {
  ctx.save();

  // Floppy ears (anchored at the temples — between forehead and eyes)
  const earOffset = g.eyeDist * 1.05;
  const earTopY = g.forehead.y + (g.eyeMid.y - g.forehead.y) * 0.4;

  ctx.translate(g.eyeMid.x, earTopY);
  ctx.rotate(g.angle);

  const earW = g.eyeDist * 0.7;
  const earH = g.eyeDist * 1.4;

  // Left ear
  ctx.fillStyle = "#a16207";
  ctx.strokeStyle = "#422006";
  ctx.lineWidth = earW * 0.06;

  for (const sign of [-1, 1]) {
    ctx.beginPath();
    ctx.ellipse(sign * earOffset, earH * 0.1, earW * 0.5, earH * 0.55,
                sign * 0.25, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    // inner pink
    ctx.fillStyle = "#fda4af";
    ctx.beginPath();
    ctx.ellipse(sign * earOffset, earH * 0.18, earW * 0.28, earH * 0.4,
                sign * 0.25, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = "#a16207";
  }

  ctx.restore();

  // Black dog nose on the actual nose tip
  ctx.save();
  ctx.translate(g.noseTip.x, g.noseTip.y);
  ctx.rotate(g.angle);
  const noseR = g.eyeDist * 0.16;
  ctx.fillStyle = "#0f172a";
  ctx.beginPath();
  ctx.ellipse(0, 0, noseR, noseR * 0.78, 0, 0, Math.PI * 2);
  ctx.fill();
  // highlight
  ctx.fillStyle = "rgba(255,255,255,0.35)";
  ctx.beginPath();
  ctx.ellipse(-noseR * 0.3, -noseR * 0.35, noseR * 0.25, noseR * 0.15,
              0, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();

  // Tongue under the lower lip
  ctx.save();
  ctx.translate(g.upperLip.x, g.upperLip.y + g.eyeDist * 0.55);
  ctx.rotate(g.angle);
  ctx.fillStyle = "#fb7185";
  ctx.strokeStyle = "#9f1239";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.ellipse(0, 0, g.eyeDist * 0.22, g.eyeDist * 0.32,
              0, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
  ctx.restore();
}
