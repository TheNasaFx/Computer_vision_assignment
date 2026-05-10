export type Detection = {
  bbox: number[];
  confidence: number;
  class_id: number;
  class_name: string;
  track_id: number | null;
};

type Track = {
  key: string;
  bbox: number[];
  confidence: number;
  class_id: number;
  class_name: string;
  track_id: number | null;
  hits: number;
  misses: number;
  lastSeen: number;
};

type StabilizerOptions = {
  minConfidence?: number;
  instantConfidence?: number;
  minHits?: number;
  iouThreshold?: number;
  boxAlpha?: number;
  confidenceAlpha?: number;
  holdMs?: number;
  maxMisses?: number;
};

export class DetectionStabilizer {
  private tracks = new Map<string, Track>();
  private nextId = 1;

  private readonly opts: Required<StabilizerOptions>;

  constructor(options: StabilizerOptions = {}) {
    this.opts = {
      minConfidence: options.minConfidence ?? 0.18,
      instantConfidence: options.instantConfidence ?? 0.55,
      minHits: options.minHits ?? 2,
      iouThreshold: options.iouThreshold ?? 0.28,
      boxAlpha: options.boxAlpha ?? 0.42,
      confidenceAlpha: options.confidenceAlpha ?? 0.35,
      holdMs: options.holdMs ?? 320,
      maxMisses: options.maxMisses ?? 5,
    };
  }

  reset() {
    this.tracks.clear();
    this.nextId = 1;
  }

  update(detections: Detection[], now = performance.now()): Detection[] {
    const clean = detections
      .filter((d) => d.bbox.length === 4 && d.confidence >= this.opts.minConfidence)
      .sort((a, b) => b.confidence - a.confidence);

    const unmatched = new Set(this.tracks.keys());

    for (const det of clean) {
      let track = this.findTrack(det, unmatched);
      if (!track) {
        track = this.createTrack(det, now);
        this.tracks.set(track.key, track);
      }

      unmatched.delete(track.key);
      const alpha = track.hits === 0 ? 1 : this.opts.boxAlpha;
      track.bbox = mixBox(track.bbox, det.bbox, alpha);
      track.confidence = lerp(track.confidence, det.confidence, this.opts.confidenceAlpha);
      track.class_id = det.class_id;
      track.class_name = det.class_name;
      track.track_id = det.track_id;
      track.hits += 1;
      track.misses = 0;
      track.lastSeen = now;
    }

    for (const key of unmatched) {
      const track = this.tracks.get(key);
      if (!track) continue;
      track.misses += 1;
      track.confidence *= 0.92;
      if (track.misses > this.opts.maxMisses || now - track.lastSeen > this.opts.holdMs) {
        this.tracks.delete(key);
      }
    }

    return Array.from(this.tracks.values())
      .filter((track) => this.isVisible(track, now))
      .sort((a, b) => a.bbox[0] - b.bbox[0])
      .map((track) => ({
        bbox: track.bbox,
        confidence: track.confidence,
        class_id: track.class_id,
        class_name: track.class_name,
        track_id: track.track_id,
      }));
  }

  private findTrack(det: Detection, unmatched: Set<string>): Track | null {
    if (det.track_id !== null) {
      const exact = this.tracks.get(trackKey(det));
      if (exact && unmatched.has(exact.key)) return exact;
    }

    let best: Track | null = null;
    let bestScore = 0;
    for (const key of unmatched) {
      const candidate = this.tracks.get(key);
      if (!candidate || candidate.class_id !== det.class_id) continue;
      const score = iou(candidate.bbox, det.bbox);
      if (score > bestScore) {
        bestScore = score;
        best = candidate;
      }
    }
    return bestScore >= this.opts.iouThreshold ? best : null;
  }

  private createTrack(det: Detection, now: number): Track {
    return {
      key: det.track_id !== null ? trackKey(det) : `local:${this.nextId++}`,
      bbox: det.bbox.slice(),
      confidence: det.confidence,
      class_id: det.class_id,
      class_name: det.class_name,
      track_id: det.track_id,
      hits: 0,
      misses: 0,
      lastSeen: now,
    };
  }

  private isVisible(track: Track, now: number): boolean {
    if (now - track.lastSeen > this.opts.holdMs) return false;
    if (track.confidence >= this.opts.instantConfidence) return true;
    return track.hits >= this.opts.minHits;
  }
}

function trackKey(det: Detection): string {
  return `track:${det.class_id}:${det.track_id}`;
}

function mixBox(a: number[], b: number[], alpha: number): number[] {
  return [
    lerp(a[0], b[0], alpha),
    lerp(a[1], b[1], alpha),
    lerp(a[2], b[2], alpha),
    lerp(a[3], b[3], alpha),
  ];
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

function iou(a: number[], b: number[]): number {
  const x1 = Math.max(a[0], b[0]);
  const y1 = Math.max(a[1], b[1]);
  const x2 = Math.min(a[2], b[2]);
  const y2 = Math.min(a[3], b[3]);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const areaA = Math.max(0, a[2] - a[0]) * Math.max(0, a[3] - a[1]);
  const areaB = Math.max(0, b[2] - b[0]) * Math.max(0, b[3] - b[1]);
  const union = areaA + areaB - inter;
  return union > 0 ? inter / union : 0;
}
