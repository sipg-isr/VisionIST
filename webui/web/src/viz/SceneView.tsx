/** scene visualizer — a reconstruction in one orbiting three.js view:
 *
 *   - sparse points  (P, 3), optional colors (P, 3) in 0..1
 *   - cameras        (F, 3, 4) world → camera [R | t] (default) OR (F, 4, 4)
 *                    camera → world, when the def declares
 *                    `cameras_convention: camera_to_world` — a frustum per
 *                    camera + the path through the centres, coloured in
 *                    frame order (viridis first → last, first camera white)
 *                    — legible even when the camera barely moves; a
 *                    cameras-only scene (no points) is legal, for pose-only
 *                    replies; frustum shape from intrinsics (3, 3, pixels)
 *                    + image_size [W, H] when given, else a 60° 4:3 default
 *   - dense points   (N, 3) + optional dense_frames (N,) — e.g. depth/point
 *                    maps per frame. When the dense source is a (T, H, W, 3)
 *                    grid and no dense_frames is declared, the per-point
 *                    frame index is derived from the shape (point i →
 *                    frame i / (H·W)), so the cloud can be coloured by the
 *                    frame each layer came from.
 *
 *  Coordinates are OpenCV camera conventions (x right, y down, z forward);
 *  the view flips y and z so "up" is up and camera 0 looks into the screen.
 *  Nothing here knows a box: the def's params name the keys.
 */
import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import type { SerValue } from "../api";
import { fetchTyped, flattenNumbers, isRef } from "../resolvers";
import { heatColor } from "./MatrixHeatmap";

type Num = ArrayLike<number>;

interface SceneData {
  points: Float32Array | null;
  colors: Float32Array | null;
  cameras: Num | null;            // F × 12, row-major [R | t]
  dense: Float32Array | null;
  denseColors: Float32Array | null;
  denseFrames: Num | null;
  K: Num | null;
  size: [number, number] | null;
}

async function numbers(v: unknown): Promise<Num | null> {
  if (v === undefined || v === null) return null;
  if (isRef(v)) {
    const t = await fetchTyped(v as SerValue);
    return t && t.data.length > 0 ? t.data : null;
  }
  return flattenNumbers(v);
}

function f32(a: Num | null): Float32Array | null {
  if (!a) return null;
  return a instanceof Float32Array ? a : Float32Array.from(a);
}

/** Per-level lengths of a regular nested array (inline values only). */
function inferShape(v: unknown): number[] | null {
  const s: number[] = [];
  let cur: unknown = v;
  while (Array.isArray(cur) && cur.length > 0) { s.push(cur.length); cur = cur[0]; }
  return s.length ? s : null;
}

/** Dense cloud in one fetch, plus its per-frame index: explicit in the
 *  buffer ref (shape), otherwise inferred from the inline nesting. A
 *  (T, H, W, 3) grid — the common per-frame point/depth-map shape — yields
 *  a frame index per point (point i → i / (H·W)). Nothing here knows a box. */
async function denseTyped(
  v: unknown,
): Promise<{ data: Float32Array | null; frames: Num | null }> {
  if (v === undefined || v === null) return { data: null, frames: null };
  let raw: ArrayLike<number> | null = null;
  let shape: number[] | null = null;
  if (isRef(v)) {
    const t = await fetchTyped(v as SerValue);
    if (!t || t.data.length === 0) return { data: null, frames: null };
    raw = t.data;
    shape = t.shape ?? null;
  } else {
    const flat = flattenNumbers(v);
    if (!flat || flat.length === 0) return { data: null, frames: null };
    raw = flat;
    shape = inferShape(v);
  }
  const data = new Float32Array(raw as number[]);
  let frames: Num | null = null;
  if (shape && shape.length === 4 && Number(shape[3]) === 3) {
    const T = Math.max(1, Number(shape[0])), H = Number(shape[1]), W = Number(shape[2]);
    const N = data.length / 3;
    if (T * H * W === N) {
      const per = H * W;
      const fr = new Float64Array(N);
      for (let i = 0; i < N; i++) fr[i] = Math.floor(i / per);
      frames = fr;
    }
  }
  return { data, frames };
}

/** three.js vertex colors are 0..1; boxes may hand over integer 0..255
 *  RGB (e.g. uint8 scene colours) — rescaled (sampled, cheap), otherwise
 *  every point would clamp to washed-out white.  Box-agnostic heuristic: */
function unitize(a: Float32Array | null): Float32Array | null {
  if (!a || !a.length) return a;
  let m = 0;
  const step = Math.max(1, Math.floor(a.length / 20000));
  for (let i = 0; i < a.length; i += step) {
    for (let j = 0; j < 3 && i + j < a.length; j++) { m = Math.max(m, a[i + j]); }
  }
  if (m <= 1.5) return a;
  const out = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = a[i] / m;
  return out;
}

/** OpenCV (x, y, z) → view (x, −y, −z), in place. */
function toView(a: Float32Array): Float32Array {
  for (let i = 0; i < a.length; i += 3) { a[i + 1] = -a[i + 1]; a[i + 2] = -a[i + 2]; }
  return a;
}

function rgbOf(css: string): [number, number, number] {
  const m = css.match(/\d+/g);
  return m ? [Number(m[0]) / 255, Number(m[1]) / 255, Number(m[2]) / 255] : [1, 1, 1];
}

export function SceneView({
  title, points, colors, cameras, densePoints, denseColors, denseFrames, intrinsics, imageSize,
  camerasConvention = "world_to_camera",
}: {
  title?: string;
  points: unknown;
  colors?: unknown;
  cameras?: unknown;
  camerasConvention?: string;   // "world_to_camera" (default, e.g. sfm/colmap) | "camera_to_world"
  densePoints?: unknown;
  denseColors?: unknown;
  denseFrames?: unknown;
  intrinsics?: unknown;
  imageSize?: unknown;
}) {
  const [data, setData] = useState<SceneData | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [showSparse, setShowSparse] = useState(true);
  const [showDense, setShowDense] = useState(true);
  const [showCams, setShowCams] = useState(true);
  const [byFrame, setByFrame] = useState(true);
  const [ptSize, setPtSize] = useState(1);
  const [frustumScale, setFrustumScale] = useState(1);

  useEffect(() => {
    let alive = true;
    setData(null);
    setErr(null);
    void (async () => {
      try {
        const [p, c, cam, denseArr, dc, df, K] = await Promise.all([
          numbers(points), numbers(colors), numbers(cameras),
          denseTyped(densePoints), numbers(denseColors), numbers(denseFrames), numbers(intrinsics),
        ]);
        const sz = flattenNumbers(imageSize);
        if (!alive) return;
        const d = denseArr.data;
        if (!p && !d && !cam) { setErr("no points or cameras in this response"); return; }
        // an explicit dense_frames wins; otherwise the derived one (above),
        // which keeps the “color by source frame” toggle available for any
        // per-frame grid cloud
        const frames = df ?? denseArr.frames;
        setData({
          points: p ? toView(Float32Array.from(p)) : null,
          colors: unitize(f32(c)),
          cameras: cam,
          dense: d ? toView(d) : null,
          denseColors: unitize(f32(dc)),
          denseFrames: frames,
          K: K && K.length >= 9 ? K : null,
          size: sz && sz.length >= 2 ? [sz[0], sz[1]] : null,
        });
      } catch (e) {
        if (alive) setErr(String((e as Error).message || e));
      }
    })();
    return () => { alive = false; };
  }, [points, colors, cameras, densePoints, denseColors, denseFrames, intrinsics, imageSize]);

  const nCamsRaw = data?.cameras ? data.cameras.length : 0;
  const nSparse = data?.points ? data.points.length / 3 : 0;
  const nDense = data?.dense ? data.dense.length / 3 : 0;

  return (
    <div>
      {title && <div className="viz-caption">{title}</div>}
      {err && <div className="note">{err}</div>}
      {!data && !err && <span className="spinnerbox"><span className="spinner" /></span>}
      {data && (
        <>
          <div className="btnrow" style={{ marginBottom: 6, flexWrap: "wrap", gap: 10, fontSize: 12.5 }}>
            {nSparse > 0 && (
              <label><input type="checkbox" checked={showSparse} onChange={(e) => setShowSparse(e.target.checked)} /> tracks ({nSparse.toLocaleString()})</label>
            )}
            {nDense > 0 && (
              <label><input type="checkbox" checked={showDense} onChange={(e) => setShowDense(e.target.checked)} /> depth ({nDense.toLocaleString()})</label>
            )}
            {nCamsRaw > 0 && (
              <label><input type="checkbox" checked={showCams} onChange={(e) => setShowCams(e.target.checked)} /> cameras ({Math.floor(nCamsRaw / (camerasConvention === "camera_to_world" ? 16 : 12))})</label>
            )}
            {nDense > 0 && data.denseFrames && (
              <label title="color the dense points by the frame they came from (viridis: first → last) — per-frame layering and motion">
                <input type="checkbox" checked={byFrame} onChange={(e) => setByFrame(e.target.checked)} /> color cloud by source frame
              </label>
            )}
            <label>point size
              <input type="range" min={0.25} max={4} step={0.25} value={ptSize}
                onChange={(e) => setPtSize(Number(e.target.value))} style={{ width: 90, marginLeft: 6 }} />
            </label>
            {nCamsRaw > 0 && (
              <label title="scale the camera frustums (1 = default)">
                frustum
                <input type="range" min={0.1} max={4} step={0.1} value={frustumScale}
                  onChange={(e) => setFrustumScale(Number(e.target.value))} style={{ width: 90, marginLeft: 6 }} />
              </label>
            )}
          </div>
          <SceneCanvas data={data} showSparse={showSparse} showDense={showDense}
            showCams={showCams} byFrame={byFrame} ptSize={ptSize} frustumScale={frustumScale}
            c2w={camerasConvention === "camera_to_world"} />
          <div className="hint" style={{ fontFamily: "var(--mono)", fontSize: 11 }}>
            drag: orbit · right-drag: pan · wheel: zoom · frustums run first → last in order (first camera white)
          </div>
        </>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------

/** Scene extent: 2–98 % range of the sparse points (dense when there are
 *  none) — robust to a few far-off points — grown to hold every camera
 *  centre (cameras may orbit the points). */
function extent(pts: Float32Array, centres: number[][]): { center: THREE.Vector3; size: number } {
  const lo: number[] = [], hi: number[] = [];
  const n = pts.length / 3;
  const step = Math.max(1, Math.floor(n / 20000));
  for (let c = 0; c < 3; c++) {
    const vals: number[] = [];
    for (let i = 0; i < n; i += step) { const x = pts[i * 3 + c]; if (Number.isFinite(x)) vals.push(x); }
    vals.sort((a, b) => a - b);
    let l = vals.length ? vals[Math.floor(0.02 * (vals.length - 1))] : Infinity;
    let h = vals.length ? vals[Math.ceil(0.98 * (vals.length - 1))] : -Infinity;
    for (const C of centres) { l = Math.min(l, C[c]); h = Math.max(h, C[c]); }
    lo.push(Number.isFinite(l) ? l : -1);
    hi.push(Number.isFinite(h) ? h : 1);
  }
  return {
    center: new THREE.Vector3((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2),
    size: Math.max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2], 1),   // ≥ 1 unit: a zero-extent scene (e.g. all cameras at the origin) would otherwise collapse to nothing
  };
}

function SceneCanvas({
  data, showSparse, showDense, showCams, byFrame, ptSize, frustumScale, c2w,
}: {
  data: SceneData; showSparse: boolean; showDense: boolean; showCams: boolean;
  byFrame: boolean; ptSize: number; frustumScale: number; c2w: boolean;
}) {
  const holderRef = useRef<HTMLDivElement | null>(null);
  const objs = useRef<{ sparse?: THREE.Points; dense?: THREE.Points; cams?: THREE.Group;
    denseColor?: THREE.BufferAttribute; frameColor?: THREE.BufferAttribute;
    base?: { sparse: number; dense: number }; camsGeo?: THREE.BufferGeometry;
    frustumScaleFn?: (s: number) => void }>({});
  const [failed, setFailed] = useState<string | null>(null);

  // camera centres + the per-camera “camera-frame point → world” map
  const camGeo = useMemo(() => {
    const cams = data.cameras;
    if (!cams || !cams.length) return null;
    // world_to_camera: (F, 3, 4) [R|t] — 12 per camera (sfm/colmap).
    // camera_to_world:  (F, 4, 4) row-major — 16 per camera (abot_recon).
    const per = c2w ? 16 : 12;
    if (cams.length % per !== 0) return null;
    const F = Math.floor(cams.length / per);
    const centres: number[][] = [];
    const apply: ((j: number, x: number[]) => number)[] = [];
    for (let f = 0; f < F; f++) {
      const o = f * per;
      const R = [0, 1, 2, 4, 5, 6, 8, 9, 10].map((i) => Number(cams[o + i]));
      if (c2w) {
        // row-major 4×4: R = rows 0–2, translation = COLUMN 3 → indices 3, 7,
        // 11 (NOT 12–14, which are the homogeneous last row — that read put
        // every centre at the origin: the “rotating in place” bug)
        const t = [Number(cams[o + 3]), Number(cams[o + 7]), Number(cams[o + 11])];
        // centre = the translation; world point of camera-frame x = t + R·x.
        // NOTE: R is stored row-major, so row j of R is R[3*j .. 3*j+2] and we take
        // (R·x)[j] = R[3j]x0 + R[3j+1]x1 + R[3j+2]x2.  (Taking column j instead —
        // R[j],R[j+3],R[j+6] — gives Rᵀ·x, which is correct for the [R|t] world->
        // camera matrix below but WRONG here: with M=I (frame 0) it looked right,
        // then every real rotation rendered the frustum on a different axis.)
        centres.push(t);
        apply.push((j, x) => t[j] + R[3 * j] * x[0] + R[3 * j + 1] * x[1] + R[3 * j + 2] * x[2]);
      } else {
        const t = [Number(cams[o + 3]), Number(cams[o + 7]), Number(cams[o + 11])];
        // centre = −Rᵀt; world point of camera-frame x = Rᵀ(x − t)
        const C = [0, 1, 2].map((j) => -(R[j] * t[0] + R[3 + j] * t[1] + R[6 + j] * t[2]));
        centres.push(C);
        apply.push((j, x) => R[j] * (x[0] - t[0]) + R[3 + j] * (x[1] - t[1]) + R[6 + j] * (x[2] - t[2]));
      }
    }
    return { F, centres, apply };
  }, [data.cameras, c2w]);

  useEffect(() => {
    const holder = holderRef.current;
    if (!holder) return;
    let disposed = false, raf = 0;
    const disposables: { dispose: () => void }[] = [];
    let renderer: THREE.WebGLRenderer, controls: OrbitControls;
    try {
      const centresView = (camGeo?.centres ?? []).map((c) => [c[0], -c[1], -c[2]]);
      const ext = extent(data.points ?? data.dense ?? new Float32Array(0), centresView);
      const size = ext.size;

      const scene = new THREE.Scene();
      scene.background = new THREE.Color("#0a0d11");
      const camera = new THREE.PerspectiveCamera(50, holder.clientWidth / holder.clientHeight, size / 1e4, size * 1e3);
      renderer = new THREE.WebGLRenderer({ antialias: true });
      renderer.setPixelRatio(window.devicePixelRatio || 1);
      renderer.setSize(holder.clientWidth, holder.clientHeight);
      holder.appendChild(renderer.domElement);

      const root = new THREE.Group();
      root.position.copy(ext.center.clone().negate());
      scene.add(root);

      const mkPoints = (pos: Float32Array, col: Float32Array | null, n: number, sz: number, fallback: string) => {
        const g = new THREE.BufferGeometry();
        g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
        let colorAttr: THREE.BufferAttribute | undefined;
        if (col && col.length >= n * 3) {
          colorAttr = new THREE.BufferAttribute(col, 3);
          g.setAttribute("color", colorAttr);
        }
        const m = new THREE.PointsMaterial({
          size: sz, sizeAttenuation: true, vertexColors: !!colorAttr,
          color: colorAttr ? 0xffffff : new THREE.Color(fallback),
        });
        disposables.push(g, m);
        return { pts: new THREE.Points(g, m), colorAttr };
      };

      const base = { sparse: size / 120, dense: size / 400 };
      objs.current.base = base;
      if (data.dense) {
        const n = data.dense.length / 3;
        const { pts, colorAttr } = mkPoints(data.dense, data.denseColors, n, base.dense, "#8aa4c8");
        objs.current.dense = pts;
        objs.current.denseColor = colorAttr;
        if (data.denseFrames && data.denseFrames.length === n) {
          let maxF = 0;
          for (let i = 0; i < n; i++) maxF = Math.max(maxF, Number(data.denseFrames[i]));
          const F = Math.max(1, (camGeo?.F ?? 0) || maxF + 1);
          const lut = Array.from({ length: F }, (_, f) => rgbOf(heatColor(F > 1 ? f / (F - 1) : 0.5)));
          const fc = new Float32Array(n * 3);
          for (let i = 0; i < n; i++) {
            const c = lut[Math.min(F - 1, Math.max(0, Math.round(Number(data.denseFrames[i]))))];
            fc[i * 3] = c[0]; fc[i * 3 + 1] = c[1]; fc[i * 3 + 2] = c[2];
          }
          objs.current.frameColor = new THREE.BufferAttribute(fc, 3);
        }
        root.add(pts);
      }
      if (data.points) {
        const n = data.points.length / 3;
        const { pts } = mkPoints(data.points, data.colors, n, base.sparse, "#ffd166");
        objs.current.sparse = pts;
        root.add(pts);
      }

      if (camGeo && camGeo.F > 0) {
        const group = new THREE.Group();
        // frustum image size: from the def when given; else, with intrinsics
        // known, from the principal point (cx, cy assumed near centre —
        // W = 2·cx, H = 2·cy); otherwise a 60° 4:3 default.  (A K built for a
        // 1600px image against a 4px "image" degenerates to a point.)
        const K9 = data.K ? Array.from({ length: Math.min(9, data.K.length) }, (_, i) => Number(data.K![i])) : null;
        const W = data.size ? data.size[0] : (K9 ? 2 * K9[2] : 4);
        const H = data.size ? data.size[1] : (K9 ? 2 * K9[5] : 3);
        const baseL = size * 0.06;   // base frustum length — the “frustum” slider multiplies this
        const base4 = [[0, 0], [W, 0], [W, H], [0, H]];
        // corners for a given length L — keeps the image's W:H aspect (K9 pinhole where
        // given; else one shared scale, longer side = L)
        const cornersFor = (L: number): [number, number, number][] => {
          if (K9) {
            const fx = K9[0], fy = K9[4], cx = K9[2], cy = K9[5];
            return base4.map(([u, v]) => [((u - cx) / fx) * L, ((v - cy) / fy) * L, L] as [number, number, number]);
          }
          const cx = W / 2, cy = H / 2, sc = L / (Math.max(W, H) / 2);
          return base4.map(([u, v]) => [(u - cx) * sc, (v - cy) * sc, L] as [number, number, number]);
        };
        const F = camGeo.F;
        const camLut = F > 1
          ? Array.from({ length: F }, (_, f) => rgbOf(heatColor(f / (F - 1))))
          : [[1, 1, 1] as [number, number, number]];
        // per-segment colour — frame-ordered (viridis first → last), first camera white;
        // independent of the frustum size, computed once
        const segCol: number[] = [];
        for (let f = 0; f < F; f++) {
          const col = f === 0 ? [1, 1, 1] : camLut[f];
          for (let k = 0; k < 8; k++) segCol.push(col[0], col[1], col[2], col[0], col[1], col[2]);
        }
        // the camera path (through the centres) — fixed, never scaled with the frustums
        const path: number[] = [];
        for (let f = 0; f < F; f++) { const C0 = camGeo.centres[f]; path.push(C0[0], -C0[1], -C0[2]); }
        // recompute frustum positions at length L, per camera, around that camera's
        // centre — so a frustum resizes in place (colours + path stay fixed)
        const segsFor = (L: number): Float32Array => {
          const segs: number[] = [];
          const corners = cornersFor(L);
          for (let f = 0; f < F; f++) {
            const C = camGeo.centres[f];
            const w = corners.map((x) => [0, 1, 2].map((j) => camGeo.apply[f](j, x)));
            const push = (a: number[], b: number[]) => segs.push(a[0], -a[1], -a[2], b[0], -b[1], -b[2]);
            for (let i = 0; i < 4; i++) { push(C, w[i]); push(w[i], w[(i + 1) % 4]); }
          }
          return new Float32Array(segs);
        };
        const frustumGeo = new THREE.BufferGeometry();
        frustumGeo.setAttribute("position", new THREE.BufferAttribute(segsFor(baseL), 3));
        frustumGeo.setAttribute("color", new THREE.BufferAttribute(new Float32Array(segCol), 3));
        const frustumMat = new THREE.LineBasicMaterial({ vertexColors: true });
        disposables.push(frustumGeo, frustumMat);
        group.add(new THREE.LineSegments(frustumGeo, frustumMat));
        if (path.length > 3) {
          const pg = new THREE.BufferGeometry();
          pg.setAttribute("position", new THREE.BufferAttribute(new Float32Array(path), 3));
          const pm = new THREE.LineBasicMaterial(); pm.color.set("#e8edf3");
          disposables.push(pg, pm);
          group.add(new THREE.Line(pg, pm));
        }
        objs.current.cams = group;
        objs.current.camsGeo = frustumGeo;
        // frustum-size slider: rescale only the frustum geometry (cheap — no scene rebuild)
        objs.current.frustumScaleFn = (s: number) => {
          const g = objs.current.camsGeo;
          if (!g) return;
          g.setAttribute("position", new THREE.BufferAttribute(segsFor(baseL * Math.max(0.01, s)), 3));
          (g.getAttribute("position") as THREE.BufferAttribute).needsUpdate = true;
          g.computeBoundingSphere();
        };
        root.add(group);
      }

      controls = new OrbitControls(camera, renderer.domElement);
      controls.enableDamping = true;
      // start behind and slightly above the first camera, looking at the scene
      camera.position.set(size * 0.15, size * 0.35, size * 1.3);
      controls.target.set(0, 0, 0);
      controls.update();

      const loop = () => {
        if (disposed) return;
        controls.update();
        renderer.render(scene, camera);
        raf = requestAnimationFrame(loop);
      };
      loop();

      const ro = new ResizeObserver(() => {
        if (disposed || !holder.clientWidth) return;
        camera.aspect = holder.clientWidth / holder.clientHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(holder.clientWidth, holder.clientHeight);
      });
      ro.observe(holder);

      return () => {
        disposed = true;
        cancelAnimationFrame(raf);
        ro.disconnect();
        controls.dispose();
        renderer.dispose();
        for (const d of disposables) d.dispose();
        if (renderer.domElement.parentNode === holder) holder.removeChild(renderer.domElement);
        objs.current = {};
      };
    } catch (e) {
      setFailed(String((e as Error).message || e));
      return undefined;
    }
  }, [data, camGeo]);

  // toggles: no scene rebuild
  useEffect(() => {
    const o = objs.current;
    if (o.sparse) {
      o.sparse.visible = showSparse;
      (o.sparse.material as THREE.PointsMaterial).size = (o.base?.sparse ?? 1) * ptSize;
    }
    if (o.dense) {
      o.dense.visible = showDense;
      const m = o.dense.material as THREE.PointsMaterial;
      m.size = (o.base?.dense ?? 1) * ptSize;
      const want = byFrame && o.frameColor ? o.frameColor : o.denseColor;
      if (want && o.dense.geometry.getAttribute("color") !== want) {
        o.dense.geometry.setAttribute("color", want);
        m.vertexColors = true;
        m.color.set(0xffffff);
        m.needsUpdate = true;
      }
    }
    if (o.cams) o.cams.visible = showCams;
    o.frustumScaleFn?.(frustumScale);
  }, [showSparse, showDense, showCams, byFrame, ptSize, frustumScale, data, camGeo]);

  return (
    <div>
      <div className="glbview" style={{ height: 520 }} ref={holderRef} />
      {failed && <div className="note" style={{ color: "var(--err)" }}>scene render failed: {failed}</div>}
    </div>
  );
}
