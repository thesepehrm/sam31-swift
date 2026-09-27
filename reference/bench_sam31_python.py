import time, sys
import numpy as np, mlx.core as mx, cv2
from PIL import Image
from mlx_vlm.utils import load_model
from mlx_vlm.models.sam3_1.processing_sam3_1 import Sam31Processor
from mlx_vlm.models.sam3.generate import Sam3Predictor
from mlx_vlm.models.sam3_1.generate import _get_backbone_features

MP = "./sam3.1-bf16"
VIDEO = "clip.mp4"
N_FRAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 120

def T(label, fn, *a, **k):
    t = time.perf_counter(); r = fn(*a, **k); dt = (time.perf_counter() - t) * 1000
    print(f"{label:<44} {dt:8.1f} ms", flush=True); return r

def save_overlay(img_rgb, mask, path, color=(90, 138, 181)):
    m = cv2.resize(mask.astype(np.uint8), (img_rgb.shape[1], img_rgb.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    out = img_rgb.copy().astype(np.float32)
    out[m] = out[m] * 0.45 + np.array(color) * 0.55
    Image.fromarray(out.astype(np.uint8)).save(path)
    return m.mean()

model = T("load model", load_model, __import__("pathlib").Path(MP))
proc = Sam31Processor.from_pretrained(MP)
pred = Sam3Predictor(model, proc, score_threshold=0.3)

cap = cv2.VideoCapture(VIDEO)
frames = []
for _ in range(N_FRAMES):
    ok, f = cap.read()
    if not ok: break
    frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
H, W = frames[0].shape[:2]
print(f"video {W}x{H}, {len(frames)} frames")
img = Image.fromarray(frames[0])

# ---- text prompt (detector path) ----
r = T("text 'person' (cold)", pred.predict, img, text_prompt="person")
r = T("text 'person' (warm)", pred.predict, img, text_prompt="person")
print("  detections:", len(r.scores), np.round(r.scores, 2))
if len(r.scores): print("  text mask coverage:", round(save_overlay(frames[0], r.masks[0], "out-text.png"), 3))
r2 = T("text 'city skyline' (warm backbone? no cache)", pred.predict, img, text_prompt="city skyline")
print("  detections:", len(r2.scores))

# ---- box prompt ----
if len(r.scores):
    b = r.boxes[0]
    rb = T("box-guided (text 'object' + box)", pred.predict, img, text_prompt="object", boxes=np.array([b]))
    print("  box detections:", len(rb.scores))

# ---- click prompt (interactive tracker path) ----
inputs = proc.preprocess_image(img)
pv = mx.array(inputs["pixel_values"])
print("  pixel_values", pv.shape)
S = pv.shape[-2] if pv.shape[1] == 3 else pv.shape[1]
bb = T("backbone (frame analysis)", _get_backbone_features, model, pv)
ff = T("tracker frame features", model.tracker_frame_features, bb)

def click(points, labels, tag):
    st = model.tracker_model.init_state(1)
    pc = mx.array([[[float(x) * S / W, float(y) * S / H] for x, y in points]], dtype=mx.float32)
    pl = mx.array([labels], dtype=mx.int32)
    t = time.perf_counter()
    out = model.tracker_model.track_step(st, frame_idx=0, is_init_cond_frame=True, frame_features=ff,
                                         point_inputs={"point_coords": pc, "point_labels": pl}, num_frames=len(frames))
    mx.eval(out["pred_masks_high_res"])
    dt = (time.perf_counter() - t) * 1000
    m = np.array(out["pred_masks_high_res"][0, 0]) > 0
    cov = save_overlay(frames[0], m, f"out-click-{tag}.png")
    print(f"{'click ' + tag:<44} {dt:8.1f} ms   coverage {cov:.3f}", flush=True)
    return st, out

# person sits right-of-centre, lower third in this clip
px, py = (W * 0.73, H * 0.66)
if len(r.scores):
    x0, y0, x1, y1 = r.boxes[0]; px, py = ((x0 + x1) / 2, (y0 + y1) / 2)
click([(px, py)], [1], "1-cold")
click([(px, py)], [1], "1-warm")
st, out = click([(px, py), (W * 0.3, H * 0.3)], [1, 0], "2-pos-neg")

# ---- track the click through the clip (memory tracker) ----
t0 = time.perf_counter(); per = []
for i in range(1, len(frames)):
    t = time.perf_counter()
    p = mx.array(proc.preprocess_image(Image.fromarray(frames[i]))["pixel_values"])
    b = _get_backbone_features(model, p)
    o = model.track_step(st, b, i)
    per.append((time.perf_counter() - t) * 1000)
    if i == len(frames) - 1:
        save_overlay(frames[i], np.array(o["pred_masks_high_res"][0, 0]) > 0, "out-track-last.png")
tot = time.perf_counter() - t0
print(f"track {len(per)} frames: total {tot:.1f} s, median {np.median(per):.0f} ms/frame, p90 {np.percentile(per,90):.0f} ms, "
      f"= {len(per)/tot:.1f} fps (realtime would be 30)")
print("peak MLX memory GB:", round(mx.get_peak_memory() / 1e9, 2))
