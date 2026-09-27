"""Dump fixed inputs + intermediate tensors from mlx-vlm SAM 3.1 for Swift parity tests.

Usage: .venv/bin/python parity/dump.py --weights weights/sam3.1-bf16 --video <mp4> --out parity/fixtures
Hooks are instance-level class swaps; mlx-vlm source is never modified.
Every hooked module records each call; the fixtures keep the first call of each stage.
"""
import argparse, json
from pathlib import Path

import cv2
import mlx.core as mx
import numpy as np
from PIL import Image

import mlx_vlm
from mlx_vlm.utils import load_model
from mlx_vlm.models.sam3_1.processing_sam3_1 import Sam31Processor
from mlx_vlm.models.sam3.generate import Sam3Predictor
from mlx_vlm.models.sam3_1.generate import _get_backbone_features, _detect_with_backbone

CAP: dict = {}
HOOKED: list = []


def hook(module, name, pick=lambda o: o):
    """Record module outputs on every call as CAP[name] = [out0, out1, ...]."""
    cls = module.__class__

    def call(self, *a, **k):
        out = cls.__call__(self, *a, **k)
        CAP.setdefault(name, []).append(pick(out))
        return out

    module.__class__ = type(cls.__name__ + "Hooked", (cls,), {"__call__": call})
    HOOKED.append((module, cls))


def unhook_all():
    """Restore every hooked module's class so later stages do not record into CAP."""
    for module, cls in HOOKED:
        module.__class__ = cls
    HOOKED.clear()


def first():
    """Keep the first recorded call per hook; also report how many calls happened."""
    return {k: v[0] for k, v in CAP.items()}, {k: len(v) for k, v in CAP.items()}


def flat(prefix, obj, out):
    """Flatten arrays / lists / tuples / dicts into out[prefix.i.key] = array. None is skipped."""
    if obj is None:
        return
    if isinstance(obj, mx.array):
        out[prefix] = obj
    elif isinstance(obj, np.ndarray):
        if obj.dtype == np.float64:
            obj = obj.astype(np.float32)
        elif obj.dtype == np.bool_:
            obj = obj.astype(np.uint8)
        out[prefix] = mx.array(obj)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            flat(f"{prefix}.{i}", v, out)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            flat(f"{prefix}.{k}", v, out)
    elif isinstance(obj, bool):
        out[prefix] = mx.array(obj)
    elif isinstance(obj, int):
        out[prefix] = mx.array(obj, dtype=mx.int32)
    elif isinstance(obj, float):
        out[prefix] = mx.array(obj, dtype=mx.float32)
    else:
        raise TypeError(f"{prefix}: cannot save {type(obj).__name__}")


def save(path, d, manifest, calls=None):
    arrs = {}
    for k, v in d.items():
        flat(k, v, arrs)
    mx.eval(list(arrs.values()))
    mx.save_safetensors(str(path), arrs)
    manifest["files"][path.name] = {k: [list(v.shape), str(v.dtype)] for k, v in arrs.items()}
    if calls:
        manifest.setdefault("hook_calls", {})[path.name] = calls
    CAP.clear()
    print(f"  wrote {path.name}: {len(arrs)} arrays", flush=True)


def coverage(logits):
    return float((np.array(logits.astype(mx.float32)) > 0).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    manifest = {"mlx_vlm": mlx_vlm.__version__, "files": {}}
    sanity = {}

    model = load_model(Path(args.weights))
    proc = Sam31Processor.from_pretrained(args.weights)
    pred = Sam3Predictor(model, proc, score_threshold=0.3)
    det, trk = model.detector_model, model.tracker_model

    # ---- inputs ----
    cap = cv2.VideoCapture(args.video)
    frames = []
    for _ in range(10):
        ok, f = cap.read(); assert ok
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    H, W = frames[0].shape[:2]
    yy, xx = np.mgrid[0:720, 0:1280]
    synth = np.stack([(xx * 255 // 1279), (yy * 255 // 719), (((xx // 40) + (yy // 40)) % 2) * 255], -1).astype(np.uint8)
    Image.fromarray(frames[0]).save(out / "imageA.png")
    Image.fromarray(synth).save(out / "imageB.png")
    manifest["video"] = {"path": str(args.video), "width": W, "height": H, "frames": len(frames)}
    save(out / "inputs.safetensors", {"imageA": frames[0], "imageB": synth, "clip": np.stack(frames)}, manifest)

    # ---- preprocess ----
    pvA = mx.array(proc.preprocess_image(Image.fromarray(frames[0]))["pixel_values"])
    pvB = mx.array(proc.preprocess_image(Image.fromarray(synth))["pixel_values"])
    save(out / "preprocess.safetensors", {"pixel_values_A": pvA, "pixel_values_B": pvB}, manifest)

    # ---- vision (image A) ----
    # vit.patch_embed = PatchEmbeddings output (B, N, C), before pos-embed add and pre-layernorm.
    # vit.block.i = output of backbone.layers[i] (B, H, W, C). Backbone runs exactly once here.
    bb = det.vision_encoder.backbone
    hook(bb.embeddings, "vit.patch_embed")
    for i in (0, 8, 16, len(bb.layers) - 1):
        hook(bb.layers[i], f"vit.block.{i}")
    feats = _get_backbone_features(model, pvA)
    unhook_all()
    d_, i_, p_ = det.vision_encoder.neck(feats, need_det=True, need_interactive=True, need_propagation=True)
    pos = [det._pos_enc(f) for f in d_]
    vis, calls = first()
    vis.update({"vit.out": feats, "neck.det": d_, "neck.interactive": i_, "neck.propagation": p_, "neck.det_pos": pos})
    save(out / "vision.safetensors", vis, manifest, calls)

    # ---- tokenizer + text ----
    lines = (Path(__file__).parent / "tokenizer_corpus.txt").read_text(encoding="utf-8").splitlines()
    corpus = []
    for ln in lines:
        if ln.startswith("ESC:"): ln = ln[4:].encode().decode("unicode_escape")
        elif ln.startswith("RAW:"): ln = ln[4:]
        corpus.append(ln)
    toks = []
    for s in corpus:
        e = proc.preprocess_text(s)
        toks.append({"text": s, "input_ids": e["input_ids"][0].tolist(), "attention_mask": e["attention_mask"][0].tolist()})
    (out / "tokenizer.json").write_text(json.dumps(toks, ensure_ascii=False, indent=0), encoding="utf-8")
    manifest["tokenizer_entries"] = len(toks)
    txt = {}
    for name in ("person", "the dancer", "city skyline"):
        e = proc.preprocess_text(name)
        ids, am = mx.array(e["input_ids"]), mx.array(e["attention_mask"])
        key = name.replace(" ", "_")
        txt[f"{key}.input_ids"] = ids
        txt[f"{key}.attention_mask"] = am
        txt[f"{key}.hidden"] = det.text_encoder(ids, am)
        txt[f"{key}.embeds"] = det.get_input_embeddings(ids, am)
    save(out / "text.safetensors", txt, manifest)

    # ---- detect ("person" on A) ----
    # Each hooked module runs exactly once for one prompt (see hook_calls in manifest).
    hook(det.detr_encoder, "detr.enc")
    hook(det.detr_decoder, "detr.dec", lambda o: {"hs": o[0], "ref_boxes": o[1], "presence": o[2]})
    hook(det.dot_product_scoring, "det.scoring")
    hook(det.mask_decoder, "det.mask_decoder", lambda o: {"pred_masks": o["pred_masks"], "semantic_seg": o.get("semantic_seg")})
    res = _detect_with_backbone(pred, feats, ["person"], (W, H), 0.3)
    unhook_all()
    detd, calls = first()
    detd.update({"final.boxes": res.boxes.astype(np.float32), "final.scores": res.scores.astype(np.float32),
                 "final.masks": res.masks.astype(np.uint8)})
    save(out / "detect.safetensors", detd, manifest, calls)
    sanity["detect"] = {"count": int(len(res.scores)), "scores": [round(float(s), 4) for s in res.scores],
                        "boxes": [[round(float(x), 1) for x in b] for b in res.boxes],
                        "mask_coverage": [round(float(m.mean()), 4) for m in res.masks]}

    # ---- interactive (clicks / box on A) ----
    ff = model.tracker_frame_features(feats)
    S = 1008
    person = res.boxes[0] if len(res.scores) else np.array([W * 0.6, H * 0.4, W * 0.85, H * 0.95])
    px, py = float((person[0] + person[2]) / 2 * S / W), float((person[1] + person[3]) / 2 * S / H)
    nx, ny = 0.3 * S, 0.3 * S
    box = np.array([person[0] * S / W, person[1] * S / H, person[2] * S / W, person[3] * S / H], np.float32)
    manifest["prompts"] = {"pos": [px, py], "neg": [nx, ny], "box": box.tolist(), "refine": [px, py + 20],
                           "space": "1008x1008 model input", "labels": {"pos": 1, "neg": 0, "box_tl": 2, "box_br": 3}}
    hook(trk.interactive_sam_prompt_encoder, "sam.prompt", lambda o: {"sparse": o[0], "dense": o[1]})
    hook(trk.interactive_sam_mask_decoder, "sam.decoder")
    inter = {}
    cases = {
        "click1": ([[px, py]], [1]),
        "click2": ([[px, py], [nx, ny]], [1, 0]),
        "box": ([[float(box[0]), float(box[1])], [float(box[2]), float(box[3])]], [2, 3]),
    }
    all_calls = {}
    sanity["interactive"] = {}
    for tag, (pts, lbl) in cases.items():
        st = trk.init_state(1)
        o = trk.track_step(st, frame_idx=0, is_init_cond_frame=True, frame_features=ff,
                           point_inputs={"point_coords": mx.array([pts], dtype=mx.float32),
                                         "point_labels": mx.array([lbl], dtype=mx.int32)}, num_frames=10)
        for k in ("pred_masks", "pred_masks_high_res", "object_score_logits", "obj_ptr"):
            if k in o: inter[f"{tag}.out.{k}"] = o[k]
        c, n = first()
        for k, v in c.items(): inter[f"{tag}.{k}"] = v
        all_calls[tag] = n
        CAP.clear()
        sanity["interactive"][tag] = {"coverage": round(coverage(o["pred_masks_high_res"]), 4),
                                      "obj_score": round(float(o["object_score_logits"].reshape(-1)[0].item()), 3)}
    unhook_all()
    save(out / "interactive.safetensors", inter, manifest, all_calls)

    # ---- tracking: obj0 by click2 on f0, obj1 by add_mask_prompt on f0, propagate 1..4,
    #      refine obj0 on f5 with a positive click, propagate 6..9 ----
    # num_frames (ruling R8): every tracking call passes num_frames=10 explicitly. Model.track_step is
    # not used because it calls propagate with num_frames=None (-> frame_idx+1), which changes
    # _get_tpos_enc's max_abs_pos.
    # mem.encoder.first = first memory-encoder call = frame 0 click2 track_step (1 object, before add_mask),
    # the raw MultiplexMemoryEncoder output (before _encode_new_memory adds no_obj_embed_spatial).
    # The hook stays on for the whole stage so hook_calls records the total count.
    hook(trk.memory_encoder, "mem.encoder", lambda o: {"features": o[0], "pos_enc": o[1]})
    st = trk.init_state(1)
    trk.track_step(st, frame_idx=0, is_init_cond_frame=True, frame_features=ff,
                   point_inputs={"point_coords": mx.array([cases["click2"][0]], dtype=mx.float32),
                                 "point_labels": mx.array([cases["click2"][1]], dtype=mx.int32)}, num_frames=10)
    first_mem = {k: v for k, v in CAP["mem.encoder"][0].items()}
    tr = {"mem.encoder.first": first_mem}
    # Second object (ruling R7): a detection mask only if "person" gives >= 2 hits; otherwise the
    # interactive mask of a single positive click at the `neg` point, from a separate 1-object state
    # on frame 0, thresholded at > 0 to a {0,1} float32 1008x1008 mask.
    if len(res.scores) > 1:
        m1 = np.array(Image.fromarray(res.masks[1] * 255).resize((S, S), Image.BILINEAR)).astype(np.float32) / 255
        manifest["add_mask_source"] = "detection 1 mask (uint8*255) PIL-bilinear resized to 1008, /255"
    else:
        ox, oy = nx, ny
        st2 = trk.init_state(1)
        o2 = trk.track_step(st2, frame_idx=0, is_init_cond_frame=True, frame_features=ff,
                            point_inputs={"point_coords": mx.array([[[ox, oy]]], dtype=mx.float32),
                                          "point_labels": mx.array([[1]], dtype=mx.int32)}, num_frames=10)
        m1 = np.array((o2["pred_masks_high_res"][0, 0] > 0).astype(mx.float32))
        del st2, o2
        manifest["prompts"]["obj2"] = [float(ox), float(oy)]
        manifest["add_mask_source"] = ("pred_masks_high_res[0,0] > 0 (float32) of a 1-object interactive "
                                       "track_step on frame 0 with one positive click at prompts.obj2")
    sanity["add_mask_coverage"] = round(float(m1.mean()), 4)
    tr["add_mask"] = mx.array(m1)
    f0 = trk.add_mask_prompt(st, 0, ff, mx.array(m1)[None])
    tr["f0.masks"] = f0["pred_masks_high_res"]
    tr["f0.object_score_logits"] = f0["object_score_logits"]
    sanity["track"] = {"f0": [round(coverage(m), 4) for m in f0["pred_masks_high_res"]]}
    for i in range(1, 10):
        pv = mx.array(proc.preprocess_image(Image.fromarray(frames[i]))["pixel_values"])
        b = _get_backbone_features(model, pv)
        if i == 5:
            ffi = model.tracker_frame_features(b)  # interactive + propagation (refine needs both)
            # Refine (ruling R6): propagate f5 normally first (recorded as f5.pre_refine.*), then call
            # track_step again on f5 in propagation_and_interaction mode: objects_to_interact=[0], a
            # positive click at prompts.refine, NO prev_sam_mask_logits. Memory attention only reads
            # frames < 5, so the stored f5 output does not feed the second call; the refine output
            # replaces state.non_cond_frame_outputs[5]. It is recorded as f5.masks / f5.object_score_logits.
            prev = trk.propagate(st, 5, ffi, num_frames=10)
            tr["f5.pre_refine.masks"] = prev["pred_masks_high_res"]
            tr["f5.pre_refine.object_score_logits"] = prev["object_score_logits"]
            mx.eval(prev["pred_masks_high_res"], prev["object_score_logits"])
            sanity["track"]["f5.pre_refine"] = [round(coverage(m), 4) for m in prev["pred_masks_high_res"]]
            o = trk.track_step(st, frame_idx=5, is_init_cond_frame=False, frame_features=ffi,
                               point_inputs={"point_coords": mx.array([[[px, py + 20]]], dtype=mx.float32),
                                             "point_labels": mx.array([[1]], dtype=mx.int32)},
                               objects_to_interact=[0], num_frames=10)
        else:
            o = trk.propagate(st, i, model.tracker_frame_features(b, need_interactive=False), num_frames=10)
        tr[f"f{i}.masks"] = o["pred_masks_high_res"]
        tr[f"f{i}.object_score_logits"] = o["object_score_logits"]
        mx.eval(tr[f"f{i}.masks"], tr[f"f{i}.object_score_logits"])
        sanity["track"][f"f{i}"] = [round(coverage(m), 4) for m in o["pred_masks_high_res"]]
    mem_calls = len(CAP.get("mem.encoder", []))
    unhook_all()
    save(out / "track.safetensors", tr, manifest, {"mem.encoder": mem_calls})

    manifest["sanity"] = sanity
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(sanity, indent=1))
    print("fixtures written to", out)


if __name__ == "__main__":
    main()
