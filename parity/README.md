# Parity fixtures

`dump.py` runs the mlx-vlm 0.7.3 SAM 3.1 reference on fixed inputs and saves inputs, intermediate tensors and outputs. The Swift parity suites in `Tests/SAM31Tests/Parity/` compare against these files. Python is used only here, never at runtime.

## Setup

```bash
uv venv .venv --python 3.12 && uv pip install -r parity/requirements.txt
hf download mlx-community/sam3.1-bf16 --local-dir weights/sam3.1-bf16
```

The checkpoint in `sam3.1-bf16` is stored as fp32. The first run also downloads the CLIP tokenizer (`openai/clip-vit-base-patch32`) into the Hugging Face cache.

## Dump

```bash
.venv/bin/python parity/dump.py --weights weights/sam3.1-bf16 \
  --video clip.mp4 \
  --out parity/fixtures
```

This takes about a minute on an M4 Pro and writes about 860 MB. `parity/fixtures/` is gitignored.

## Run the parity tests

Parity suites read `SAM31_WEIGHTS` and `SAM31_FIXTURES`. If either is unset, the suites are skipped rather than failed. Under xcodebuild, pass them with the `TEST_RUNNER_` prefix:

```bash
TEST_RUNNER_SAM31_WEIGHTS=$PWD/weights/sam3.1-bf16 TEST_RUNNER_SAM31_FIXTURES=$PWD/parity/fixtures \
  xcodebuild test -scheme sam31-swift-Package -destination 'platform=macOS' -derivedDataPath .build/xcode -quiet
```

## Fixture keys

Shapes and dtypes for every key are listed in `manifest.json` under `files`. Coordinates are in 1008×1008 model-input space. The video frames are 1920×1080 RGB.

| File                      | Keys                                                                                                                                                                                                                                 |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `inputs.safetensors`      | `imageA` (frame 0, uint8 HWC), `imageB` (synthetic 1280×720 gradient/checker), `clip` (frames 0–9, uint8)                                                                                                                            |
| `preprocess.safetensors`  | `pixel_values_A`, `pixel_values_B` (1,1008,1008,3)                                                                                                                                                                                   |
| `vision.safetensors`      | `vit.patch_embed` (PatchEmbeddings output before pos-embed and layernorm, (1,5184,1024)), `vit.block.{0,8,16,31}`, `vit.out`, `neck.{det,interactive,propagation}.{0,1,2}` (288/144/72), `neck.det_pos.{0,1,2}`                      |
| `tokenizer.json`          | list of `{text, input_ids, attention_mask}`, one entry per line of `tokenizer_corpus.txt` after `ESC:`/`RAW:` handling                                                                                                               |
| `text.safetensors`        | `{person,the_dancer,city_skyline}.{input_ids,attention_mask}` (int64), `.hidden` (text encoder, 1024-d), `.embeds` (projected, 256-d)                                                                                                |
| `detect.safetensors`      | `detr.enc`, `detr.dec.{hs,ref_boxes,presence}`, `det.scoring`, `det.mask_decoder.{pred_masks,semantic_seg}`, `final.{boxes,scores,masks}` (source-pixel boxes, uint8 masks at 1080×1920), all for "person" on image A, threshold 0.3 |
| `interactive.safetensors` | for `click1`, `click2` and `box`: `<tag>.out.{pred_masks,pred_masks_high_res,object_score_logits,obj_ptr}`, `<tag>.sam.prompt.{sparse,dense}`, `<tag>.sam.decoder.{masks,iou_pred,sam_tokens_out,object_score_logits}`               |
| `track.safetensors`       | `mem.encoder.first.{features,pos_enc}`, `add_mask`, `f0.*`, `f1..f9.{masks,object_score_logits}`, `f5.pre_refine.{masks,object_score_logits}`, `mask_prompt.out.{pred_masks,pred_masks_high_res,object_score_logits,obj_ptr}` (fresh 1-object `track_step` with `add_mask` as the mask prompt) |
| `manifest.json`           | `mlx_vlm` version, `video`, `prompts`, `add_mask_source`, `hook_calls`, `tokenizer_entries`, `sanity`, `files`                                                                                                                       |

### Notes on specific keys

- `mem.encoder.first.features` is the raw `MultiplexMemoryEncoder` output. It is taken **before** `_encode_new_memory` adds `no_obj_embed_spatial`.
- DETR consumes only `neck.det_pos.2`, flattened to (1, 5184, 256). The other two scales are dumped for the pos-encoding unit test only.
- `final.boxes` are in source pixels (1920×1080), clipped to [0, max(H, W)]. Swift tests compare them with a tolerance of 1 px × W/1008 (the 1 px gate in 1008 space).

### Prompts (`manifest.prompts`)

- `pos` is the centre of detection 0's box. `neg` is (302.4, 302.4).
- `box` is detection 0's box.
- `refine` is `pos` shifted +20 in y.
- `obj2` is the click that made `add_mask`.
- Labels: positive = 1, negative = 0, box corners = 2 (top-left) and 3 (bottom-right).
- Interactive cases, each on a fresh 1-object state on frame 0 with `is_init_cond_frame=True` and `num_frames=10`:
  - `click1` = [pos]
  - `click2` = [pos, neg]
  - `box` = two corner points

### Tracking scenario (the Swift test must replay it exactly)

Every tracking call passes `num_frames=10` (ruling R8). `_get_tpos_enc` depends on it through `max_abs_pos = min(num_frames, 16)`. The dump never uses `Model.track_step`, because that passes `num_frames=None`, which becomes `frame_idx + 1`.

| Call | num_frames |
|---|---|
| f0 click2 `track_step` | 10 |
| obj2 click `track_step` (separate state) | 10 |
| f0 `add_mask_prompt` | n/a (merge path; no memory attention) |
| f1–f4, f6–f9 `tracker.propagate` | 10 |
| f5 `tracker.propagate` (pre-refine) | 10 |
| f5 refine `tracker.track_step` | 10 |

1. `st = init_state(1)`. Run `track_step` on f0 with `is_init_cond_frame=True` and the click2 points.
   - `mem.encoder.first` records the memory encoder of this step (1 object).
2. Build the second object's mask (ruling R7):
   - If "person" gives 2 or more detections, use detection 1's mask.
   - Otherwise, use a separate 1-object state with one positive click at `prompts.obj2` on f0. Take `add_mask = (pred_masks_high_res[0,0] > 0)` as float32 1008×1008.
   - `manifest.add_mask_source` says which path ran. Currently it is the click path, because there is 1 detection.
3. `add_mask_prompt(st, 0, ff0, add_mask[None])`. f0 is already tracked, so the mask is merged as object 1. Its output is `f0.*`.
4. Propagate frames 1–4 with `tracker.propagate(st, i, model.tracker_frame_features(bb, need_interactive=False), num_frames=10)`.
5. On f5, build full frame features (interactive and propagation):
   - Run `tracker.propagate(st, 5, ffi, num_frames=10)` with those full features and record the result as `f5.pre_refine.*`.
   - Refine (ruling R6): call `tracker.track_step` again on f5 with `is_init_cond_frame=False`, `point_inputs=[[refine]]` with label [[1]], `objects_to_interact=[0]`, `num_frames=10`, and **no** `prev_sam_mask_logits`. This is the `propagation_and_interaction` mode.
   - The second call does not raise, so nothing is popped from the state. Memory attention reads only frames before 5, so the stored pre-refine output does not affect it.
   - The refine output replaces `non_cond_frame_outputs[5]` and is recorded as `f5.*`.
6. Propagate frames 6–9 as in step 4.

## Deviations from the plan's dump.py

- The refine no longer passes `prev_sam_mask_logits`.
  - That argument selects `interaction_only` mode, which runs only the interacted object and fails in `MultiplexState.mux` on a 2-object session: `AssertionError: num_valid=1 != total_valid_entries=2`.
  - See ruling R6.
- `add_mask` now comes from a click, not the plan's synthetic square (ruling R7).
- The memory-encoder key names changed: `mem.encoder.first.0` became `mem.encoder.first.features`, and `mem.encoder.first.1` became `mem.encoder.first.pos_enc`.
- New keys: `f0.masks`, `f0.object_score_logits`, `f5.pre_refine.object_score_logits`.
- Hooks are removed after each stage, and each stage keeps the first call. `manifest.hook_calls` shows that every hooked module in the vision, detect and interactive stages ran exactly once per stage. In the track stage the memory-encoder hook stays on for the whole stage and runs 13 times: f0 click2, the obj2 click, the add_mask re-encode, f1–f4, f5 pre-refine, f5 refine and f6–f9. The fixture keeps the first call.
- `flat()` skips `None`, casts float64 to float32 and bool to uint8, and raises on unknown types.
- Tracking uses `tracker.propagate(..., num_frames=10)` instead of `Model.track_step` (ruling R8).
- Corpus lines with trailing whitespace are written with `ESC:` and `\x20` so formatters cannot strip them. The resulting texts are unchanged.
- Box prompt coordinates are passed as Python floats, because `mx.array` rejects `np.float32` scalars.
