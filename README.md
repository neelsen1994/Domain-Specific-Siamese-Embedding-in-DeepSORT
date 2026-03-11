# Siamese Embedding Model for DeepSORT

## Requirements

- Python 3.7+
- `requirements.txt`
- Recommended conda env: **`mot`** (TF 2.11, OpenCV 4.10)

---

## Proper ReID Training Pipeline (`scripts/`)

A production-grade training pipeline for turkey re-identification,
producing a DeepSORT-compatible frozen `.pb` model.

### Model

DeepSORT-style compact CNN (~223k params):

```
Input (128×64×3, raw BGR float32 [0,255])
  ÷255 normalisation (inside model)
  Conv32 → BN → ReLU → Conv32 → BN → ReLU → MaxPool
  ResBlock(32) × 2
  Conv64 → BN → ReLU → MaxPool
  ResBlock(64) × 2
  GlobalAveragePool → Dropout → Dense(128) → L2Norm
Output: 128-D L2-normalised embedding  ("features:0")
```

Normalization is embedded inside the model so the `.pb` accepts raw BGR
`[0,255]` patches, exactly as DeepSORT's inference code provides.

---

### 1 — Training

```bash
conda activate mot
python scripts/train_reid.py \
    --dataset_path dataset_siam_21 \
    --output_dir   runs/turkey_reid \
    --epochs 150   \
    --P 16 --K 4   \
    --lr 1e-3      \
    --seed 42
```

**Key flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--P` | 16 | Identities per batch |
| `--K` | 4  | Images per identity → `batch_size = P×K` |
| `--margin` | 0.3 | Triplet loss margin |
| `--epochs` | 150 | Max epochs |
| `--patience` | 25 | Early-stopping patience |
| `--no_augment` | off | Disable augmentation (debug) |
| `--resume` | – | Path to `.h5` weights to resume from |
| `--split_manifest` | – | Reuse an existing split JSON |

**Output files:**

```
runs/turkey_reid/
  split_manifest.json          deterministic train/val/test split
  checkpoints/best_weights.h5  best weights by val rank-1
  best_model.pb                frozen .pb for DeepSORT
  best_model_keras.h5          Keras weights alias
  training_log.json            per-epoch loss + metrics
```

---

### 2 — Evaluation (CMC + mAP)

```bash
python scripts/eval_reid.py \
    --weights  runs/turkey_reid/best_model_keras.h5 \
    --manifest runs/turkey_reid/split_manifest.json \
    --show_distances # optional
```

Evaluation with .pb model

```bash
python scripts/eval_reid_pb.py \
    --model_pb runs/turkey_reid/best_model.pb \
    --manifest runs/turkey_reid/split_manifest.json \
    --show_distances # optional
```

Cross-split evaluation (query=test, gallery=train+val):

```bash
python scripts/eval_reid.py \
    --weights  runs/turkey_reid/best_model_keras.h5 \
    --manifest runs/turkey_reid/split_manifest.json \
    --query_split test \
    --gallery_splits train val \
    --show_distances # optional
```

Exhaustive pairwise testing with plots:

```bash
python feat_vectest_exhaustive_v2.py --pb_path  runs/turkey_reid/best_model.pb --manifest runs/turkey_reid/split_manifest.json --output_dir embruns
```

---

### 3 — Export `.pb` (standalone)

```bash
python scripts/export_pb.py \
    --weights runs/turkey_reid/best_model_keras.h5 \
    --output  runs/turkey_reid/best_model.pb
```

---

### 4 — Sanity check

Verifies the `.pb` loads correctly and that same-ID similarity > diff-ID:

```bash
python scripts/sanity_check.py \
    --pb_path runs/turkey_reid/best_model.pb \
    --image1  dataset_siam_21/object_1/frame_100.jpg \
    --image2  dataset_siam_21/object_1/frame_200.jpg \
    --image3  dataset_siam_21/object_2/frame_100.jpg
```

---

### 5 — Plug `.pb` into DeepSORT

In `tracker.py`, change the model path:

```python
# before
encoder_model_filename = 'model_feature_extractor/mars-small128.pb'
# after
encoder_model_filename = 'runs/turkey_reid/best_model.pb'
```

The `generate_detections.py` has been patched to use `name=""` when loading
the graph (instead of `name="net"`), so `images:0` and `features:0` resolve
correctly. No other changes needed.

You may want to tune `max_cosine_distance` in `tracker.py` (currently `0.08`)
after evaluating the new model's embedding distribution.

---

