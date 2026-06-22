"""
Evaluate YOLOv8 object detector on the test split.
Reports: mAP@50, mAP@50-95, Precision, Recall.

Usage:
    conda run -n mot python scripts/eval_detector.py \
        --weights model_obj_detector/best.pt \
        --data    Dataset/data.yaml \
        --imgsz   640 \
        --batch   16
"""

import argparse
import tempfile
import yaml
from pathlib import Path
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="model_obj_detector/best.pt")
    p.add_argument("--data",    default="Dataset/data.yaml")
    p.add_argument("--imgsz",   type=int, default=640)
    p.add_argument("--batch",   type=int, default=16)
    p.add_argument("--split",   default="test",
                   choices=["train", "val", "test"],
                   help="Dataset split to evaluate on")
    p.add_argument("--conf",    type=float, default=0.001,
                   help="Confidence threshold (low value = standard COCO eval)")
    p.add_argument("--iou",     type=float, default=0.6,
                   help="NMS IoU threshold")
    p.add_argument("--device",  default="",
                   help="Device: '' = auto, '0' = GPU 0, 'cpu' = CPU")
    return p.parse_args()


def main():
    args = parse_args()

    model = YOLO(args.weights)

    # Resolve data.yaml to absolute paths so Ultralytics finds the images
    # regardless of its internal `datasets_dir` setting.
    data_yaml_path = Path(args.data).resolve()
    data_dir = data_yaml_path.parent
    with open(data_yaml_path) as f:
        data_cfg = yaml.safe_load(f)

    # Make all split paths absolute
    for key in ("train", "val", "test"):
        if key in data_cfg and not Path(data_cfg[key]).is_absolute():
            data_cfg[key] = str(data_dir / data_cfg[key])

    # Write a temp yaml with absolute paths
    tmp_yaml = tempfile.NamedTemporaryFile(
        suffix=".yaml", mode="w", delete=False
    )
    yaml.dump(data_cfg, tmp_yaml)
    tmp_yaml.flush()
    abs_data_yaml = tmp_yaml.name

    results = model.val(
        data=abs_data_yaml,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        verbose=True,
    )

    box = results.box

    map50     = box.map50
    map50_95  = box.map
    precision = box.mp      # mean precision across classes
    recall    = box.mr      # mean recall across classes

    print("\n" + "=" * 50)
    print("  YOLOv8 Detector Evaluation Results")
    print("=" * 50)
    print(f"  Split        : {args.split}")
    print(f"  Weights      : {args.weights}")
    print(f"  Image size   : {args.imgsz}")
    print(f"  Conf thresh  : {args.conf}")
    print(f"  NMS IoU      : {args.iou}")
    print("-" * 50)
    print(f"  Precision    : {precision:.4f}")
    print(f"  Recall       : {recall:.4f}")
    print(f"  mAP@50       : {map50:.4f}")
    print(f"  mAP@50-95    : {map50_95:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
