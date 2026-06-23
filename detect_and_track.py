#!/usr/bin/env python3
import torch
import os
import sys
import argparse
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(
        description="Demo YOLO 2D Detection, Classification, and Tracking CLI Utility"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="detect",
        choices=["detect", "classify", "track"],
        help="Task to perform: 'detect' (2D detection), 'classify' (image classification), or 'track' (object tracking)."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model file/name (e.g. yolo11n.pt, yolo11n-cls.pt, yolov8n.pt). If not specified, a default model is selected based on the task."
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to source image, video, directory of images, or webcam index (e.g. 0)."
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Object confidence threshold for detection/tracking (default: 0.25)."
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="Intersection over Union (IoU) threshold for NMS (default: 0.45)."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Device to run model on (e.g. cpu, cuda:0, cuda, mps). Default is auto-select."
    )
    parser.add_argument(
        "--save",
        action="store_true",
        default=True,
        help="Save annotated results to disk (default: True)."
    )
    parser.add_argument(
        "--no-save",
        dest="save",
        action="store_false",
        help="Do not save annotated results."
    )
    parser.add_argument(
        "--show",
        action="store_true",
        default=False,
        help="Show results in real-time OpenCV window (default: False)."
    )
    
    args = parser.parse_args()
    
    # 1. Select default model if not provided
    if not args.model:
        if args.task == "classify":
            args.model = "yolo11n-cls.pt"
        else:
            args.model = "yolo11n.pt"  # nano model is fast and downloads quickly
            
    print("=" * 60)
    print(" YOLO 2D Detection & Tracking Showcase CLI ")
    print("=" * 60)
    print(f"Task:      {args.task}")
    print(f"Model:     {args.model}")
    print(f"Source:    {args.source}")
    print(f"Conf:      {args.conf}")
    print(f"IoU:       {args.iou}")
    print(f"Device:    {args.device if args.device else 'Auto-detect'}")
    print(f"Save:      {args.save}")
    print(f"Show Live: {args.show}")
    print("-" * 60)

    # 2. Check if source is a webcam index
    source = args.source
    if source.isdigit():
        source = int(source)
    else:
        # Check if source file/folder exists
        if not os.path.exists(source):
            print(f"[ERROR] Source path '{source}' does not exist.")
            sys.exit(1)
            
    # 3. Load YOLO model
    print(f"[INFO] Loading YOLO model '{args.model}'...")
    try:
        model = YOLO(args.model)
    except Exception as e:
        print(f"[ERROR] Failed to load model '{args.model}': {e}")
        sys.exit(1)

    # 4. Perform Task
    print(f"[INFO] Starting inference task: {args.task}...")
    try:
        if args.task == "detect":
            results = model.predict(
                source=source,
                conf=args.conf,
                iou=args.iou,
                device=args.device or None,
                save=args.save,
                show=args.show
            )
        elif args.task == "track":
            results = model.track(
                source=source,
                conf=args.conf,
                iou=args.iou,
                device=args.device or None,
                save=args.save,
                show=args.show,
                persist=True  # persist tracks across frames
            )
        elif args.task == "classify":
            results = model.predict(
                source=source,
                conf=args.conf,
                device=args.device or None,
                save=args.save,
                show=args.show
            )
        else:
            raise ValueError(f"Unknown task: {args.task}")
            
    except Exception as e:
        print(f"[ERROR] An error occurred during inference: {e}")
        sys.exit(1)
        
    # 5. Summarize and report results
    print("-" * 60)
    print("[SUCCESS] Inference completed!")
    
    if results and len(results) > 0:
        # Get save directory if results were saved
        if args.save:
            save_dir = getattr(results[0], "save_dir", "runs")
            print(f"[INFO] Annotated outputs saved to: {save_dir}")
            
        # Summary of the first frame/image
        print("\n--- Summary of first processed frame/image ---")
        res = results[0]
        
        if args.task == "classify":
            if res.probs is not None:
                top1_idx = res.probs.top1
                top1_conf = float(res.probs.top1conf)
                top5_indices = res.probs.top5
                top5_confs = [float(c) for c in res.probs.top5conf]
                
                print(f"Top-1 Prediction: Index {top1_idx} ({res.names[top1_idx]}) with conf {top1_conf:.4f}")
                print("Top-5 Predictions:")
                for idx, conf in zip(top5_indices, top5_confs):
                    print(f"  - {res.names[idx]}: {conf:.4f}")
        else:
            # Detection or Tracking
            boxes = res.boxes
            if boxes is not None:
                print(f"Total objects detected: {len(boxes)}")
                
                # Check for tracks
                has_ids = hasattr(boxes, "id") and boxes.id is not None
                
                # Count classes
                class_counts = {}
                for i in range(len(boxes)):
                    cls_id = int(boxes.cls[i])
                    cls_name = res.names[cls_id]
                    class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
                    
                    if i < 5:  # print first 5 objects
                        conf_val = float(boxes.conf[i])
                        box_coords = boxes.xyxy[i].tolist()
                        track_id_str = f", Track ID: {int(boxes.id[i])}" if has_ids else ""
                        print(f"  Obj {i+1}: {cls_name} (Conf: {conf_val:.2f}{track_id_str}) | Box: {[round(c, 1) for c in box_coords]}")
                
                if len(boxes) > 5:
                    print(f"  ... and {len(boxes) - 5} more objects.")
                    
                print("\nClass count breakdown:")
                for cls_name, count in class_counts.items():
                    print(f"  - {cls_name}: {count}")
    else:
        print("[WARNING] No results returned.")
    print("=" * 60)

if __name__ == "__main__":
    main()
