import os
import time
import argparse
import json
from pathlib import Path
import multiprocessing as mp

import cv2
import numpy as np
from datasets import load_from_disk

from metrics import db_eval_iou, db_eval_boundary
from pycocotools import mask as cocomask

NUM_WORKERS = 64  # you can override via CLI if you want

# These will be populated in main()
hf_data_dict = None
video_root_dir = None


def decode_mask(mask, w=None, h=None):
    """
    Decode mask from various formats to numpy array.
    
    Args:
        mask: Mask in various formats (None, numpy array, COCO RLE)
        w: Width of the mask (used for None case)
        h: Height of the mask (used for None case)
        
    Returns:
        Decoded mask as a numpy array
    """
    if mask is None:
        return np.zeros((h, w), dtype=np.uint8)
    elif isinstance(mask, np.ndarray):
        return mask
    else:
        return cocomask.decode(mask)
        
def decode_masks(masks, w, h):
    """
    Decode a list of masks.
    
    Args:
        masks: List of masks in various formats
        w: Width of the masks
        h: Height of the masks
        
    Returns:
        List of decoded masks as numpy arrays
    """
    return np.array([decode_mask(mask, w, h) for mask in masks])

def eval_queue(q, rank, out_dict, predictions_dir):
    """
    Worker loop: consumes query_ids from queue, computes J and F, writes to out_dict.
    This mirrors the logic/style of the original MeViS script.
    """
    global hf_data_dict, video_root_dir

    while not q.empty():
        try:
            query_id = q.get_nowait()
        except Exception:
            # Queue empty / contention
            break

        if query_id not in hf_data_dict:
            # No GT for this query_id, skip
            continue

        datum = hf_data_dict[query_id]

        video_id = datum["video"]
        w, h = datum["w"], datum["h"]
        num_frames = datum["n_frames"]

        # ------------------------------------------------------------------
        # 1) Build frame-index mapping from video_dir
        # ------------------------------------------------------------------
        video_dir = Path(video_root_dir) / video_id
        if not video_dir.exists():
            print(f"[Rank {rank}] Warning: video dir not found: {video_dir}")
            continue

        frame_files = sorted(
            [f for f in os.listdir(video_dir) if f.lower().endswith((".jpg", ".png"))]
        )
        frame_names = [Path(f).stem for f in frame_files]
        frame_name_to_idx = {name: idx for idx, name in enumerate(frame_names)}

        if len(frame_names) != num_frames:
            print(
                f"[Rank {rank}] Warning: mismatch for video {video_id} "
                f"(GT n_frames={num_frames}, found {len(frame_names)} frames)"
            )

        # ------------------------------------------------------------------
        # 2) Load predicted masks for this query_id
        #     predictions_dir / <query_id> / <frame_name>.png
        # ------------------------------------------------------------------
        pred_dir = Path(predictions_dir) / query_id
        if not pred_dir.exists():
            print(f"[Rank {rank}] Warning: pred dir not found for {query_id}: {pred_dir}")
            continue

        pred_masks = {}
        mask_files = sorted([f for f in pred_dir.iterdir() if f.suffix.lower() == ".png"])

        if len(mask_files) == 0:
            print(f"[Rank {rank}] Warning: no pred masks for {query_id} in {pred_dir}")
            continue

        for mask_file in mask_files:
            stem = mask_file.stem
            if stem not in frame_name_to_idx:
                # If your preds are indexed by frame index instead of name,
                # switch this to: frame_idx = int(stem)
                print(
                    f"[Rank {rank}] Warning: frame {stem} not in video {video_id}. "
                    f"Pred file: {mask_file}"
                )
                continue

            frame_idx = frame_name_to_idx[stem]

            mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"[Rank {rank}] Warning: failed to read {mask_file}")
                continue

            # Binarize
            mask = (mask > 127).astype(np.uint8)
            pred_masks[frame_idx] = mask

        if not pred_masks:
            print(f"[Rank {rank}] Warning: after filtering, no pred masks for {query_id}")
            continue

        # ------------------------------------------------------------------
        # 3) Decode GT masks from HF RLEs
        #     datum["masks"] is expected to be: obj_id -> list[frame] of RLEs/None
        # ------------------------------------------------------------------
        # Per-object masks: dict[obj_id] -> np.ndarray [T, H, W]
        gt_masks_per_obj = {}
        for obj_id, masks in datum["masks"].items():
            if masks is None:
                continue
            # decode_masks should return [T, H, W] boolean or uint8
            obj_arr = decode_masks(masks, w=w, h=h)
            gt_masks_per_obj[obj_id] = obj_arr

        # Union over objects into single GT mask per frame
        gt_masks = np.zeros((num_frames, h, w), dtype=np.uint8)
        for obj_id, obj_masks in gt_masks_per_obj.items():
            # obj_masks: [T, H, W]
            T = min(num_frames, obj_masks.shape[0])
            gt_masks[:T] += obj_masks[:T].astype(np.uint8)

        # Clip to 0/1 in case multiple objects overlap
        gt_masks = (gt_masks > 0).astype(np.uint8)

        # ------------------------------------------------------------------
        # 4) Build predicted mask volume [T, H, W]
        # ------------------------------------------------------------------
        pred_volume = np.zeros((num_frames, h, w), dtype=np.uint8)
        for t, m in pred_masks.items():
            if t < num_frames:
                pred_volume[t] = m.astype(np.uint8)

        # ------------------------------------------------------------------
        # 5) Compute J and F, store in shared dict
        # ------------------------------------------------------------------
        try:
            j = db_eval_iou(gt_masks, pred_volume).mean()
            f = db_eval_boundary(gt_masks, pred_volume).mean()
        except Exception as e:
            print(f"[Rank {rank}] Error computing metrics for {query_id}: {e}")
            continue

        out_dict[query_id] = [float(j), float(f)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MeViS-style J/F evaluation using HF dataset + per-query predictions"
    )
    parser.add_argument(
        "--annotation",
        type=str,
        required=True,
        help="HuggingFace dataset path (load_from_disk) containing GT masks",
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        required=True,
        help="Root directory with videos as folders of frames (video_dir/<video_id>/*.jpg|*.png)",
    )
    parser.add_argument(
        "--predictions_dir",
        type=str,
        required=True,
        help="Directory with per-query prediction subdirs (predictions_dir/<query_id>/*.png)",
    )
    parser.add_argument(
        "--save_name",
        type=str,
        default="mevis_hf_result.json",
        help="Path to save per-query [J, F] results as JSON",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=NUM_WORKERS,
        help="Number of parallel workers",
    )

    args = parser.parse_args()

    # global hf_data_dict, video_root_dir
    video_root_dir = args.video_dir

    # ----------------------------------------------------------------------
    # Load HF dataset and create dict: query_id -> datum
    # Expect fields: id, video, w, h, n_frames, masks
    # ----------------------------------------------------------------------
    print(f"Loading HF dataset from: {args.annotation}")
    hf_dataset = load_from_disk(args.annotation)
    print(f"Loaded {len(hf_dataset)} GT entries")

    hf_data_dict = {}
    for datum in hf_dataset:
        qid = datum["id"]
        hf_data_dict[qid] = datum

    # ----------------------------------------------------------------------
    # Build queue of query_ids that have prediction dirs
    # ----------------------------------------------------------------------
    queue = mp.Queue()
    pred_root = Path(args.predictions_dir)

    num_enqueued = 0
    for qid in hf_data_dict.keys():
        pred_dir = pred_root / qid
        if pred_dir.exists() and any(pred_dir.iterdir()):
            queue.put(qid)
            num_enqueued += 1

    print(f"Enqueued {num_enqueued} queries that have predictions")

    manager = mp.Manager()
    output_dict = manager.dict()

    # ----------------------------------------------------------------------
    # Spawn workers
    # ----------------------------------------------------------------------
    start_time = time.time()
    processes = []
    num_workers = min(args.num_workers, num_enqueued) if num_enqueued > 0 else 0

    for rank in range(num_workers):
        p = mp.Process(
            target=eval_queue,
            args=(queue, rank, output_dict, args.predictions_dir),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # ----------------------------------------------------------------------
    # Persist results
    # ----------------------------------------------------------------------
    # Convert Manager dict to regular dict for JSON
    out_dict = dict(output_dict)

    os.makedirs(os.path.dirname(args.save_name) or ".", exist_ok=True)
    with open(args.save_name, "w") as f:
        json.dump(out_dict, f)

    # Compute global J, F, J&F
    if len(out_dict) == 0:
        print("No successful evaluations. Check paths / dataset schema.")
    else:
        j_vals = [out_dict[k][0] for k in out_dict]
        f_vals = [out_dict[k][1] for k in out_dict]

        mean_j = float(np.mean(j_vals))
        mean_f = float(np.mean(f_vals))
        mean_jf = (mean_j + mean_f) / 2.0

        print(f"J:   {mean_j:.4f}")
        print(f"F:   {mean_f:.4f}")
        print(f"J&F: {mean_jf:.4f}")

    end_time = time.time()
    print("time: %.4f s" % (end_time - start_time))