from pathlib import Path
import argparse
import json
import time

import cv2
import numpy as np
import torch
from tqdm import tqdm

from hamer.models import load_hamer
from hamer.utils import recursive_to
from hamer.datasets.vitdet_dataset import ViTDetDataset
from hamer.utils.renderer import Renderer, cam_crop_to_full
from hamer.datasets.utils import expand_to_aspect_ratio, gen_trans_from_patch_cv, trans_point2d

import wildhands.common.data_utils as data_utils

from hand_crop import (
    DEFAULT_MEDIAPIPE_MODEL,
    bbox_iou,
    create_mediapipe_hand_detector,
    detect_mediapipe_hands_with_fallback,
    draw_hand_bbox_debug,
)

LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)


def alpha_mask_bbox(alpha, thr=1e-4):
    ys, xs = np.where(alpha > thr)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def parse_focal_candidates(spec, default_focal):
    candidates = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        candidates.append(float(part))
    candidates.append(float(default_focal))
    return sorted(set(candidates))


def bbox_center(box):
    x1, y1, x2, y2 = [float(v) for v in box]
    return np.asarray([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32)


def bbox_span(box):
    x1, y1, x2, y2 = [float(v) for v in box]
    return max(x2 - x1, y2 - y1, 1.0)


def candidate_quality(item):
    orig = item["orig"]
    quality = 0.0
    if orig.get("source") == "full":
        quality += 1.0
    quality += 0.25 * float(orig.get("mediapipe_handedness_score") or 0.0)
    quality += 1e-5 * float((orig["bbox"][2] - orig["bbox"][0]) * (orig["bbox"][3] - orig["bbox"][1]))
    return quality


def smooth_track_handedness(track, window_size=5, min_margin=0.12, stability_bias=0.25, flip_streak=3):
    history_is_right = list(track.get("history_is_right", []))
    history_scores = list(track.get("history_handedness_scores", []))
    if len(history_is_right) == 0:
        current = int(track.get("is_right", 0))
        return current, {
            "weighted_left": None,
            "weighted_right": None,
            "margin": None,
            "window_size": int(window_size),
            "used_history": 0,
            "changed": False,
            "flip_streak_required": int(flip_streak),
            "recent_raw_streak": 0,
            "recent_raw_flag": None,
        }

    history_is_right = history_is_right[-window_size:]
    history_scores = history_scores[-window_size:]
    weighted_left = 0.0
    weighted_right = 0.0
    for flag, score in zip(history_is_right, history_scores):
        weight = float(score) if score is not None else 0.5
        weight = max(weight, 0.05)
        if int(flag):
            weighted_right += weight
        else:
            weighted_left += weight

    prev_smoothed = int(track.get("smoothed_is_right", track.get("is_right", 0)))
    recent_raw_flag = int(history_is_right[-1])
    recent_raw_streak = 0
    for flag in reversed(history_is_right):
        if int(flag) != recent_raw_flag:
            break
        recent_raw_streak += 1

    if prev_smoothed:
        weighted_right += float(stability_bias)
    else:
        weighted_left += float(stability_bias)

    total = max(weighted_left + weighted_right, 1e-6)
    margin = abs(weighted_right - weighted_left) / total
    proposed = 1 if weighted_right >= weighted_left else 0
    allow_flip = True
    if proposed != prev_smoothed:
        allow_flip = proposed == recent_raw_flag and recent_raw_streak >= int(flip_streak)

    if margin < float(min_margin) or not allow_flip:
        smoothed = prev_smoothed
    else:
        smoothed = proposed

    return smoothed, {
        "weighted_left": float(weighted_left),
        "weighted_right": float(weighted_right),
        "margin": float(margin),
        "window_size": int(window_size),
        "used_history": int(len(history_is_right)),
        "changed": bool(smoothed != int(track.get("raw_is_right", track.get("is_right", 0)))),
        "flip_streak_required": int(flip_streak),
        "recent_raw_streak": int(recent_raw_streak),
        "recent_raw_flag": int(recent_raw_flag),
        "flip_blocked_by_streak": bool(proposed != prev_smoothed and not allow_flip),
    }


def temporal_match_score(item, track):
    item_box = item["orig"]["bbox"]
    track_box = track["bbox_orig"]
    iou = bbox_iou(item_box, track_box)
    dist = float(np.linalg.norm(bbox_center(item_box) - bbox_center(track_box)))
    norm_dist = dist / max(bbox_span(item_box), bbox_span(track_box), 1.0)
    handedness_bonus = 0.15 if int(item["orig"]["is_right"]) == int(track["is_right"]) else -0.05
    return float(iou - 0.35 * norm_dist + handedness_bonus), float(iou), float(norm_dist)


def same_candidate_like(item_a, item_b, iou_thr=0.2, center_ratio_thr=0.45):
    box_a = item_a["orig"]["bbox"]
    box_b = item_b["orig"]["bbox"]
    iou = bbox_iou(box_a, box_b)
    if iou >= iou_thr:
        return True
    dist = float(np.linalg.norm(bbox_center(box_a) - bbox_center(box_b)))
    norm_dist = dist / max(bbox_span(box_a), bbox_span(box_b), 1.0)
    if norm_dist <= center_ratio_thr and int(item_a["orig"]["is_right"]) == int(item_b["orig"]["is_right"]):
        return True
    return False


def make_track_from_item(item, track_id, frame_idx, birth_frames):
    return {
        "track_id": int(track_id),
        "bbox_orig": [float(v) for v in item["orig"]["bbox"]],
        "bbox_square": [float(v) for v in item["square"]["bbox"]],
        "raw_is_right": int(item["orig"]["is_right"]),
        "is_right": int(item["orig"]["is_right"]),
        "smoothed_is_right": int(item["orig"]["is_right"]),
        "hits": 1,
        "misses": 0,
        "confirmed": bool(birth_frames <= 1),
        "last_seen_frame": int(frame_idx),
        "source": item["orig"].get("source"),
        "mediapipe_handedness_score": item["orig"].get("mediapipe_handedness_score"),
        "candidate_orig": dict(item["orig"]),
        "candidate_square": dict(item["square"]),
        "reused_previous": False,
        "history_centers": [bbox_center(item["orig"]["bbox"]).astype(float).tolist()],
        "history_spans": [float(bbox_span(item["orig"]["bbox"]))],
        "history_is_right": [int(item["orig"]["is_right"])],
        "history_handedness_scores": [item["orig"].get("mediapipe_handedness_score")],
        "recent_motion_norm": None,
        "recent_scale_change": None,
        "handedness_smoothing": None,
    }


def update_track_from_item(track, item, frame_idx, birth_frames):
    prev_center = bbox_center(track["bbox_orig"])
    prev_span = float(bbox_span(track["bbox_orig"]))
    new_center = bbox_center(item["orig"]["bbox"])
    new_span = float(bbox_span(item["orig"]["bbox"]))
    motion_norm = float(np.linalg.norm(new_center - prev_center) / max(prev_span, new_span, 1.0))
    scale_change = float(max(new_span / max(prev_span, 1.0), prev_span / max(new_span, 1.0)))

    track["bbox_orig"] = [float(v) for v in item["orig"]["bbox"]]
    track["bbox_square"] = [float(v) for v in item["square"]["bbox"]]
    track["raw_is_right"] = int(item["orig"]["is_right"])
    track["is_right"] = int(item["orig"]["is_right"])
    track["hits"] = int(track.get("hits", 0)) + 1
    track["misses"] = 0
    track["confirmed"] = bool(track["hits"] >= birth_frames)
    track["last_seen_frame"] = int(frame_idx)
    track["source"] = item["orig"].get("source")
    track["mediapipe_handedness_score"] = item["orig"].get("mediapipe_handedness_score")
    track["candidate_orig"] = dict(item["orig"])
    track["candidate_square"] = dict(item["square"])
    track["reused_previous"] = False
    centers = list(track.get("history_centers", []))
    spans = list(track.get("history_spans", []))
    handedness_hist = list(track.get("history_is_right", []))
    handedness_score_hist = list(track.get("history_handedness_scores", []))
    centers.append(new_center.astype(float).tolist())
    spans.append(new_span)
    handedness_hist.append(int(item["orig"]["is_right"]))
    handedness_score_hist.append(item["orig"].get("mediapipe_handedness_score"))
    track["history_centers"] = centers[-8:]
    track["history_spans"] = spans[-8:]
    track["history_is_right"] = handedness_hist[-8:]
    track["history_handedness_scores"] = handedness_score_hist[-8:]
    track["recent_motion_norm"] = motion_norm
    track["recent_scale_change"] = scale_change
    return track


def filter_duplicate_new_items(items, iou_thr=0.2, center_ratio_thr=0.45):
    kept = []
    suppressed = []
    for item in sorted(items, key=candidate_quality, reverse=True):
        duplicate_with = None
        for kept_item in kept:
            if same_candidate_like(item, kept_item, iou_thr=iou_thr, center_ratio_thr=center_ratio_thr):
                duplicate_with = int(kept_item["candidate_idx"])
                break
        if duplicate_with is None:
            kept.append(item)
        else:
            suppressed.append(
                {
                    "candidate_idx": int(item["candidate_idx"]),
                    "reason": "same_frame_duplicate",
                    "kept_candidate_idx": duplicate_with,
                }
            )
    kept.sort(key=lambda item: item["candidate_idx"])
    return kept, suppressed


def apply_temporal_hand_filtering(
    candidates_orig,
    candidates_square,
    temporal_state,
    frame_idx,
    birth_frames=2,
    max_missed=2,
    duplicate_iou=0.2,
    duplicate_center_ratio=0.45,
    second_hand_birth_frames=5,
    second_hand_max_motion_norm=0.6,
    second_hand_max_scale_change=1.8,
    handedness_window=5,
    handedness_min_margin=0.12,
    handedness_stability_bias=0.25,
    handedness_flip_streak=3,
):
    temporal_state = temporal_state or {"tracks": [], "next_track_id": 0}
    tracks = [dict(track) for track in temporal_state.get("tracks", [])]

    current_items = []
    for idx, (candidate_orig, candidate_square) in enumerate(zip(candidates_orig, candidates_square)):
        current_items.append(
            {
                "candidate_idx": int(idx),
                "orig": dict(candidate_orig),
                "square": dict(candidate_square),
            }
        )

    pairings = []
    for item in current_items:
        for track in tracks:
            score, iou, norm_dist = temporal_match_score(item, track)
            if iou > 0.01 or norm_dist < 0.8:
                pairings.append(
                    {
                        "candidate_idx": int(item["candidate_idx"]),
                        "track_id": int(track["track_id"]),
                        "score": float(score),
                        "iou": float(iou),
                        "norm_dist": float(norm_dist),
                    }
                )

    pairings.sort(key=lambda item: item["score"], reverse=True)
    matched_candidate_indices = set()
    matched_track_ids = set()
    matches = []
    for pairing in pairings:
        candidate_idx = pairing["candidate_idx"]
        track_id = pairing["track_id"]
        if candidate_idx in matched_candidate_indices or track_id in matched_track_ids:
            continue
        matches.append(pairing)
        matched_candidate_indices.add(candidate_idx)
        matched_track_ids.add(track_id)

    items_by_idx = {item["candidate_idx"]: item for item in current_items}
    track_by_id = {int(track["track_id"]): track for track in tracks}
    suppressed = []

    for match in matches:
        track = track_by_id[match["track_id"]]
        item = items_by_idx[match["candidate_idx"]]
        update_track_from_item(track, item, frame_idx, birth_frames)

    unmatched_items = [item for item in current_items if item["candidate_idx"] not in matched_candidate_indices]
    filtered_unmatched_items, duplicate_suppressed = filter_duplicate_new_items(
        unmatched_items,
        iou_thr=duplicate_iou,
        center_ratio_thr=duplicate_center_ratio,
    )
    suppressed.extend(duplicate_suppressed)

    next_track_id = int(temporal_state.get("next_track_id", 0))
    for item in filtered_unmatched_items:
        new_track = make_track_from_item(item, next_track_id, frame_idx, birth_frames)
        tracks.append(new_track)
        track_by_id[new_track["track_id"]] = new_track
        next_track_id += 1

    updated_tracks = []
    active_tracks = []
    for track in tracks:
        if track["track_id"] not in matched_track_ids and track["last_seen_frame"] != int(frame_idx):
            track["misses"] = int(track.get("misses", 0)) + 1
            track["reused_previous"] = True
        if int(track.get("misses", 0)) > int(max_missed):
            continue
        smoothed_is_right, smoothing_debug = smooth_track_handedness(
            track,
            window_size=handedness_window,
            min_margin=handedness_min_margin,
            stability_bias=handedness_stability_bias,
            flip_streak=handedness_flip_streak,
        )
        track["smoothed_is_right"] = int(smoothed_is_right)
        track["is_right"] = int(smoothed_is_right)
        track["handedness_smoothing"] = smoothing_debug
        updated_tracks.append(track)

    confirmed_tracks = [track for track in updated_tracks if bool(track.get("confirmed", False))]
    confirmed_tracks.sort(key=lambda track: int(track["track_id"]))

    primary_tracks = confirmed_tracks[:1]
    secondary_tracks = []
    suppressed = list(suppressed)
    for track in confirmed_tracks[1:]:
        enough_hits = int(track.get("hits", 0)) >= int(second_hand_birth_frames)
        motion_ok = (
            track.get("recent_motion_norm") is None
            or float(track.get("recent_motion_norm")) <= float(second_hand_max_motion_norm)
        )
        scale_ok = (
            track.get("recent_scale_change") is None
            or float(track.get("recent_scale_change")) <= float(second_hand_max_scale_change)
        )
        if enough_hits and motion_ok and scale_ok:
            secondary_tracks.append(track)
        else:
            suppressed.append(
                {
                    "track_id": int(track["track_id"]),
                    "reason": "second_hand_gate",
                    "hits": int(track.get("hits", 0)),
                    "recent_motion_norm": track.get("recent_motion_norm"),
                    "recent_scale_change": track.get("recent_scale_change"),
                }
            )

    active_tracks = primary_tracks + secondary_tracks

    filtered_candidates_orig = []
    filtered_candidates_square = []
    filtered_is_right = []
    for track in active_tracks:
        candidate_orig = dict(track["candidate_orig"])
        candidate_square = dict(track["candidate_square"])
        candidate_orig["track_id"] = int(track["track_id"])
        candidate_square["track_id"] = int(track["track_id"])
        candidate_orig["raw_is_right"] = int(track.get("raw_is_right", candidate_orig.get("is_right", 0)))
        candidate_orig["smoothed_is_right"] = int(track["is_right"])
        candidate_orig["is_right"] = int(track["is_right"])
        candidate_square["raw_is_right"] = int(track.get("raw_is_right", candidate_square.get("is_right", 0)))
        candidate_square["smoothed_is_right"] = int(track["is_right"])
        candidate_square["is_right"] = int(track["is_right"])
        filtered_candidates_orig.append(candidate_orig)
        filtered_candidates_square.append(candidate_square)
        filtered_is_right.append(int(track["is_right"]))

    temporal_debug = {
        "raw_candidate_count": int(len(current_items)),
        "output_candidate_count": int(len(filtered_candidates_orig)),
        "matches": matches,
        "suppressed_candidates": suppressed,
        "active_track_ids": [int(track["track_id"]) for track in active_tracks],
        "tentative_track_ids": [int(track["track_id"]) for track in updated_tracks if not track.get("confirmed", False)],
        "reused_track_ids": [int(track["track_id"]) for track in active_tracks if track.get("reused_previous", False)],
        "tracks": [
            {
                "track_id": int(track["track_id"]),
                "raw_is_right": int(track.get("raw_is_right", track["is_right"])),
                "is_right": int(track["is_right"]),
                "smoothed_is_right": int(track.get("smoothed_is_right", track["is_right"])),
                "hits": int(track["hits"]),
                "misses": int(track["misses"]),
                "confirmed": bool(track["confirmed"]),
                "reused_previous": bool(track.get("reused_previous", False)),
                "recent_motion_norm": track.get("recent_motion_norm"),
                "recent_scale_change": track.get("recent_scale_change"),
                "handedness_smoothing": track.get("handedness_smoothing"),
                "bbox": [float(v) for v in track["bbox_orig"]],
            }
            for track in updated_tracks
        ],
    }
    new_state = {
        "tracks": updated_tracks,
        "next_track_id": int(next_track_id),
    }
    return filtered_candidates_orig, filtered_candidates_square, filtered_is_right, new_state, temporal_debug


def get_hamer_bbox_center_and_size(box, cfg, rescale_factor=2.0):
    box = np.asarray(box, dtype=np.float32)
    center = (box[2:4] + box[0:2]) / 2.0
    scale = rescale_factor * (box[2:4] - box[0:2]) / 200.0
    bbox_shape = cfg.MODEL.get("BBOX_SHAPE", None)
    bbox_size = expand_to_aspect_ratio(scale * 200.0, target_aspect_ratio=bbox_shape).max()
    return center, float(bbox_size)


def mediapipe_landmarks_to_hamer_crop_coords(landmarks_xy, image_shape, box, cfg, is_right, rescale_factor=2.0):
    crop_size = float(cfg.MODEL.IMAGE_SIZE)
    center, bbox_size = get_hamer_bbox_center_and_size(box, cfg, rescale_factor=rescale_factor)
    trans = gen_trans_from_patch_cv(
        float(center[0]),
        float(center[1]),
        float(bbox_size),
        float(bbox_size),
        crop_size,
        crop_size,
        1.0,
        0.0,
    )

    points = np.asarray(landmarks_xy, dtype=np.float32).copy()
    if not is_right:
        points[:, 0] = image_shape[1] - points[:, 0] - 1

    crop_points = np.stack([trans_point2d(pt, trans) for pt in points], axis=0)
    crop_points = crop_points / crop_size - 0.5
    return crop_points.astype(np.float32)


def resolve_hamer_handedness_from_mediapipe(
    model,
    cfg,
    device,
    image_bgr,
    candidates,
    hands,
    batch_size=1,
    rescale_factor=2.0,
    trust_mediapipe_score_threshold=0.75,
    min_margin_to_override=0.05,
):
    if len(candidates) == 0:
        return candidates, []

    hand_lookup = {
        (hand.get("hand_index", 0), hand.get("source", "full")): hand
        for hand in hands
    }

    auto_boxes = []
    auto_right = []
    auto_targets = []
    auto_meta = []

    for candidate_idx, candidate in enumerate(candidates):
        hand_key = (candidate.get("hand_index", 0), candidate.get("source", "full"))
        hand = hand_lookup.get(hand_key)
        if hand is None or len(hand.get("landmarks", [])) == 0:
            continue

        landmarks_xy = np.asarray(
            [[float(pt["x"]), float(pt["y"])] for pt in hand["landmarks"]],
            dtype=np.float32,
        )
        if landmarks_xy.shape[0] == 0:
            continue

        for right_flag in (0, 1):
            auto_boxes.append(np.asarray(candidate["bbox"], dtype=np.float32))
            auto_right.append(float(right_flag))
            auto_targets.append(
                mediapipe_landmarks_to_hamer_crop_coords(
                    landmarks_xy,
                    image_bgr.shape,
                    candidate["bbox"],
                    cfg,
                    is_right=right_flag,
                    rescale_factor=rescale_factor,
                )
            )
            auto_meta.append(
                {
                    "candidate_idx": candidate_idx,
                    "is_right": int(right_flag),
                }
            )

    if len(auto_boxes) == 0:
        return candidates, []

    dataset = ViTDetDataset(
        cfg,
        image_bgr,
        np.stack(auto_boxes),
        np.asarray(auto_right, dtype=np.float32),
        rescale_factor=rescale_factor,
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    resolved_debug = []
    candidate_errors = {}
    sample_offset = 0
    for batch in dataloader:
        batch = recursive_to(batch, device)
        with torch.no_grad():
            out = model(batch)
        pred_keypoints_2d = out["pred_keypoints_2d"].detach().cpu().numpy()

        for local_idx in range(pred_keypoints_2d.shape[0]):
            sample_idx = sample_offset + local_idx
            meta = auto_meta[sample_idx]
            target = auto_targets[sample_idx]
            pred = pred_keypoints_2d[local_idx]
            error = float(np.mean(np.linalg.norm(pred - target, axis=1)))
            candidate_errors.setdefault(meta["candidate_idx"], {})[meta["is_right"]] = error
        sample_offset += pred_keypoints_2d.shape[0]

    updated_candidates = []
    for candidate_idx, candidate in enumerate(candidates):
        updated_candidate = dict(candidate)
        errors = candidate_errors.get(candidate_idx, {})
        raw_is_right = int(candidate["is_right"])
        raw_score = candidate.get("mediapipe_handedness_score")
        updated_candidate["mediapipe_raw_is_right"] = raw_is_right
        updated_candidate["mediapipe_raw_handedness_score"] = (
            float(raw_score) if raw_score is not None else None
        )
        if 0 in errors and 1 in errors:
            best_fit_is_right = 1 if errors[1] <= errors[0] else 0
            margin = float(abs(errors[1] - errors[0]))
            allow_override = True
            if raw_score is not None and float(raw_score) >= float(trust_mediapipe_score_threshold):
                allow_override = False
            if margin < float(min_margin_to_override):
                allow_override = False

            if best_fit_is_right != raw_is_right and allow_override:
                best_is_right = best_fit_is_right
            else:
                best_is_right = raw_is_right

            updated_candidate["auto_handedness_enabled"] = True
            updated_candidate["auto_handedness_error_left"] = float(errors[0])
            updated_candidate["auto_handedness_error_right"] = float(errors[1])
            updated_candidate["auto_handedness_margin"] = margin
            updated_candidate["auto_handedness_best_fit_is_right"] = int(best_fit_is_right)
            updated_candidate["auto_handedness_trust_mediapipe_score_threshold"] = float(trust_mediapipe_score_threshold)
            updated_candidate["auto_handedness_min_margin_to_override"] = float(min_margin_to_override)
            updated_candidate["auto_handedness_override_applied"] = bool(best_is_right != raw_is_right)
            updated_candidate["auto_handedness_override_blocked"] = bool(best_fit_is_right != raw_is_right and not allow_override)
            updated_candidate["auto_handedness_changed"] = bool(best_is_right != raw_is_right)
        else:
            best_is_right = raw_is_right
            updated_candidate["auto_handedness_enabled"] = False
        updated_candidates.append(updated_candidate)

        hand_key = (candidate.get("hand_index", 0), candidate.get("source", "full"))
        hand = hand_lookup.get(hand_key)
        if hand is not None:
            hand["resolved_is_right"] = int(best_is_right)
            hand["resolved_handedness"] = "Right" if best_is_right else "Left"

        resolved_debug.append(
            {
                "candidate_idx": int(candidate_idx),
                "bbox": [float(v) for v in candidate["bbox"]],
                "raw_is_right": int(raw_is_right),
                "raw_handedness_score": float(raw_score) if raw_score is not None else None,
                "best_fit_is_right": int(best_fit_is_right) if 0 in errors and 1 in errors else int(raw_is_right),
                "selected_is_right": int(best_is_right),
                "error_left": float(errors[0]) if 0 in errors else None,
                "error_right": float(errors[1]) if 1 in errors else None,
                "margin": float(abs(errors[1] - errors[0])) if 0 in errors and 1 in errors else None,
            }
        )

    return updated_candidates, resolved_debug


def get_square_render_mapping(orig_shape, render_res):
    orig_h, orig_w = orig_shape[:2]
    input_res = float(max(orig_h, orig_w))
    scale = float(render_res) / input_res
    pad_x = (input_res - float(orig_w)) / 2.0
    pad_y = (input_res - float(orig_h)) / 2.0
    return pad_x, pad_y, scale


def map_xy_to_square_render(x, y, orig_shape, render_res):
    pad_x, pad_y, scale = get_square_render_mapping(orig_shape, render_res)
    return (float(x) + pad_x) * scale, (float(y) + pad_y) * scale


def map_bbox_to_square_render(bbox, orig_shape, render_res):
    x1, y1, x2, y2 = bbox
    mx1, my1 = map_xy_to_square_render(x1, y1, orig_shape, render_res)
    mx2, my2 = map_xy_to_square_render(x2, y2, orig_shape, render_res)
    return [mx1, my1, mx2, my2]


def map_hands_to_square_render(hands, orig_shape, render_res):
    mapped_hands = []
    for hand in hands:
        mapped_hand = dict(hand)
        mapped_landmarks = []
        for landmark in hand.get("landmarks", []):
            mx, my = map_xy_to_square_render(landmark["x"], landmark["y"], orig_shape, render_res)
            mapped_landmark = dict(landmark)
            mapped_landmark["x"] = float(mx)
            mapped_landmark["y"] = float(my)
            mapped_landmarks.append(mapped_landmark)
        mapped_hand["landmarks"] = mapped_landmarks
        if "roi_xyxy" in hand and hand["roi_xyxy"] is not None:
            mapped_hand["roi_xyxy"] = [
                int(round(v))
                for v in map_bbox_to_square_render(hand["roi_xyxy"], orig_shape, render_res)
            ]
        mapped_hands.append(mapped_hand)
    return mapped_hands


def map_candidates_to_square_render(candidates, orig_shape, render_res):
    mapped_candidates = []
    for candidate in candidates:
        mapped_candidate = dict(candidate)
        mapped_candidate["bbox"] = map_bbox_to_square_render(candidate["bbox"], orig_shape, render_res)
        if "roi_xyxy" in candidate and candidate["roi_xyxy"] is not None:
            mapped_candidate["roi_xyxy"] = [
                int(round(v))
                for v in map_bbox_to_square_render(candidate["roi_xyxy"], orig_shape, render_res)
            ]
        mapped_candidates.append(mapped_candidate)
    return mapped_candidates


def flip_bbox_xyxy(bbox, image_width):
    x1, y1, x2, y2 = [float(v) for v in bbox]
    max_x = float(image_width - 1)
    return [
        max(0.0, min(max_x, max_x - x2)),
        y1,
        max(0.0, min(max_x, max_x - x1)),
        y2,
    ]


def flip_candidate_list_back(candidates, image_width):
    flipped = []
    for candidate in candidates:
        updated = dict(candidate)
        updated["bbox"] = flip_bbox_xyxy(candidate["bbox"], image_width)
        if "roi_xyxy" in candidate and candidate["roi_xyxy"] is not None:
            updated["roi_xyxy"] = [
                int(round(v))
                for v in flip_bbox_xyxy(candidate["roi_xyxy"], image_width)
            ]
        flipped.append(updated)
    return flipped


def project_points_to_image(points_3d, cam_t, focal_length, image_size):
    image_h, image_w = image_size[:2]
    points = np.asarray(points_3d, dtype=np.float32)
    cam_t = np.asarray(cam_t, dtype=np.float32).reshape(1, 3)
    cam_points = points + cam_t
    z = np.clip(cam_points[:, 2:3], 1e-6, None)
    projected = cam_points[:, :2] / z
    projected *= float(focal_length)
    projected[:, 0] += float(image_w) / 2.0
    projected[:, 1] += float(image_h) / 2.0
    return projected.astype(np.float32)


def preprocess_frame_to_square(frame_bgr, render_res):
    cv_img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    input_res = max(cv_img.shape[:2])
    image = data_utils.generate_patch_image_clean(
        cv_img,
        [cv_img.shape[1] / 2, cv_img.shape[0] / 2, input_res, input_res],
        1.0,
        0.0,
        [render_res, render_res],
        cv2.INTER_CUBIC,
    )[0]
    img = image.clip(0, 255)
    return img.astype(np.uint8)[..., ::-1], input_res


def prepare_frame_candidates(
    frame_bgr,
    mp_module,
    mp_hand_detector,
    model,
    cfg,
    device,
    args,
    timestamp_ms=None,
):
    timing = {}
    original_frame_bgr = frame_bgr
    if args.mirror_input:
        frame_bgr = cv2.flip(frame_bgr, 1)
    orig_shape = frame_bgr.shape

    stage_start = time.perf_counter()
    square_bgr, input_res = preprocess_frame_to_square(frame_bgr, args.render_res)
    timing["preprocess_s"] = float(time.perf_counter() - stage_start)

    stage_start = time.perf_counter()
    detection_info = detect_mediapipe_hands_with_fallback(
        mp_module,
        mp_hand_detector,
        frame_bgr,
        timestamp_ms=timestamp_ms,
        swap_hands=args.mediapipe_swap_hands,
    )
    timing["mediapipe_s"] = float(time.perf_counter() - stage_start)

    hands_square = map_hands_to_square_render(detection_info["hands"], orig_shape, args.render_res)
    candidates_square = map_candidates_to_square_render(detection_info["candidates"], orig_shape, args.render_res)
    candidates_orig = [dict(candidate) for candidate in detection_info["candidates"]]
    auto_handedness_debug = []

    if args.mediapipe_auto_handedness and len(candidates_square) > 0:
        stage_start = time.perf_counter()
        candidates_square, auto_handedness_debug = resolve_hamer_handedness_from_mediapipe(
            model,
            cfg,
            device,
            square_bgr,
            candidates_square,
            hands_square,
            batch_size=args.batch_size,
            rescale_factor=2.0,
            trust_mediapipe_score_threshold=args.mediapipe_handedness_score_threshold,
            min_margin_to_override=args.mediapipe_auto_handedness_margin_threshold,
        )
        timing["mediapipe_auto_handedness_s"] = float(time.perf_counter() - stage_start)
    else:
        timing["mediapipe_auto_handedness_s"] = 0.0

    final_is_right = [int(candidate["is_right"]) for candidate in candidates_square]
    for idx in range(min(len(candidates_orig), len(final_is_right))):
        candidates_orig[idx]["is_right"] = final_is_right[idx]

    if args.mirror_input:
        candidates_orig = flip_candidate_list_back(candidates_orig, original_frame_bgr.shape[1])

    return {
        "original_frame_bgr": original_frame_bgr,
        "processed_frame_bgr": frame_bgr,
        "orig_shape": orig_shape,
        "square_bgr": square_bgr,
        "input_res": input_res,
        "timing": timing,
        "detection_info": detection_info,
        "hands_square": hands_square,
        "candidates_square": candidates_square,
        "candidates_orig": candidates_orig,
        "auto_handedness_debug": auto_handedness_debug,
        "final_is_right": final_is_right,
    }


def estimate_focal_candidate_errors(
    model,
    cfg,
    device,
    square_bgr,
    input_res,
    candidates_square,
    hands_square,
    focal_candidates,
    batch_size=1,
    rescale_factor=2.0,
):
    if len(candidates_square) == 0:
        return {}, 0

    hand_lookup = {
        (hand.get("hand_index", 0), hand.get("source", "full")): hand
        for hand in hands_square
    }

    boxes = []
    is_right = []
    targets = []
    metas = []
    for candidate_idx, candidate in enumerate(candidates_square):
        hand_key = (candidate.get("hand_index", 0), candidate.get("source", "full"))
        hand = hand_lookup.get(hand_key)
        if hand is None or len(hand.get("landmarks", [])) == 0:
            continue
        target_xy = np.asarray(
            [[float(pt["x"]), float(pt["y"])] for pt in hand["landmarks"]],
            dtype=np.float32,
        )
        if target_xy.shape[0] == 0:
            continue
        boxes.append(np.asarray(candidate["bbox"], dtype=np.float32))
        is_right.append(float(candidate["is_right"]))
        targets.append(target_xy)
        metas.append({"candidate_idx": int(candidate_idx)})

    if len(boxes) == 0:
        return {}, 0

    dataset = ViTDetDataset(
        cfg,
        square_bgr,
        np.stack(boxes),
        np.asarray(is_right, dtype=np.float32),
        rescale_factor=rescale_factor,
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    scaled_focal_candidates = {
        float(focal): float(focal) * square_bgr.shape[1] / float(input_res)
        for focal in focal_candidates
    }
    error_sums = {float(focal): 0.0 for focal in focal_candidates}
    error_counts = {float(focal): 0 for focal in focal_candidates}

    sample_offset = 0
    for batch in dataloader:
        batch = recursive_to(batch, device)
        with torch.no_grad():
            out = model(batch)

        batch_right = batch["right"]
        multiplier = (2 * batch_right - 1)
        pred_cam = out["pred_cam"].clone()
        pred_cam[:, 1] = multiplier * pred_cam[:, 1]
        box_center = batch["box_center"].float()
        box_size = batch["box_size"].float()
        img_size = batch["img_size"].float()

        pred_keypoints_3d = out["pred_keypoints_3d"].detach().cpu().numpy()
        batch_right_np = batch_right.detach().cpu().numpy().astype(int)

        cam_t_per_focal = {}
        for focal, scaled_focal in scaled_focal_candidates.items():
            cam_t_per_focal[focal] = cam_crop_to_full(
                pred_cam,
                box_center,
                box_size,
                img_size,
                scaled_focal,
            ).detach().cpu().numpy()

        for local_idx in range(pred_keypoints_3d.shape[0]):
            sample_idx = sample_offset + local_idx
            joints = pred_keypoints_3d[local_idx].copy()
            right_flag = int(batch_right_np[local_idx])
            joints[:, 0] = (2 * right_flag - 1) * joints[:, 0]
            target_xy = targets[sample_idx]

            for focal, scaled_focal in scaled_focal_candidates.items():
                proj_xy = project_points_to_image(
                    joints,
                    cam_t_per_focal[focal][local_idx],
                    scaled_focal,
                    square_bgr.shape,
                )
                error = float(np.mean(np.linalg.norm(proj_xy - target_xy, axis=1)))
                error_sums[focal] += error
                error_counts[focal] += 1

        sample_offset += pred_keypoints_3d.shape[0]

    mean_errors = {}
    for focal in focal_candidates:
        focal = float(focal)
        if error_counts[focal] > 0:
            mean_errors[focal] = error_sums[focal] / error_counts[focal]
        else:
            mean_errors[focal] = None
    return mean_errors, int(sum(error_counts.values()))


def auto_select_focal_for_video(
    video_path,
    mp_module,
    mp_hand_detector,
    model,
    cfg,
    device,
    args,
):
    focal_candidates = parse_focal_candidates(args.auto_focal_candidates, args.focal_length)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video for focal search: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    aggregated_sums = {float(focal): 0.0 for focal in focal_candidates}
    aggregated_counts = {float(focal): 0 for focal in focal_candidates}
    searched_frame_indices = []
    used_detected_frames = 0
    sampled_frames = 0

    try:
        with tqdm(total=args.auto_focal_frames, desc=f"{video_path.stem}_focal", unit="frame") as pbar:
            frame_idx = 0
            while used_detected_frames < args.auto_focal_frames:
                ok, frame_bgr = cap.read()
                if not ok:
                    break

                if args.auto_focal_stride > 1 and (frame_idx % args.auto_focal_stride) != 0:
                    frame_idx += 1
                    continue

                sampled_frames += 1
                prepared = prepare_frame_candidates(
                    frame_bgr,
                    mp_module,
                    mp_hand_detector,
                    model,
                    cfg,
                    device,
                    args,
                    timestamp_ms=int(round(frame_idx * 1000.0 / fps)),
                )
                if len(prepared["candidates_square"]) == 0:
                    frame_idx += 1
                    continue

                frame_errors, frame_error_count = estimate_focal_candidate_errors(
                    model,
                    cfg,
                    device,
                    prepared["square_bgr"],
                    prepared["input_res"],
                    prepared["candidates_square"],
                    prepared["hands_square"],
                    focal_candidates,
                    batch_size=args.batch_size,
                    rescale_factor=2.0,
                )
                if frame_error_count <= 0:
                    frame_idx += 1
                    continue

                used_detected_frames += 1
                searched_frame_indices.append(int(frame_idx))
                for focal, error in frame_errors.items():
                    if error is None:
                        continue
                    aggregated_sums[float(focal)] += float(error)
                    aggregated_counts[float(focal)] += 1
                pbar.update(1)
                frame_idx += 1
    finally:
        cap.release()

    mean_errors = {}
    valid_candidates = []
    for focal in focal_candidates:
        focal = float(focal)
        if aggregated_counts[focal] > 0:
            mean_errors[focal] = aggregated_sums[focal] / aggregated_counts[focal]
            valid_candidates.append((focal, mean_errors[focal]))
        else:
            mean_errors[focal] = None

    if len(valid_candidates) == 0:
        selected_focal = float(args.focal_length)
    else:
        selected_focal = min(valid_candidates, key=lambda item: item[1])[0]

    debug = {
        "enabled": True,
        "selected_focal_length": float(selected_focal),
        "default_focal_length": float(args.focal_length),
        "candidate_focal_lengths": [float(x) for x in focal_candidates],
        "candidate_mean_errors": {
            str(int(focal) if float(focal).is_integer() else focal): (
                float(error) if error is not None else None
            )
            for focal, error in mean_errors.items()
        },
        "searched_frame_indices": searched_frame_indices,
        "sampled_frames": int(sampled_frames),
        "detected_frames_used": int(used_detected_frames),
        "frame_stride": int(args.auto_focal_stride),
        "max_detected_frames": int(args.auto_focal_frames),
    }
    return float(selected_focal), debug


def crop_square_overlay_back_to_original(square_rgb, orig_shape, render_res):
    orig_h, orig_w = orig_shape[:2]
    pad_x, pad_y, scale = get_square_render_mapping(orig_shape, render_res)

    x1 = max(0, int(round(pad_x * scale)))
    y1 = max(0, int(round(pad_y * scale)))
    x2 = min(render_res, int(round((pad_x + orig_w) * scale)))
    y2 = min(render_res, int(round((pad_y + orig_h) * scale)))

    if x2 <= x1 or y2 <= y1:
        return cv2.resize(square_rgb, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)

    crop_rgb = square_rgb[y1:y2, x1:x2]
    return cv2.resize(crop_rgb, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)


def render_hamer_on_square(
    model,
    cfg,
    renderer,
    device,
    square_bgr,
    boxes,
    is_right,
    scaled_focal_length,
    candidate_metas=None,
    batch_size=1,
):
    dataset = ViTDetDataset(cfg, square_bgr, boxes, is_right, rescale_factor=2.0)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_verts = []
    all_cam_t = []
    all_right = []
    all_hand_outputs = []
    total_model_s = 0.0
    total_render_single_s = 0.0
    total_render_multi_s = 0.0
    total_render_fallback_s = 0.0

    for batch in dataloader:
        batch = recursive_to(batch, device)
        stage_start = time.perf_counter()
        with torch.no_grad():
            out = model(batch)
        total_model_s += float(time.perf_counter() - stage_start)

        batch_right = batch["right"]
        multiplier = (2 * batch_right - 1)
        pred_cam_raw = out["pred_cam"].detach().cpu().numpy()
        pred_cam = out["pred_cam"].clone()
        pred_cam[:, 1] = multiplier * pred_cam[:, 1]
        pred_cam_np = pred_cam.detach().cpu().numpy()
        box_center = batch["box_center"].float()
        box_size = batch["box_size"].float()
        img_size = batch["img_size"].float()
        pred_cam_t_full = cam_crop_to_full(
            pred_cam,
            box_center,
            box_size,
            img_size,
            scaled_focal_length,
        ).detach().cpu().numpy()

        batch_size_local = batch_right.shape[0]
        for n in range(batch_size_local):
            right_flag = int(batch_right[n].cpu().numpy())
            verts = out["pred_vertices"][n].detach().cpu().numpy()
            verts[:, 0] = (2 * right_flag - 1) * verts[:, 0]
            cam_t = pred_cam_t_full[n]
            pred_keypoints_3d = out["pred_keypoints_3d"][n].detach().cpu().numpy().copy()
            pred_keypoints_3d[:, 0] = (2 * right_flag - 1) * pred_keypoints_3d[:, 0]
            pred_keypoints_2d = out["pred_keypoints_2d"][n].detach().cpu().numpy()
            mano_params = out["pred_mano_params"]
            global_idx = len(all_hand_outputs)
            meta = candidate_metas[global_idx] if candidate_metas is not None and global_idx < len(candidate_metas) else {}

            all_verts.append(verts)
            all_cam_t.append(cam_t)
            all_right.append(right_flag)
            all_hand_outputs.append(
                {
                    "track_id": meta.get("track_id"),
                    "hand_index": meta.get("hand_index"),
                    "source": meta.get("source"),
                    "bbox_square": [float(v) for v in meta.get("bbox", [])] if meta.get("bbox") is not None else None,
                    "pred_cam_crop_raw": pred_cam_raw[n].astype(float).tolist(),
                    "pred_cam_crop": pred_cam_np[n].astype(float).tolist(),
                    "cam_t_full": np.asarray(cam_t).astype(float).tolist(),
                    "is_right": int(right_flag),
                    "mano": {
                        "global_orient_rotmat": mano_params["global_orient"][n].detach().cpu().numpy().astype(float).tolist(),
                        "hand_pose_rotmat": mano_params["hand_pose"][n].detach().cpu().numpy().astype(float).tolist(),
                        "betas": mano_params["betas"][n].detach().cpu().numpy().astype(float).tolist(),
                    },
                    "pred_keypoints_3d": pred_keypoints_3d.astype(float).tolist(),
                    "pred_keypoints_2d_crop": pred_keypoints_2d.astype(float).tolist(),
                    "verts_min": np.min(verts, axis=0).astype(float).tolist(),
                    "verts_max": np.max(verts, axis=0).astype(float).tolist(),
                }
            )

    if len(all_verts) == 0:
        return None, {
            "model_inference_s": total_model_s,
            "render_single_s": 0.0,
            "render_multi_s": 0.0,
            "render_fallback_compose_s": 0.0,
            "hands": [],
        }

    misc_args = dict(
        mesh_base_color=LIGHT_BLUE,
        scene_bg_color=(1, 1, 1),
        focal_length=scaled_focal_length,
    )

    input_img_rgb = square_bgr.astype(np.float32)[:, :, ::-1] / 255.0
    input_img_rgba = np.concatenate([input_img_rgb, np.ones_like(input_img_rgb[:, :, :1])], axis=2)

    single_views = []
    stage_start = time.perf_counter()
    for verts_np, cam_t_np, right_flag in zip(all_verts, all_cam_t, all_right):
        single_view = renderer.render_rgba(
            verts_np,
            cam_t=np.asarray(cam_t_np).copy(),
            render_res=[square_bgr.shape[1], square_bgr.shape[0]],
            is_right=int(right_flag),
            **misc_args,
        )
        single_views.append(single_view)
    total_render_single_s += float(time.perf_counter() - stage_start)

    stage_start = time.perf_counter()
    cam_view_multi = renderer.render_rgba_multiple(
        all_verts,
        cam_t=all_cam_t,
        render_res=[square_bgr.shape[1], square_bgr.shape[0]],
        is_right=all_right,
        **misc_args,
    )
    total_render_multi_s += float(time.perf_counter() - stage_start)

    cam_view = cam_view_multi
    stage_start = time.perf_counter()
    multi_alpha = cam_view_multi[:, :, 3]
    if float(multi_alpha.sum()) <= 1e-6:
        valid_single = [view for view in single_views if float(view[:, :, 3].sum()) > 1e-6]
        if len(valid_single) > 0:
            fallback_view = np.zeros_like(valid_single[0])
            for single_view in valid_single:
                single_alpha = single_view[:, :, 3:]
                fallback_view[:, :, :3] = (
                    fallback_view[:, :, :3] * (1 - single_alpha)
                    + single_view[:, :, :3] * single_alpha
                )
                fallback_view[:, :, 3:] = np.maximum(fallback_view[:, :, 3:], single_alpha)
            cam_view = fallback_view
    total_render_fallback_s += float(time.perf_counter() - stage_start)

    overlay_square_rgb = input_img_rgba[:, :, :3] * (1 - cam_view[:, :, 3:]) + cam_view[:, :, :3] * cam_view[:, :, 3:]
    render_debug = {
        "num_hands": len(all_verts),
        "is_right": [int(x) for x in all_right],
        "alpha_sum": float(cam_view[:, :, 3].sum()),
        "alpha_bbox": alpha_mask_bbox(cam_view[:, :, 3]),
        "cam_t": [np.asarray(x).astype(float).tolist() for x in all_cam_t],
        "model_inference_s": total_model_s,
        "render_single_s": total_render_single_s,
        "render_multi_s": total_render_multi_s,
        "render_fallback_compose_s": total_render_fallback_s,
        "hands": all_hand_outputs,
    }
    return overlay_square_rgb, render_debug


def serialize_candidate_list(candidates):
    serialized = []
    for candidate in candidates:
        item = {}
        for key, value in candidate.items():
            if key == "bbox":
                item[key] = [float(v) for v in value]
            elif isinstance(value, np.ndarray):
                item[key] = value.astype(float).tolist()
            elif isinstance(value, (np.integer, np.floating)):
                item[key] = value.item()
            elif isinstance(value, list):
                converted = []
                for elem in value:
                    if isinstance(elem, (np.integer, np.floating)):
                        converted.append(elem.item())
                    else:
                        converted.append(elem)
                item[key] = converted
            else:
                item[key] = value
        serialized.append(item)
    return serialized


def process_frame(
    frame_bgr,
    mp_module,
    mp_hand_detector,
    model,
    cfg,
    renderer,
    device,
    args,
    timestamp_ms=None,
    focal_length=None,
    temporal_state=None,
    frame_idx=0,
):
    prepared = prepare_frame_candidates(
        frame_bgr,
        mp_module,
        mp_hand_detector,
        model,
        cfg,
        device,
        args,
        timestamp_ms=timestamp_ms,
    )
    timing = dict(prepared["timing"])
    original_frame_bgr = prepared["original_frame_bgr"]
    orig_shape = prepared["orig_shape"]
    square_bgr = prepared["square_bgr"]
    input_res = prepared["input_res"]
    detection_info = prepared["detection_info"]
    candidates_square = prepared["candidates_square"]
    candidates_orig = prepared["candidates_orig"]
    auto_handedness_debug = prepared["auto_handedness_debug"]
    final_is_right = prepared["final_is_right"]
    temporal_debug = None

    if args.temporal_tracking:
        candidates_orig, candidates_square, final_is_right, temporal_state, temporal_debug = apply_temporal_hand_filtering(
            candidates_orig,
            candidates_square,
            temporal_state,
            frame_idx=frame_idx,
            birth_frames=args.temporal_birth_frames,
            max_missed=args.temporal_max_missed,
            duplicate_iou=args.temporal_duplicate_iou,
            duplicate_center_ratio=args.temporal_duplicate_center_ratio,
            second_hand_birth_frames=args.temporal_second_hand_birth_frames,
            second_hand_max_motion_norm=args.temporal_second_hand_max_motion_norm,
            second_hand_max_scale_change=args.temporal_second_hand_max_scale_change,
            handedness_window=args.temporal_handedness_window,
            handedness_min_margin=args.temporal_handedness_min_margin,
            handedness_stability_bias=args.temporal_handedness_stability_bias,
            handedness_flip_streak=args.temporal_handedness_flip_streak,
        )

    bbox_debug_frame = draw_hand_bbox_debug(
        original_frame_bgr if args.mirror_input else frame_bgr,
        [candidate["bbox"] for candidate in candidates_orig],
        [candidate["is_right"] for candidate in candidates_orig],
    )

    if len(candidates_square) == 0:
        frame_debug = {
            "num_hands_detected": 0,
            "fallback_used": bool(detection_info["fallback_used"]),
            "fallback_source": detection_info["fallback_source"],
            "candidates": [],
            "auto_handedness_debug": [],
            "temporal_debug": temporal_debug,
            "mirror_input": bool(args.mirror_input),
            "timing": timing,
        }
        return original_frame_bgr.copy(), bbox_debug_frame, frame_debug, temporal_state

    boxes = np.stack([candidate["bbox"] for candidate in candidates_square])
    right = np.asarray(final_is_right, dtype=np.float32)
    active_focal_length = float(args.focal_length if focal_length is None else focal_length)
    scaled_focal_length = active_focal_length * args.render_res / input_res

    overlay_square_rgb, render_debug = render_hamer_on_square(
        model,
        cfg,
        renderer,
        device,
        square_bgr,
        boxes,
        right,
        scaled_focal_length,
        candidate_metas=candidates_square,
        batch_size=args.batch_size,
    )
    timing["model_inference_s"] = render_debug["model_inference_s"]
    timing["render_single_s"] = render_debug["render_single_s"]
    timing["render_multi_s"] = render_debug["render_multi_s"]
    timing["render_fallback_compose_s"] = render_debug["render_fallback_compose_s"]

    overlay_orig_rgb = crop_square_overlay_back_to_original(overlay_square_rgb, orig_shape, args.render_res)
    render_frame_bgr = np.clip(overlay_orig_rgb[:, :, ::-1] * 255.0, 0, 255).astype(np.uint8)
    if args.mirror_input:
        render_frame_bgr = cv2.flip(render_frame_bgr, 1)

    frame_debug = {
        "num_hands_detected": int(len(candidates_square)),
        "fallback_used": bool(detection_info["fallback_used"]),
        "fallback_source": detection_info["fallback_source"],
        "candidates": serialize_candidate_list(candidates_orig),
        "auto_handedness_debug": auto_handedness_debug,
        "temporal_debug": temporal_debug,
        "mirror_input": bool(args.mirror_input),
        "render": {
            "num_hands": render_debug["num_hands"],
            "is_right": render_debug["is_right"],
            "alpha_sum": render_debug["alpha_sum"],
            "alpha_bbox": render_debug["alpha_bbox"],
            "cam_t": render_debug["cam_t"],
            "input_focal_length": float(active_focal_length),
            "scaled_focal_length": float(scaled_focal_length),
            "hands": render_debug["hands"],
        },
        "timing": timing,
    }
    return render_frame_bgr, bbox_debug_frame, frame_debug, temporal_state


def open_video_writer(path, width, height, fps):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {path}")
    return writer


def collect_video_paths(video_path=None, video_folder=None):
    if video_path is not None:
        return [Path(video_path)]
    return sorted(Path(video_folder).glob("*.mp4"))


def main():
    parser = argparse.ArgumentParser(description="MediaPipe + HaMeR video renderer")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--video_path", type=str, default=None, help="Single input MP4 file")
    input_group.add_argument("--video_folder", type=str, default=None, help="Folder containing MP4 videos")
    parser.add_argument("--out_folder", type=str, default="out_video_demo", help="Output folder for rendered videos and debug files")
    parser.add_argument("--hamer_ckpt", type=str, default="downloads/_DATA/hamer_ckpts/checkpoints/hamer.ckpt", help="Path to HaMeR checkpoint")
    parser.add_argument("--mediapipe_model", type=str, default=DEFAULT_MEDIAPIPE_MODEL, help="Path to MediaPipe hand_landmarker.task model")
    parser.add_argument("--mediapipe_num_hands", type=int, default=2, help="Maximum number of hands for MediaPipe to detect")
    parser.add_argument("--mirror_input", action="store_true", help="Flip each frame horizontally before MediaPipe/HaMeR, then flip outputs back to the original orientation")
    parser.add_argument("--mediapipe_swap_hands", action="store_true", help="Swap MediaPipe handedness labels before HaMeR fitting")
    parser.add_argument("--mediapipe_auto_handedness", dest="mediapipe_auto_handedness", action="store_true", help="Try both left/right in HaMeR and keep the lower-error fit")
    parser.add_argument("--no_mediapipe_auto_handedness", dest="mediapipe_auto_handedness", action="store_false", help="Disable HaMeR-based handedness correction")
    parser.add_argument("--mediapipe_handedness_score_threshold", type=float, default=0.75, help="Keep MediaPipe handedness when its score is at least this high")
    parser.add_argument("--mediapipe_auto_handedness_margin_threshold", type=float, default=0.05, help="Only override MediaPipe handedness when HaMeR left/right error gap is at least this large")
    parser.add_argument("--temporal_tracking", dest="temporal_tracking", action="store_true", help="Use previous-frame track matching to suppress duplicate hands and short-lived false positives")
    parser.add_argument("--no_temporal_tracking", dest="temporal_tracking", action="store_false", help="Disable temporal hand filtering and render raw per-frame detections")
    parser.add_argument("--temporal_birth_frames", type=int, default=2, help="New hands must persist for this many frames before they are rendered")
    parser.add_argument("--temporal_max_missed", type=int, default=2, help="Keep confirmed hands alive for this many missed frames using the previous bbox")
    parser.add_argument("--temporal_duplicate_iou", type=float, default=0.2, help="Suppress new same-frame candidates when their bbox IoU exceeds this threshold")
    parser.add_argument("--temporal_duplicate_center_ratio", type=float, default=0.45, help="Suppress new same-frame candidates when centers are this close relative to hand size")
    parser.add_argument("--temporal_second_hand_birth_frames", type=int, default=5, help="Require the second simultaneous hand to persist for this many frames before rendering it")
    parser.add_argument("--temporal_second_hand_max_motion_norm", type=float, default=0.6, help="Reject a new second-hand track if its frame-to-frame center motion is too jumpy relative to hand size")
    parser.add_argument("--temporal_second_hand_max_scale_change", type=float, default=1.8, help="Reject a new second-hand track if its bbox scale changes too abruptly")
    parser.add_argument("--temporal_handedness_window", type=int, default=5, help="Smooth per-track handedness over the last N observations")
    parser.add_argument("--temporal_handedness_min_margin", type=float, default=0.12, help="Only flip smoothed handedness when the recent vote margin is at least this large")
    parser.add_argument("--temporal_handedness_stability_bias", type=float, default=0.25, help="Small bias toward the previous smoothed handedness to reduce left/right flicker")
    parser.add_argument("--temporal_handedness_flip_streak", type=int, default=3, help="Require this many consecutive raw handedness observations before the smoothed label is allowed to flip")
    parser.add_argument("--render_res", type=int, default=840, help="Square render resolution used internally by HaMeR")
    parser.add_argument("--focal_length", type=float, default=1000, help="Approximate camera focal length")
    parser.add_argument("--auto_focal_search", action="store_true", help="Estimate a better focal length from the first few detected video frames before rendering")
    parser.add_argument("--auto_focal_candidates", type=str, default="700,900,1000,1200,1400", help="Comma-separated focal-length candidates to evaluate during auto focal search")
    parser.add_argument("--auto_focal_frames", type=int, default=10, help="Number of detected frames to use when searching for the best focal length")
    parser.add_argument("--auto_focal_stride", type=int, default=5, help="Only inspect every Nth frame during auto focal search")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for HaMeR inference")
    parser.add_argument("--max_frames", type=int, default=-1, help="Only process the first N frames per video; use -1 for all frames")
    parser.set_defaults(mediapipe_auto_handedness=True, temporal_tracking=True)
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    out_root = Path(args.out_folder)
    out_root.mkdir(parents=True, exist_ok=True)

    model, cfg = load_hamer(args.hamer_ckpt)
    model = model.to(device)
    model.eval()
    renderer = Renderer(cfg, faces=model.mano.faces)

    video_paths = collect_video_paths(args.video_path, args.video_folder)
    if len(video_paths) == 0:
        raise FileNotFoundError("No MP4 videos found to process")

    for video_path in video_paths:
        mp_module, mp_hand_detector = create_mediapipe_hand_detector(
            args.mediapipe_model,
            num_hands=args.mediapipe_num_hands,
            running_mode="video",
        )
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open video: {video_path}")

            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            stem = video_path.stem
            render_video_path = out_root / f"{stem}_render.mp4"
            bbox_video_path = out_root / f"{stem}_bbox.mp4"
            frame_debug_path = out_root / f"{stem}_frames.jsonl"
            frame_params_dir = out_root / f"{stem}_frame_params"
            summary_path = out_root / f"{stem}_summary.json"
            frame_params_dir.mkdir(parents=True, exist_ok=True)
            active_focal_length = float(args.focal_length)
            auto_focal_debug = None

            if args.auto_focal_search:
                active_focal_length, auto_focal_debug = auto_select_focal_for_video(
                    video_path,
                    mp_module,
                    mp_hand_detector,
                    model,
                    cfg,
                    device,
                    args,
                )
                print(f"[{stem}] selected focal_length={active_focal_length:.2f}")
                mp_hand_detector.close()
                mp_module, mp_hand_detector = create_mediapipe_hand_detector(
                    args.mediapipe_model,
                    num_hands=args.mediapipe_num_hands,
                    running_mode="video",
                )

            render_writer = open_video_writer(render_video_path, width, height, fps)
            bbox_writer = open_video_writer(bbox_video_path, width, height, fps)

            processed_frames = 0
            detected_frames = 0
            missed_frames = []
            fallback_frames = []
            time_totals = {}
            temporal_state = {"tracks": [], "next_track_id": 0}

            with open(frame_debug_path, "w", encoding="utf-8") as debug_f:
                progress_total = total_frames if args.max_frames <= 0 else min(total_frames, args.max_frames)
                with tqdm(total=progress_total, desc=stem, unit="frame") as pbar:
                    frame_idx = 0
                    while True:
                        ok, frame_bgr = cap.read()
                        if not ok:
                            break
                        if args.max_frames > 0 and frame_idx >= args.max_frames:
                            break

                        frame_start = time.perf_counter()
                        render_frame_bgr, bbox_debug_frame, frame_debug, temporal_state = process_frame(
                            frame_bgr,
                            mp_module,
                            mp_hand_detector,
                            model,
                            cfg,
                            renderer,
                            device,
                            args,
                            timestamp_ms=int(round(frame_idx * 1000.0 / fps)),
                            focal_length=active_focal_length,
                            temporal_state=temporal_state,
                            frame_idx=frame_idx,
                        )
                        frame_debug["frame_idx"] = int(frame_idx)
                        frame_debug["timestamp_s"] = float(frame_idx / fps)
                        frame_debug["total_s"] = float(time.perf_counter() - frame_start)

                        render_writer.write(render_frame_bgr)
                        bbox_writer.write(bbox_debug_frame)
                        debug_f.write(json.dumps(frame_debug, ensure_ascii=False) + "\n")
                        frame_json_path = frame_params_dir / f"{frame_idx:06d}.json"
                        with open(frame_json_path, "w", encoding="utf-8") as frame_f:
                            json.dump(frame_debug, frame_f, indent=2, ensure_ascii=False)

                        processed_frames += 1
                        if frame_debug["num_hands_detected"] > 0:
                            detected_frames += 1
                        else:
                            missed_frames.append(int(frame_idx))
                        if frame_debug["fallback_used"] and frame_debug["num_hands_detected"] > 0:
                            fallback_frames.append(
                                {
                                    "frame_idx": int(frame_idx),
                                    "fallback_source": frame_debug["fallback_source"],
                                }
                            )

                        for key, value in frame_debug["timing"].items():
                            time_totals[key] = time_totals.get(key, 0.0) + float(value)
                        time_totals["total_s"] = time_totals.get("total_s", 0.0) + float(frame_debug["total_s"])

                        frame_idx += 1
                        pbar.update(1)

            render_writer.release()
            bbox_writer.release()
            cap.release()

            summary = {
                "video_path": str(video_path),
                "render_video_path": str(render_video_path),
                "bbox_video_path": str(bbox_video_path),
                "frame_debug_path": str(frame_debug_path),
                "frame_params_dir": str(frame_params_dir),
                "fps": float(fps),
                "frame_size": [int(width), int(height)],
                "mirror_input": bool(args.mirror_input),
                "selected_focal_length": float(active_focal_length),
                "default_focal_length": float(args.focal_length),
                "auto_focal_search": auto_focal_debug,
                "temporal_tracking": {
                    "enabled": bool(args.temporal_tracking),
                    "birth_frames": int(args.temporal_birth_frames),
                    "max_missed": int(args.temporal_max_missed),
                    "duplicate_iou": float(args.temporal_duplicate_iou),
                    "duplicate_center_ratio": float(args.temporal_duplicate_center_ratio),
                    "second_hand_birth_frames": int(args.temporal_second_hand_birth_frames),
                    "second_hand_max_motion_norm": float(args.temporal_second_hand_max_motion_norm),
                    "second_hand_max_scale_change": float(args.temporal_second_hand_max_scale_change),
                    "handedness_window": int(args.temporal_handedness_window),
                    "handedness_min_margin": float(args.temporal_handedness_min_margin),
                    "handedness_stability_bias": float(args.temporal_handedness_stability_bias),
                    "handedness_flip_streak": int(args.temporal_handedness_flip_streak),
                },
                "total_frames_in_video": int(total_frames),
                "processed_frames": int(processed_frames),
                "frames_with_hand": int(detected_frames),
                "frames_without_hand": int(processed_frames - detected_frames),
                "missed_frame_indices": missed_frames,
                "fallback_frames": fallback_frames,
                "timing_totals_s": time_totals,
            }
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

            print(f"Saved render video to {render_video_path}")
            print(f"Saved bbox video to {bbox_video_path}")
            print(f"Saved frame debug to {frame_debug_path}")
            print(f"Saved summary to {summary_path}")
        finally:
            mp_hand_detector.close()


if __name__ == "__main__":
    main()
