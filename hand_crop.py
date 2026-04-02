from pathlib import Path
import argparse
import json

import cv2
import numpy as np


DEFAULT_MEDIAPIPE_MODEL = str(Path(__file__).resolve().parent / 'mediapipe_model' / 'hand_landmarker.task')
DEFAULT_MIN_HAND_DETECTION_CONFIDENCE = 0.1
DEFAULT_MIN_HAND_PRESENCE_CONFIDENCE = 0.1
DEFAULT_MEDIAPIPE_BBOX_EXPAND_SCALE = 1.25
DEFAULT_MEDIAPIPE_BBOX_MIN_BOX_SIZE = 28
DEFAULT_MEDIAPIPE_BBOX_UP_SHIFT_RATIO = 0.04
MEDIAPIPE_HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]
DEFAULT_EGO_FALLBACK_WINDOWS = [
    ('bottom_center_tight', (0.30, 0.55, 0.70, 1.00)),
    ('bottom_half', (0.00, 0.50, 1.00, 1.00)),
    ('right_half', (0.50, 0.00, 1.00, 1.00)),
    ('bottom_right_70', (0.30, 0.30, 1.00, 1.00)),
    ('bottom_center_70', (0.15, 0.30, 0.85, 1.00)),
]


def draw_hand_bbox_debug(image_bgr, bboxes, is_right):
    vis = image_bgr.copy()
    if len(bboxes) == 0:
        cv2.putText(
            vis,
            'No hand bbox',
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return vis

    for idx, (bbox, right_flag) in enumerate(zip(bboxes, is_right)):
        x1, y1, x2, y2 = [int(v) for v in bbox]
        color = (0, 200, 255) if right_flag else (255, 120, 0)
        label = f'R{idx}' if right_flag else f'L{idx}'
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            vis,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )
    return vis


def draw_mediapipe_hands_debug(image_bgr, detection_result):
    vis = image_bgr.copy()
    if detection_result is None or not getattr(detection_result, 'hand_landmarks', None):
        cv2.putText(
            vis,
            'No MediaPipe hand',
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return vis

    img_h, img_w = vis.shape[:2]
    for idx, hand_landmarks in enumerate(detection_result.hand_landmarks):
        handedness_list = detection_result.handedness[idx] if idx < len(detection_result.handedness) else []
        handedness = handedness_list[0].category_name if len(handedness_list) > 0 else 'Unknown'
        color = (0, 200, 255) if handedness.lower() == 'right' else (255, 120, 0)

        points = []
        for landmark in hand_landmarks:
            x = int(np.clip(landmark.x * img_w, 0, img_w - 1))
            y = int(np.clip(landmark.y * img_h, 0, img_h - 1))
            points.append((x, y))
            cv2.circle(vis, (x, y), 3, color, -1)

        for start_idx, end_idx in MEDIAPIPE_HAND_CONNECTIONS:
            if start_idx < len(points) and end_idx < len(points):
                cv2.line(vis, points[start_idx], points[end_idx], color, 2)

        if len(points) > 0:
            xs = [pt[0] for pt in points]
            ys = [pt[1] for pt in points]
            cv2.putText(
                vis,
                f'{handedness[0].upper()}{idx}' if handedness else f'H{idx}',
                (min(xs), max(20, min(ys) - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
                cv2.LINE_AA,
            )
    return vis


def bbox_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def make_expanded_hand_bbox(
    keypoints,
    image_shape,
    score_thr=0.2,
    min_points=4,
    expand_scale=1.45,
    min_box_size=28,
    support_thr=0.05,
    wrist_shift_ratio_x=0.35,
    wrist_shift_ratio_y=0.12,
    support_radius_ratio=1.6,
    up_shift_ratio=0.06,
):
    valid = keypoints[:, 2] > score_thr
    if valid.sum() < min_points:
        return None, 0.0

    hand_valid = keypoints[1:, 2] > score_thr
    use_non_wrist = hand_valid.sum() >= max(3, min_points - 1)
    if use_non_wrist:
        strong_pts = keypoints[1:, :2][hand_valid]
        scores = keypoints[1:, 2][hand_valid]
        support_mask = keypoints[1:, 2] > support_thr
        support_pts = keypoints[1:, :2][support_mask]
    else:
        strong_pts = keypoints[valid, :2]
        scores = keypoints[valid, 2]
        support_mask = keypoints[:, 2] > support_thr
        support_pts = keypoints[:, :2][support_mask]

    strong_center = strong_pts.mean(axis=0)
    extent_pts = strong_pts.copy()

    if len(support_pts) > 0:
        spread_x = strong_pts[:, 0].max() - strong_pts[:, 0].min()
        spread_y = strong_pts[:, 1].max() - strong_pts[:, 1].min()
        support_radius = max(spread_x, spread_y, min_box_size) * support_radius_ratio
        support_dist = np.linalg.norm(support_pts - strong_center[None, :], axis=1)
        filtered_support = support_pts[support_dist <= support_radius]
        if len(filtered_support) > 0:
            extent_pts = np.concatenate([extent_pts, filtered_support], axis=0)

    x1, y1 = extent_pts.min(axis=0)
    x2, y2 = extent_pts.max(axis=0)
    width = x2 - x1
    height = y2 - y1
    box_size = max(width, height, min_box_size) * expand_scale

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    if use_non_wrist and keypoints[0, 2] > support_thr:
        wrist_pt = keypoints[0, :2]
        wrist_offset = wrist_pt - strong_center
        max_shift_x = 0.2 * box_size
        max_shift_y = 0.12 * box_size
        cx += float(np.clip(wrist_offset[0] * wrist_shift_ratio_x, -max_shift_x, max_shift_x))
        cy += float(np.clip(wrist_offset[1] * wrist_shift_ratio_y, -max_shift_y, max_shift_y))

    cy -= up_shift_ratio * box_size

    half = box_size / 2.0
    img_h, img_w = image_shape[:2]
    x1 = max(0.0, cx - half)
    y1 = max(0.0, cy - half)
    x2 = min(float(img_w - 1), cx + half)
    y2 = min(float(img_h - 1), cy + half)

    return [x1, y1, x2, y2], float(scores.sum())


def dedupe_hand_candidates(candidates, iou_thr=0.3):
    if len(candidates) <= 1:
        return candidates

    kept = []
    for candidate in candidates:
        merged = False
        for idx, existing in enumerate(kept):
            if bbox_iou(candidate['bbox'], existing['bbox']) > iou_thr:
                if candidate['score'] > existing['score']:
                    kept[idx] = candidate
                merged = True
                break
        if not merged:
            kept.append(candidate)
    return kept


def create_mediapipe_hand_detector(
    model_path=DEFAULT_MEDIAPIPE_MODEL,
    num_hands=2,
    min_hand_detection_confidence=DEFAULT_MIN_HAND_DETECTION_CONFIDENCE,
    min_hand_presence_confidence=DEFAULT_MIN_HAND_PRESENCE_CONFIDENCE,
    running_mode='image',
):
    try:
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
    except ImportError as exc:
        raise ImportError(
            'MediaPipe is not installed in the current environment. Install it first, '
            'for example: pip install mediapipe'
        ) from exc

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f'MediaPipe hand model not found: {model_path}')

    running_mode_name = str(running_mode).lower()
    if running_mode_name == 'image':
        running_mode_enum = vision.RunningMode.IMAGE
    elif running_mode_name == 'video':
        running_mode_enum = vision.RunningMode.VIDEO
    else:
        raise ValueError(
            f"Unsupported MediaPipe running_mode={running_mode!r}. "
            "Expected 'image' or 'video'."
        )

    base_options = python.BaseOptions(model_asset_path=str(model_path))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=running_mode_enum,
        num_hands=num_hands,
        min_hand_detection_confidence=min_hand_detection_confidence,
        min_hand_presence_confidence=min_hand_presence_confidence,
    )
    detector = vision.HandLandmarker.create_from_options(options)
    return mp, detector


def detect_mediapipe_hands(mp_module, detector, rgb_image, timestamp_ms=None):
    mp_image = mp_module.Image(
        image_format=mp_module.ImageFormat.SRGB,
        data=np.ascontiguousarray(rgb_image),
    )
    if timestamp_ms is None:
        return detector.detect(mp_image)
    return detector.detect_for_video(mp_image, int(timestamp_ms))


def serialize_mediapipe_result(detection_result):
    serialized = []
    hand_landmarks_list = getattr(detection_result, 'hand_landmarks', [])
    handedness_list = getattr(detection_result, 'handedness', [])

    for idx, hand_landmarks in enumerate(hand_landmarks_list):
        hand_data = {
            'landmarks': [
                {
                    'x': float(landmark.x),
                    'y': float(landmark.y),
                    'z': float(landmark.z),
                }
                for landmark in hand_landmarks
            ],
            'handedness': [],
        }
        if idx < len(handedness_list):
            hand_data['handedness'] = [
                {
                    'category_name': category.category_name,
                    'display_name': category.display_name,
                    'score': float(category.score),
                    'index': int(category.index),
                }
                for category in handedness_list[idx]
            ]
        serialized.append(hand_data)
    return serialized


def roi_ratios_to_xyxy(image_shape, roi_ratios):
    img_h, img_w = image_shape[:2]
    x1 = int(round(img_w * roi_ratios[0]))
    y1 = int(round(img_h * roi_ratios[1]))
    x2 = int(round(img_w * roi_ratios[2]))
    y2 = int(round(img_h * roi_ratios[3]))
    x1 = max(0, min(x1, img_w - 1))
    y1 = max(0, min(y1, img_h - 1))
    x2 = max(x1 + 1, min(x2, img_w))
    y2 = max(y1 + 1, min(y2, img_h))
    return x1, y1, x2, y2


def mapped_hands_from_result(detection_result, image_shape, roi_xyxy=None, source='full'):
    img_h, img_w = image_shape[:2]
    if roi_xyxy is None:
        roi_x1, roi_y1, roi_x2, roi_y2 = 0, 0, img_w, img_h
    else:
        roi_x1, roi_y1, roi_x2, roi_y2 = roi_xyxy

    roi_w = max(1, roi_x2 - roi_x1)
    roi_h = max(1, roi_y2 - roi_y1)

    hands = []
    hand_landmarks_list = getattr(detection_result, 'hand_landmarks', [])
    handedness_list = getattr(detection_result, 'handedness', [])
    for idx, hand_landmarks in enumerate(hand_landmarks_list):
        handedness = []
        if idx < len(handedness_list):
            handedness = [
                {
                    'category_name': category.category_name,
                    'display_name': category.display_name,
                    'score': float(category.score),
                    'index': int(category.index),
                }
                for category in handedness_list[idx]
            ]

        mapped_landmarks = []
        for landmark in hand_landmarks:
            x = float(np.clip(roi_x1 + landmark.x * roi_w, 0, img_w - 1))
            y = float(np.clip(roi_y1 + landmark.y * roi_h, 0, img_h - 1))
            mapped_landmarks.append(
                {
                    'x': x,
                    'y': y,
                    'z': float(landmark.z),
                }
            )
        hands.append(
            {
                'hand_index': idx,
                'landmarks': mapped_landmarks,
                'handedness': handedness,
                'source': source,
                'roi_xyxy': [int(roi_x1), int(roi_y1), int(roi_x2), int(roi_y2)],
            }
        )
    return hands


def draw_mapped_hands_debug(image_bgr, hands):
    vis = image_bgr.copy()
    if len(hands) == 0:
        cv2.putText(
            vis,
            'No MediaPipe hand',
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return vis

    for idx, hand in enumerate(hands):
        handedness_name = hand.get('resolved_handedness')
        if handedness_name is None:
            handedness_name = hand['handedness'][0]['category_name'] if len(hand['handedness']) > 0 else 'Unknown'
        color = (0, 200, 255) if handedness_name.lower() == 'right' else (255, 120, 0)
        points = []
        for landmark in hand['landmarks']:
            x = int(round(landmark['x']))
            y = int(round(landmark['y']))
            points.append((x, y))
            cv2.circle(vis, (x, y), 3, color, -1)

        for start_idx, end_idx in MEDIAPIPE_HAND_CONNECTIONS:
            if start_idx < len(points) and end_idx < len(points):
                cv2.line(vis, points[start_idx], points[end_idx], color, 2)

        if len(points) > 0:
            xs = [pt[0] for pt in points]
            ys = [pt[1] for pt in points]
            cv2.putText(
                vis,
                f"{handedness_name[0].upper() if handedness_name else 'H'}{idx}:{hand['source']}",
                (min(xs), max(20, min(ys) - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )
    return vis


def mapped_hands_to_candidates(
    hands,
    image_shape,
    swap_hands=False,
    image_convention='as_is',
    expand_scale=DEFAULT_MEDIAPIPE_BBOX_EXPAND_SCALE,
    min_box_size=DEFAULT_MEDIAPIPE_BBOX_MIN_BOX_SIZE,
    up_shift_ratio=DEFAULT_MEDIAPIPE_BBOX_UP_SHIFT_RATIO,
):
    candidates = []
    for hand in hands:
        landmarks = hand['landmarks']
        keypoints = np.zeros((len(landmarks), 3), dtype=np.float32)
        for idx, landmark in enumerate(landmarks):
            keypoints[idx, 0] = landmark['x']
            keypoints[idx, 1] = landmark['y']
            keypoints[idx, 2] = 1.0

        bbox, score = make_expanded_hand_bbox(
            keypoints,
            image_shape,
            expand_scale=expand_scale,
            min_box_size=min_box_size,
            up_shift_ratio=up_shift_ratio,
        )
        if bbox is None:
            continue

        handedness_name = 'Right'
        handedness_score = None
        if len(hand['handedness']) > 0:
            handedness_name = hand['handedness'][0]['category_name']
            handedness_score = float(hand['handedness'][0].get('score', 0.0))

        is_right = 1 if str(handedness_name).lower() == 'right' else 0
        if image_convention == 'non_mirrored':
            is_right = 1 - is_right
        if swap_hands:
            is_right = 1 - is_right

        candidates.append(
            {
                'hand_index': hand.get('hand_index', len(candidates)),
                'bbox': bbox,
                'is_right': is_right,
                'raw_is_right': is_right,
                'mediapipe_handedness_name': str(handedness_name),
                'mediapipe_handedness_score': handedness_score,
                'score': score,
                'source': hand['source'],
                'roi_xyxy': hand['roi_xyxy'],
            }
        )
    return candidates


def apply_single_hand_handedness_override(candidates, hands, single_hand_handedness='none'):
    if single_hand_handedness == 'none' or len(candidates) != 1:
        return candidates, hands

    forced_is_right = 1 if single_hand_handedness == 'right' else 0
    hand_index = candidates[0].get('hand_index', 0)
    candidates[0]['is_right'] = forced_is_right
    candidates[0]['handedness_override'] = single_hand_handedness

    for hand in hands:
        if hand.get('hand_index', 0) == hand_index:
            hand['resolved_handedness'] = single_hand_handedness.capitalize()
            hand['resolved_is_right'] = forced_is_right
            break
    return candidates, hands


def detect_mediapipe_hands_with_fallback(
    mp_module,
    detector,
    image_bgr,
    timestamp_ms=None,
    swap_hands=False,
    image_convention='as_is',
    single_hand_handedness='none',
    fallback_windows=DEFAULT_EGO_FALLBACK_WINDOWS,
    bbox_expand_scale=DEFAULT_MEDIAPIPE_BBOX_EXPAND_SCALE,
    bbox_min_box_size=DEFAULT_MEDIAPIPE_BBOX_MIN_BOX_SIZE,
    bbox_up_shift_ratio=DEFAULT_MEDIAPIPE_BBOX_UP_SHIFT_RATIO,
):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    detection_result = detect_mediapipe_hands(
        mp_module,
        detector,
        image_rgb,
        timestamp_ms=timestamp_ms,
    )
    hands = mapped_hands_from_result(detection_result, image_bgr.shape, source='full')
    candidates = mapped_hands_to_candidates(
        hands,
        image_bgr.shape,
        swap_hands=swap_hands,
        image_convention=image_convention,
        expand_scale=bbox_expand_scale,
        min_box_size=bbox_min_box_size,
        up_shift_ratio=bbox_up_shift_ratio,
    )
    candidates = dedupe_hand_candidates(candidates)
    candidates, hands = apply_single_hand_handedness_override(
        candidates,
        hands,
        single_hand_handedness=single_hand_handedness,
    )

    if len(candidates) > 0:
        return {
            'hands': hands,
            'candidates': candidates,
            'fallback_used': False,
            'fallback_source': 'full',
        }

    for fallback_idx, (source_name, roi_ratios) in enumerate(fallback_windows, start=1):
        x1, y1, x2, y2 = roi_ratios_to_xyxy(image_bgr.shape, roi_ratios)
        roi_bgr = image_bgr[y1:y2, x1:x2]
        if roi_bgr.size == 0:
            continue
        roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
        roi_timestamp_ms = None if timestamp_ms is None else int(timestamp_ms) + fallback_idx
        roi_result = detect_mediapipe_hands(
            mp_module,
            detector,
            roi_rgb,
            timestamp_ms=roi_timestamp_ms,
        )
        roi_hands = mapped_hands_from_result(
            roi_result,
            image_bgr.shape,
            roi_xyxy=(x1, y1, x2, y2),
            source=source_name,
        )
        roi_candidates = mapped_hands_to_candidates(
            roi_hands,
            image_bgr.shape,
            swap_hands=swap_hands,
            image_convention=image_convention,
            expand_scale=bbox_expand_scale,
            min_box_size=bbox_min_box_size,
            up_shift_ratio=bbox_up_shift_ratio,
        )
        roi_candidates = dedupe_hand_candidates(roi_candidates)
        roi_candidates, roi_hands = apply_single_hand_handedness_override(
            roi_candidates,
            roi_hands,
            single_hand_handedness=single_hand_handedness,
        )
        if len(roi_candidates) > 0:
            return {
                'hands': roi_hands,
                'candidates': roi_candidates,
                'fallback_used': True,
                'fallback_source': source_name,
            }

    return {
        'hands': [],
        'candidates': [],
        'fallback_used': True,
        'fallback_source': None,
    }


def mediapipe_result_to_candidates(
    detection_result,
    image_shape,
    swap_hands=False,
    image_convention='as_is',
    expand_scale=DEFAULT_MEDIAPIPE_BBOX_EXPAND_SCALE,
    min_box_size=DEFAULT_MEDIAPIPE_BBOX_MIN_BOX_SIZE,
    up_shift_ratio=DEFAULT_MEDIAPIPE_BBOX_UP_SHIFT_RATIO,
):
    candidates = []
    hand_landmarks_list = getattr(detection_result, 'hand_landmarks', [])
    handedness_list = getattr(detection_result, 'handedness', [])
    img_h, img_w = image_shape[:2]

    for idx, hand_landmarks in enumerate(hand_landmarks_list):
        keypoints = np.zeros((len(hand_landmarks), 3), dtype=np.float32)
        for point_idx, landmark in enumerate(hand_landmarks):
            keypoints[point_idx, 0] = float(np.clip(landmark.x * img_w, 0, img_w - 1))
            keypoints[point_idx, 1] = float(np.clip(landmark.y * img_h, 0, img_h - 1))
            keypoints[point_idx, 2] = 1.0

        bbox, score = make_expanded_hand_bbox(
            keypoints,
            image_shape,
            expand_scale=expand_scale,
            min_box_size=min_box_size,
            up_shift_ratio=up_shift_ratio,
        )
        if bbox is None:
            continue

        handedness_name = 'Right'
        if idx < len(handedness_list) and len(handedness_list[idx]) > 0:
            handedness_name = handedness_list[idx][0].category_name

        is_right = 1 if str(handedness_name).lower() == 'right' else 0
        if image_convention == 'non_mirrored':
            is_right = 1 - is_right
        if swap_hands:
            is_right = 1 - is_right

        candidates.append({'bbox': bbox, 'is_right': is_right, 'score': score})
    return candidates


def save_hand_crops(image_bgr, bboxes, is_right, out_dir, stem):
    crop_dir = Path(out_dir) / 'crops'
    crop_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for idx, (bbox, right_flag) in enumerate(zip(bboxes, is_right)):
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        crop = image_bgr[y1:y2, x1:x2]
        label = 'R' if right_flag else 'L'
        crop_path = crop_dir / f'{stem}_{label}{idx}.jpg'
        if crop.size > 0:
            cv2.imwrite(str(crop_path), crop)
            saved_paths.append(str(crop_path))
    return saved_paths


def process_single_image(
    img_path,
    out_dir,
    mp_module,
    detector,
    swap_hands=False,
    image_convention='as_is',
    single_hand_handedness='none',
    bbox_expand_scale=DEFAULT_MEDIAPIPE_BBOX_EXPAND_SCALE,
    bbox_min_box_size=DEFAULT_MEDIAPIPE_BBOX_MIN_BOX_SIZE,
    bbox_up_shift_ratio=DEFAULT_MEDIAPIPE_BBOX_UP_SHIFT_RATIO,
):
    img_path = Path(img_path)
    stem = img_path.stem

    image_bgr = cv2.imread(str(img_path))
    if image_bgr is None:
        raise RuntimeError(f'Failed to read image: {img_path}')
    detection_info = detect_mediapipe_hands_with_fallback(
        mp_module,
        detector,
        image_bgr,
        swap_hands=swap_hands,
        image_convention=image_convention,
        single_hand_handedness=single_hand_handedness,
        bbox_expand_scale=bbox_expand_scale,
        bbox_min_box_size=bbox_min_box_size,
        bbox_up_shift_ratio=bbox_up_shift_ratio,
    )
    hands = detection_info['hands']
    candidates = detection_info['candidates']
    bboxes = [item['bbox'] for item in candidates]
    is_right = [item['is_right'] for item in candidates]

    cv2.imwrite(str(out_dir / f'{stem}_mediapipe.jpg'), draw_mapped_hands_debug(image_bgr, hands))
    cv2.imwrite(str(out_dir / f'{stem}_bbox.jpg'), draw_hand_bbox_debug(image_bgr, bboxes, is_right))
    crop_paths = save_hand_crops(image_bgr, bboxes, is_right, out_dir, stem)

    result = {
        'img_path': str(img_path),
        'num_hands': len(bboxes),
        'bboxes': [[float(v) for v in bbox] for bbox in bboxes],
        'is_right': [int(v) for v in is_right],
        'crops': crop_paths,
        'fallback_used': bool(detection_info['fallback_used']),
        'fallback_source': detection_info['fallback_source'],
        'image_convention': image_convention,
        'single_hand_handedness': single_hand_handedness,
        'bbox_expand_scale': float(bbox_expand_scale),
        'bbox_min_box_size': int(bbox_min_box_size),
        'bbox_up_shift_ratio': float(bbox_up_shift_ratio),
        'mediapipe': hands,
    }
    with open(out_dir / f'{stem}.json', 'w') as f:
        json.dump(result, f, indent=2)
    return result


def collect_image_paths(img_path=None, img_folder=None, file_types=None, max_images=-1):
    if img_path is not None:
        paths = [Path(img_path)]
    else:
        paths = sorted([img for pattern in file_types for img in Path(img_folder).glob(pattern)])

    if max_images > 0:
        paths = paths[:max_images]
    return paths


def main():
    parser = argparse.ArgumentParser(description='MediaPipe hand crop utility')
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--img_path', type=str, default=None, help='Path to input image')
    input_group.add_argument('--img_folder', type=str, default=None, help='Folder with input images')
    parser.add_argument('--out_dir', type=str, default='out_mediapipe_crop', help='Directory to save debug images/json/crops')
    parser.add_argument('--model_path', type=str, default=DEFAULT_MEDIAPIPE_MODEL, help='Path to MediaPipe hand_landmarker.task')
    parser.add_argument('--num_hands', type=int, default=2, help='Maximum number of hands to detect')
    parser.add_argument('--image_convention', type=str, default='as_is', choices=['as_is', 'mirrored', 'non_mirrored'], help='How to interpret MediaPipe handedness: keep as-is, or flip for known non-mirrored frames')
    parser.add_argument('--swap_hands', action='store_true', help='Swap left/right handedness labels')
    parser.add_argument('--min_hand_detection_confidence', type=float, default=DEFAULT_MIN_HAND_DETECTION_CONFIDENCE, help='MediaPipe minimum hand detection confidence')
    parser.add_argument('--min_hand_presence_confidence', type=float, default=DEFAULT_MIN_HAND_PRESENCE_CONFIDENCE, help='MediaPipe minimum hand presence confidence')
    parser.add_argument('--single_hand_handedness', type=str, default='none', choices=['none', 'left', 'right'], help='Optional manual override when exactly one hand is detected')
    parser.add_argument('--bbox_expand_scale', type=float, default=DEFAULT_MEDIAPIPE_BBOX_EXPAND_SCALE, help='Expansion factor for hand bbox generated from MediaPipe landmarks')
    parser.add_argument('--bbox_min_box_size', type=int, default=DEFAULT_MEDIAPIPE_BBOX_MIN_BOX_SIZE, help='Minimum square hand bbox size in pixels')
    parser.add_argument('--bbox_up_shift_ratio', type=float, default=DEFAULT_MEDIAPIPE_BBOX_UP_SHIFT_RATIO, help='Shift bbox slightly upward to keep fingertips inside')
    parser.add_argument('--file_type', nargs='+', default=['*.jpg', '*.png'], help='List of file extensions to consider when using --img_folder')
    parser.add_argument('--max_images', type=int, default=-1, help='Only process the first N images after sorting; use -1 for all images')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_paths = collect_image_paths(
        img_path=args.img_path,
        img_folder=args.img_folder,
        file_types=args.file_type,
        max_images=args.max_images,
    )
    if len(image_paths) == 0:
        raise FileNotFoundError('No input images found for hand crop processing')

    mp_module, detector = create_mediapipe_hand_detector(
        args.model_path,
        num_hands=args.num_hands,
        min_hand_detection_confidence=args.min_hand_detection_confidence,
        min_hand_presence_confidence=args.min_hand_presence_confidence,
    )
    try:
        all_results = []
        for image_path in image_paths:
            result = process_single_image(
                image_path,
                out_dir,
                mp_module,
                detector,
                swap_hands=args.swap_hands,
                image_convention=args.image_convention,
                single_hand_handedness=args.single_hand_handedness,
                bbox_expand_scale=args.bbox_expand_scale,
                bbox_min_box_size=args.bbox_min_box_size,
                bbox_up_shift_ratio=args.bbox_up_shift_ratio,
            )
            result['model_path'] = str(Path(args.model_path))
            all_results.append(result)

        with open(out_dir / 'summary.json', 'w') as f:
            json.dump(
                {
                    'model_path': str(Path(args.model_path)),
                    'num_images': len(all_results),
                    'images': all_results,
                },
                f,
                indent=2,
            )
    finally:
        detector.close()


if __name__ == '__main__':
    main()
