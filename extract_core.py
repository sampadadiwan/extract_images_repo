# extract_core.py (debug-instrumented)
import os
import sys
import cv2
import json
import math
import time
import uuid
import numpy as np
import logging
import platform
from typing import List

# -----------------------------------------------------------------------------
# Logging setup
# -----------------------------------------------------------------------------
LOGGER_NAME = "extract_core"
logger = logging.getLogger(LOGGER_NAME)
# Don't configure root logging here; leave that to the caller/app.
# Add a NullHandler so "No handler found" warnings don't appear.
if not logger.handlers:
    logger.addHandler(logging.NullHandler())

def configure_debug(level=logging.DEBUG):
    """
    Call this once in your app to turn on module logging:
      from extract_core import configure_debug
      configure_debug()  # DEBUG level
    """
    logger.setLevel(level)
    # If no handlers besides NullHandler, add a StreamHandler for convenience
    has_real_handler = any(not isinstance(h, logging.NullHandler) for h in logger.handlers)
    if not has_real_handler:
        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(level)
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s", "%H:%M:%S"
        )
        sh.setFormatter(fmt)
        logger.addHandler(sh)
    _log_env_once()

_ENV_LOGGED = False
def _log_env_once():
    global _ENV_LOGGED
    if _ENV_LOGGED:
        return
    _ENV_LOGGED = True
    try:
        import cv2 as _cv2
        cv_ver = _cv2.__version__
    except Exception:
        cv_ver = "unknown"

    try:
        import numpy as _np
        np_ver = _np.__version__
    except Exception:
        np_ver = "unknown"

    tesseract_cmd = os.environ.get("TESSERACT_CMD") or "env:not_set"
    which_tess = _which("tesseract")

    info = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "process_id": os.getpid(),
        "cwd": os.getcwd(),
        "opencv_version": cv_ver,
        "numpy_version": np_ver,
        "streamlit_env": bool(os.environ.get("STREAMLIT_SERVER_PORT")),
        "uvicorn_env": "uvicorn" in " ".join(sys.argv).lower(),
        "TESSERACT_CMD_env": tesseract_cmd,
        "tesseract_which": which_tess,
    }
    logger.debug(f"[ENV] {json.dumps(info, indent=2)}")

def _which(cmd: str):
    """
    Minimal 'which' for Windows + *nix to help spot path differences
    between Streamlit and Uvicorn.
    """
    exts = os.environ.get("PATHEXT", "").split(os.pathsep) if os.name == "nt" else [""]
    paths = os.environ.get("PATH", "").split(os.pathsep)
    for p in paths:
        full = os.path.join(p, cmd)
        for ext in exts:
            candidate = full + ext
            if os.path.isfile(candidate):
                return candidate
    return None

# -----------------------------------------------------------------------------
# OCR is optional; if not present, text filter becomes a no-op
# -----------------------------------------------------------------------------
try:
    import pytesseract
    _HAS_OCR = True
    try:
        _OCR_VER = pytesseract.get_tesseract_version()
    except Exception:
        _OCR_VER = None
except Exception:
    pytesseract = None
    _HAS_OCR = False
    _OCR_VER = None

def _img_stats(name: str, arr: np.ndarray):
    if arr is None:
        logger.debug(f"[IMG] {name}: None")
        return
    h, w = arr.shape[:2]
    ch = 1 if arr.ndim == 2 else arr.shape[2]
    info = {
        "shape": (h, w, ch),
        "dtype": str(arr.dtype),
        "min": float(arr.min()) if arr.size else None,
        "max": float(arr.max()) if arr.size else None,
        "mean": float(arr.mean()) if arr.size else None,
        "nonzero": int(np.count_nonzero(arr)),
    }
    logger.debug(f"[IMG] {name}: {json.dumps(info)}")

def _bbox_str(x, y, w, h):
    return f"(x={x}, y={y}, w={w}, h={h}, area={w*h})"

def _reason(tag: str, **kwargs):
    logger.debug(f"[SKIP] {tag} :: {json.dumps(kwargs)}")

# -----------------------------------------------------------------------------
# Text ratio estimator
# -----------------------------------------------------------------------------
def text_ratio(img_bgr) -> float:
    """
    Estimate how text-heavy a crop is (0..1) using the same heuristic as your Streamlit app.
    If OCR isn't available, returns 0.0 so nothing is filtered out by text.
    """
    start = time.time()
    if not _HAS_OCR:
        logger.debug("[OCR] pytesseract not available; returning 0.0")
        return 0.0
    try:
        if _OCR_VER is not None:
            logger.debug(f"[OCR] tesseract version: {str(_OCR_VER)}")
        logger.debug(f"[OCR] pytesseract cmd: {getattr(pytesseract, 'tesseract_cmd', None)}")
        _img_stats("text_ratio.input_bgr", img_bgr)

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        _img_stats("text_ratio.gray", gray)

        _, bw = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        _img_stats("text_ratio.bw_thresh200_inv", bw)

        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        _img_stats("text_ratio.bw_open_2x2", bw)

        data = pytesseract.image_to_data(bw, output_type=pytesseract.Output.DICT, config="--psm 6")
        n = len(data.get("text", [])) if isinstance(data, dict) else -1
        logger.debug(f"[OCR] image_to_data entries: {n}")

        if n <= 0:
            return 0.0

        H, W = bw.shape[:2]
        page_area = H * W
        total_text_area, char_boxes = 0, 0
        for i in range(n):
            conf_str = data["conf"][i]
            try:
                conf = int(conf_str) if conf_str not in ("-1", "", None) else -1
            except Exception:
                conf = -1
            txt = (data["text"][i] or "").strip()
            if conf >= 60 and any(c.isalnum() for c in txt):
                w_i = data["width"][i]; h_i = data["height"][i]
                total_text_area += w_i * h_i
                char_boxes += 1

        coverage = total_text_area / (page_area + 1e-6)
        score = float(0.7 * coverage + 0.3 * min(char_boxes / 50.0, 1.0))
        dur = (time.time() - start) * 1000.0
        logger.debug(f"[OCR] coverage={coverage:.6f}, char_boxes={char_boxes}, score={score:.6f}, ms={dur:.1f}")
        return score
    except Exception as e:
        logger.exception(f"[OCR] Exception in text_ratio: {e}")
        return 0.0

# -----------------------------------------------------------------------------
# Color mask builders
# -----------------------------------------------------------------------------
def build_color_mask(img_bgr, sat_thr=28, v_min=80):
    """Mask colored pixels: keeps illustrations/photos, drops grey watermark/text."""
    logger.debug(f"[MASK] build_color_mask(sat_thr={sat_thr}, v_min={v_min})")
    _img_stats("mask.input_bgr", img_bgr)

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    S, V = hsv[..., 1], hsv[..., 2]
    _img_stats("mask.S", S)
    _img_stats("mask.V", V)

    mask = ((S > sat_thr) & (V > v_min)).astype(np.uint8) * 255
    _img_stats("mask.raw", mask)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    _img_stats("mask.open3x3", mask)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8), iterations=1)
    _img_stats("mask.close11x11", mask)

    nz = int(np.count_nonzero(mask))
    logger.debug(f"[MASK] nonzero after CLOSE: {nz}")
    return mask

def merge_mask(mask, merge_pct=2.0):
    """Dilate to fuse nearby blobs into one region (e.g., many tiny elements)."""
    H, W = mask.shape[:2]
    k = max(9, int(min(H, W) * (merge_pct / 100.0)))
    if k % 2 == 0:
        k += 1  # kernel must be odd
    logger.debug(f"[MERGE] merge_pct={merge_pct}, kernel={k}x{k}")

    before = int(np.count_nonzero(mask))
    merge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    merged = cv2.dilate(mask, merge_kernel, iterations=1)
    merged = cv2.morphologyEx(
        merged,
        cv2.MORPH_CLOSE,
        np.ones((max(3, k // 3), max(3, k // 3)), np.uint8),
        iterations=1
    )
    after = int(np.count_nonzero(merged))
    logger.debug(f"[MERGE] nonzero before={before}, after={after}")
    return merged

# -----------------------------------------------------------------------------
# Main pipeline (Streamlit parity)
# -----------------------------------------------------------------------------
def extract_crops_streamlit(
    img_bgr,
    sat_thr: int = 28,
    v_min: int = 80,
    merge_pct: float = 2.0,
    min_area_ratio: float = 0.004,
    max_area_ratio: float = 0.90,
    text_threshold: float = 0.08,
    bottom_trim_pct: int = 8,
    pad: int = 6
) -> List[np.ndarray]:
    """
    Return list of RGB crops using the same pipeline/parameters as the Streamlit app.
    Debug logs explain every decision so you can compare Streamlit vs Uvicorn runs.
    """
    _log_env_once()
    run_id = str(uuid.uuid4())[:8]
    logger.debug(
        "[BEGIN] extract_crops_streamlit "
        + json.dumps(
            dict(
                run_id=run_id,
                sat_thr=sat_thr,
                v_min=v_min,
                merge_pct=merge_pct,
                min_area_ratio=min_area_ratio,
                max_area_ratio=max_area_ratio,
                text_threshold=text_threshold,
                bottom_trim_pct=bottom_trim_pct,
                pad=pad,
                has_ocr=_HAS_OCR,
            )
        )
    )

    if img_bgr is None or not hasattr(img_bgr, "shape"):
        logger.error("[ERROR] img_bgr is None or invalid")
        return []

    H, W = img_bgr.shape[:2]
    page_area = H * W
    _img_stats("page.input_bgr", img_bgr)
    logger.debug(f"[PAGE] size=({H}x{W}), area={page_area}")

    mask = build_color_mask(img_bgr, sat_thr, v_min)
    merged = merge_mask(mask, merge_pct)
    _img_stats("page.mask", mask)
    _img_stats("page.merged", merged)

    cnts, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    logger.debug(f"[CNT] found={len(cnts)} external contours")

    crops = []
    kept = 0
    skipped = 0

    # Sort contours left-to-right, top-to-bottom for deterministic order
    boxes = [cv2.boundingRect(c) for c in cnts]
    order = sorted(range(len(boxes)), key=lambda i: (boxes[i][1] // 10, boxes[i][0] // 10))

    for idx in order:
        c = cnts[idx]
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        aspect = w / float(h) if h > 0 else math.inf

        logger.debug(f"[BOX] idx={idx} { _bbox_str(x,y,w,h) }, aspect={aspect:.3f}")

        if not (min_area_ratio * page_area <= area <= max_area_ratio * page_area):
            _reason("area_out_of_range", idx=idx, area=area, page_area=page_area,
                    min_area=min_area_ratio * page_area, max_area=max_area_ratio * page_area)
            skipped += 1
            continue

        if not (0.2 <= aspect <= 5.0):
            _reason("aspect_out_of_range", idx=idx, aspect=aspect)
            skipped += 1
            continue

        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(W, x + w + pad), min(H, y + h + pad)

        trim = int((bottom_trim_pct / 100.0) * (y1 - y0))
        y1_trimmed = max(y0 + 1, y1 - trim)

        if x1 <= x0 or y1_trimmed <= y0:
            _reason("invalid_trim_or_pad", idx=idx, x0=x0, y0=y0, x1=x1, y1=y1_trimmed)
            skipped += 1
            continue

        crop = img_bgr[y0:y1_trimmed, x0:x1].copy()
        _img_stats(f"crop[{idx}].bgr_before_text_filter", crop)

        if crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 20:
            _reason("too_small", idx=idx, h=int(crop.shape[0]) if crop.size else -1,
                    w=int(crop.shape[1]) if crop.size else -1)
            skipped += 1
            continue

        score = text_ratio(crop)
        if score > text_threshold:
            _reason("text_ratio_high", idx=idx, score=score, threshold=text_threshold)
            skipped += 1
            continue

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crops.append(rgb)
        kept += 1
        logger.debug(f"[KEEP] idx={idx} -> crop #{kept}  { _bbox_str(x0, y0, x1-x0, y1_trimmed-y0) }")

    logger.debug(f"[END] run_id={run_id}, kept={kept}, skipped={skipped}, total={len(cnts)}")
    return crops

# -----------------------------------------------------------------------------
# Optional: quick self-test (run `python extract_core.py` to verify logging)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    configure_debug()  # enable DEBUG when run as a script
    logger.info("Self-test mode: generating a dummy image")
    # Make a simple synthetic image with colored blocks and some gray text-like lines
    test = np.full((600, 800, 3), 255, np.uint8)
    cv2.rectangle(test, (50, 50), (350, 300), (0, 128, 255), -1)  # orange block
    cv2.rectangle(test, (420, 80), (760, 260), (60, 180, 60), -1)  # green block
    # fake gray text stripes
    for i in range(340, 380, 4):
        cv2.line(test, (60, i), (740, i), (200, 200, 200), 1)

    out = extract_crops_streamlit(test)
    logger.info(f"Self-test crops: {len(out)}")
