# app.py
import io, zipfile
import cv2
import numpy as np
import streamlit as st
from PIL import Image
import pytesseract
from extract_core import extract_crops_streamlit
# ...then call extract_crops_streamlit(img_bgr, sat, v, merge, min_area, max_area, text_thr, bottom_trim, pad)


# --- If you're on Windows, set your Tesseract path (uncomment & update) ---
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

st.set_page_config(page_title="Page Image Extractor", layout="wide")
st.title("Page Image Extractor (OpenCV + Merge + OCR)")

# -------------------- helpers --------------------
def text_ratio(img_bgr) -> float:
    """Estimate how text-heavy a crop is (0..1)."""
    try:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        _, bw = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        data = pytesseract.image_to_data(bw, output_type=pytesseract.Output.DICT, config="--psm 6")
        if len(data["text"]) == 0:
            return 0.0
        H, W = bw.shape[:2]
        page_area = H * W
        total_text_area, char_boxes = 0, 0
        for i in range(len(data["text"])):
            conf = int(data["conf"][i]) if data["conf"][i] != "-1" else -1
            txt = (data["text"][i] or "").strip()
            if conf >= 60 and any(c.isalnum() for c in txt):
                total_text_area += data["width"][i] * data["height"][i]
                char_boxes += 1
        coverage = total_text_area / (page_area + 1e-6)
        return float(0.7 * coverage + 0.3 * min(char_boxes / 50.0, 1.0))
    except Exception:
        return 0.0  # if OCR not available, don't filter

def build_color_mask(img_bgr, sat_thr=28, v_min=80):
    """Mask colored pixels: keeps illustrations/photos, drops grey watermark/text."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    S, V = hsv[..., 1], hsv[..., 2]
    mask = ((S > sat_thr) & (V > v_min)).astype(np.uint8) * 255
    # Gentle open so tiny bits survive; close to tidy edges
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8), iterations=1)
    return mask

def merge_mask(mask, merge_pct=2.0):
    """Dilate to fuse nearby blobs into one region (e.g., many small element photos)."""
    H, W = mask.shape[:2]
    k = max(9, int(min(H, W) * (merge_pct / 100.0)))
    if k % 2 == 0: k += 1  # kernel must be odd
    merge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    merged = cv2.dilate(mask, merge_kernel, iterations=1)
    merged = cv2.morphologyEx(merged, cv2.MORPH_CLOSE, np.ones((max(3, k//3), max(3, k//3)), np.uint8), iterations=1)
    return merged

def extract_crops(img_bgr, sat_thr, v_min, merge_pct, min_area_ratio, max_area_ratio,
                  text_threshold, bottom_trim_pct, pad=6):
    """Return merged mask and list of RGB crops."""
    H, W = img_bgr.shape[:2]
    page_area = H * W

    mask = build_color_mask(img_bgr, sat_thr, v_min)
    merged = merge_mask(mask, merge_pct)

    cnts, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    crops = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if not (min_area_ratio * page_area <= area <= max_area_ratio * page_area):
            continue
        aspect = w / float(h)
        if not (0.2 <= aspect <= 5.0):
            continue

        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(W, x + w + pad), min(H, y + h + pad)

        # Trim a bit from bottom to avoid captions entering OCR
        trim = int((bottom_trim_pct / 100.0) * (y1 - y0))
        y1 = max(y0 + 1, y1 - trim)

        if x1 <= x0 or y1 <= y0:
            continue
        crop = img_bgr[y0:y1, x0:x1].copy()
        if crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 20:
            continue

        if text_ratio(crop) > text_threshold:
            continue

        crops.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    return merged, crops

# -------------------- UI --------------------
uploaded = st.file_uploader("Upload a PNG/JPG page (300 dpi works best)", type=["png", "jpg", "jpeg"])
with st.sidebar:
    st.header("Detection settings")
    sat_thr = st.slider("Min saturation (color mask)", 5, 60, 28)
    v_min   = st.slider("Min brightness (color mask)", 40, 160, 80)
    merge_pct = st.slider("Merge distance (% of min page dim)", 0.5, 6.0, 2.0, 0.1)
    min_area_ratio = st.number_input("Min area ratio", 0.0002, 0.2, 0.004, 0.0002)
    max_area_ratio = st.number_input("Max area ratio", 0.2, 0.99, 0.90, 0.01)
    text_threshold = st.slider("Text filter threshold", 0.00, 0.20, 0.08, 0.01)
    bottom_trim_pct = st.slider("Trim bottom of crop (%)", 0, 20, 8)
    show_masks = st.checkbox("Show masks (mask & merged)", value=False)

if uploaded:
    img_bytes = np.frombuffer(uploaded.getvalue(), dtype=np.uint8)
    img_bgr = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
    if img_bgr is None:
        st.error("Failed to decode the image. Please re-upload.")
        st.stop()

    st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption="Original page", use_column_width=True)

    merged, crops = extract_crops(
        img_bgr, sat_thr, v_min, merge_pct, min_area_ratio, max_area_ratio,
        text_threshold, bottom_trim_pct, pad=6
    )

    if show_masks:
        mask_preview = build_color_mask(img_bgr, sat_thr, v_min)
        st.image(mask_preview, caption="Color mask", use_column_width=True, clamp=True)
        st.image(merged, caption="Merged mask (used for contours)", use_column_width=True, clamp=True)

    st.subheader(f"Extracted crops: {len(crops)}")
    if not crops:
        st.info("No matches. Try lowering Min saturation & Min area ratio, or increasing Merge distance (%).")
    else:
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for i, rgb in enumerate(crops, 1):
                st.image(rgb, caption=f"Crop {i}", use_column_width=True)
                pil = Image.fromarray(rgb)
                buf = io.BytesIO()
                pil.save(buf, format="PNG")
                zf.writestr(f"crop_{i:02d}.png", buf.getvalue())
        zip_buf.seek(0)
        st.download_button("Download all crops (ZIP)", data=zip_buf,
                           file_name="crops.zip", mime="application/zip")
else:
    st.caption("Tip: For tiny diagrams (atoms, etc.), increase “Merge distance (%)” and lower “Min area ratio”.")
