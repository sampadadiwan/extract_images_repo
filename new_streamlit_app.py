# app.py
"""
Streamlit UI only — all detection/cropping lives in extract_core.py.

Place this file next to extract_core.py, then run:
    streamlit run app.py

Required: extract_core.py must export:
    - extract_crops_streamlit(img_bgr, sat_thr, v_min, merge_pct, min_area_ratio,
                              max_area_ratio, text_threshold, bottom_trim_pct, pad)
    - build_color_mask(img_bgr, sat_thr, v_min)
    - merge_mask(mask, merge_pct)
"""

import io
import os
import zipfile
import cv2
import numpy as np
from PIL import Image

import streamlit as st

from extract_core import configure_debug
configure_debug()  # ensures this module emits DEBUG logs

# Import the shared cropper helpers from extract_core
try:
    from extract_core import (
        extract_crops_streamlit,
        build_color_mask,
        merge_mask,
    )
except ImportError as e:
    st.error(
        "Could not import `extract_core`. Make sure `extract_core.py` is in the same folder "
        "and defines `extract_crops_streamlit`, `build_color_mask`, and `merge_mask`."
    )
    st.stop()

# Optional: if Tesseract is not on PATH (Windows), you can set this env var
# os.environ["TESSERACT_CMD"] = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def run_ui():
    st.set_page_config(page_title="Page Image Extractor", layout="wide")
    st.title("Page Image Extractor (OpenCV + Merge + OCR)")

    uploaded = st.file_uploader("Upload a PNG/JPG page (300 dpi works best)", type=["png", "jpg", "jpeg"])

    with st.sidebar:
        st.header("Detection settings (forwarded to extract_core)")
        sat_thr = st.slider("Min saturation (color mask)", 5, 60, 28)
        v_min = st.slider("Min brightness (color mask)", 40, 160, 80)
        merge_pct = st.slider("Merge distance (% of min page dim)", 1.0, 6.0, 2.0, 0.1)
        min_area_ratio = st.number_input("Min area ratio", 0.0002, 0.2, 0.004, 0.0002, format="%.6f")
        max_area_ratio = st.number_input("Max area ratio", 0.20, 0.99, 0.90, 0.01, format="%.2f")
        text_threshold = st.slider("Text filter threshold", 0.00, 0.20, 0.08, 0.01)
        bottom_trim_pct = st.slider("Trim bottom of crop (%)", 0, 20, 8)
        pad = st.slider("Padding (px)", 0, 20, 6)
        show_masks = st.checkbox("Show masks (mask & merged)", value=False)

    if uploaded is None:
        st.caption("Tip: For tiny diagrams, increase Merge distance and lower Min area ratio.")
        return

    # Decode the uploaded image
    img_bytes = np.frombuffer(uploaded.getvalue(), dtype=np.uint8)
    img_bgr = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
    if img_bgr is None:
        st.error("Failed to decode the image. Please re-upload.")
        return

    st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption="Original page", use_column_width=True)

    # Call into extract_core for cropping
    with st.spinner("Extracting crops..."):
        crops = extract_crops_streamlit(
            img_bgr=img_bgr,
            sat_thr=sat_thr,
            v_min=v_min,
            merge_pct=merge_pct,
            min_area_ratio=min_area_ratio,
            max_area_ratio=max_area_ratio,
            text_threshold=text_threshold,
            bottom_trim_pct=bottom_trim_pct,
            pad=pad,
        )

    # Optional previews of the intermediate masks (also provided by extract_core)
    if show_masks:
        mask_preview = build_color_mask(img_bgr, sat_thr, v_min)
        merged_preview = merge_mask(mask_preview, merge_pct)
        st.image(mask_preview, caption="Color mask", use_column_width=True, clamp=True)
        st.image(merged_preview, caption="Merged mask (used for contours)", use_column_width=True, clamp=True)

    st.subheader(f"Extracted crops: {len(crops)}")
    if not crops:
        st.info("No matches. Try lowering Min saturation & Min area ratio, or increasing Merge distance.")
        return

    # Preview & offer ZIP download
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, rgb in enumerate(crops, 1):
            st.image(rgb, caption=f"Crop {i}", use_column_width=True)
            pil = Image.fromarray(rgb)
            buf = io.BytesIO()
            pil.save(buf, format="PNG")
            zf.writestr(f"crop_{i:02d}.png", buf.getvalue())
    zip_buf.seek(0)
    st.download_button(
        "Download all crops (ZIP)",
        data=zip_buf,
        file_name="crops.zip",
        mime="application/zip",
    )


if __name__ == "__main__":
    run_ui()
