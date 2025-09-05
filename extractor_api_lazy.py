#!/usr/bin/env python3
"""
extractor_api_lazy.py
FastAPI service that:
- accepts the full ConvertAPI JSON (Files[] with PNG page URLs),
- downloads each page image,
- extracts sub-images via OpenCV (tunable),
- uploads crops to a Google Drive folder,
- appends rows to a Google Sheet: [pdf_name, page_num, image_num, drive_file_url].

This version defers environment lookup and Google client creation until runtime
so that `uvicorn` can import the module even if env is not loaded yet.

Run:
  uvicorn extractor_api_lazy:app --host 0.0.0.0 --port 5000 --env-file .env
"""

import io
import os
import re
import time
from functools import lru_cache
from typing import List, Dict, Any, Optional

import cv2
import numpy as np
import requests
from PIL import Image

from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
import gspread
from googleapiclient.http import MediaIoBaseUpload


# Load .env (safe even if file is missing)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Optional OCR (used only for text filtering)
try:
    import pytesseract
    if os.getenv("TESSERACT_CMD"):
        pytesseract.pytesseract.tesseract_cmd = os.getenv("TESSERACT_CMD")
except Exception:
    pytesseract = None

# ------------------------------
# Models
# ------------------------------
class ConvertFile(BaseModel):
    FileName: str
    FileUrl: str

# --- replace your ExtractionSettings with this ---
class ExtractionSettings(BaseModel):
    # Detection settings identical to the Streamlit app
    sat_thr: int = 28                 # Min saturation (color mask)
    v_min: int = 80                   # Min brightness (color mask)
    merge_pct: float = 2.0            # Merge distance (% of min page dim)
    min_area_ratio: float = 0.004     # Min area as fraction of page
    max_area_ratio: float = 0.90      # Max area as fraction of page
    text_threshold: float = 0.08      # OCR-based text filter threshold
    bottom_trim_pct: int = 8          # Trim from bottom of crop (%)
    pad: int = 6                      # Padding around contour (px)


class ConvertApiPayload(BaseModel):
    Files: List[ConvertFile]
    SourcePdfName: Optional[str] = None
    settings: Optional[ExtractionSettings] = None


SCOPES = [
    "https://www.googleapis.com/auth/drive.file",   # create/manage files you create
    "https://www.googleapis.com/auth/spreadsheets",
]

def _get_oauth_creds():
    client_file = os.getenv("OAUTH_CLIENT_FILE", "credentials.json")
    token_file  = os.getenv("OAUTH_TOKEN_FILE",  "token.json")
    creds = None
    if os.path.exists(token_file):
        creds = Credentials.from_authorized_user_file(token_file, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            # opens a browser once, then caches token.json
            flow = InstalledAppFlow.from_client_secrets_file(client_file, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(token_file, "w", encoding="utf-8") as f:
            f.write(creds.to_json())
    return creds


# ------------------------------
# Lazy Google services
# ------------------------------
@lru_cache(maxsize=1)
def get_services():
    """OAuth-based clients; files go to YOUR Drive & YOUR Sheet."""
    gdrive_folder_id = os.getenv("GDRIVE_FOLDER_ID")
    gsheet_id        = os.getenv("GSHEET_ID")
    sheet_name       = os.getenv("SHEET_NAME", "ExtractedImages")

    if not (gdrive_folder_id and gsheet_id):
        raise RuntimeError("Set GDRIVE_FOLDER_ID and GSHEET_ID in your .env")

    creds = _get_oauth_creds()
    drive_service = build("drive", "v3", credentials=creds)
    gc = gspread.authorize(creds)
    sh = gc.open_by_key(gsheet_id)
    try:
        ws = sh.worksheet(sheet_name)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=sheet_name, rows=2000, cols=10)
        ws.append_row(["pdf_name", "page_num", "image_num", "drive_file_url"])
    return gdrive_folder_id, drive_service, ws


# ------------------------------
# Extraction helpers (OpenCV)
# ------------------------------
def _page_num_from_filename(name: str) -> int:
    m = re.findall(r"(\d+)", name or "")
    return int(m[-1]) if m else 1

def _bytes_from_url(url: str, timeout=120) -> bytes:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.content

def _pil_to_png_bytes(pil_img: Image.Image) -> bytes:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()

# --- replace your text_ratio with this ---
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


# --- replace your build_color_mask with this ---
def build_color_mask(img_bgr, sat_thr=28, v_min=80):
    """Mask colored pixels: keeps illustrations/photos, drops grey watermark/text."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    S, V = hsv[..., 1], hsv[..., 2]
    mask = ((S > sat_thr) & (V > v_min)).astype(np.uint8) * 255
    # Gentle open so tiny bits survive; close to tidy edges
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8), iterations=1)
    return mask

# --- replace your merge_mask with this ---
def merge_mask(mask, merge_pct=2.0):
    """Dilate to fuse nearby blobs into one region (e.g., many small element photos)."""
    H, W = mask.shape[:2]
    k = max(9, int(min(H, W) * (merge_pct / 100.0)))
    if k % 2 == 0:
        k += 1  # kernel must be odd
    merge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    merged = cv2.dilate(mask, merge_kernel, iterations=1)
    merged = cv2.morphologyEx(merged, cv2.MORPH_CLOSE, np.ones((max(3, k//3), max(3, k//3)), np.uint8), iterations=1)
    return merged


# --- replace your merge_mask with this ---
def merge_mask(mask, merge_pct=2.0):
    """Dilate to fuse nearby blobs into one region (e.g., many small element photos)."""
    H, W = mask.shape[:2]
    k = max(9, int(min(H, W) * (merge_pct / 100.0)))
    if k % 2 == 0:
        k += 1  # kernel must be odd
    merge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    merged = cv2.dilate(mask, merge_kernel, iterations=1)
    merged = cv2.morphologyEx(merged, cv2.MORPH_CLOSE, np.ones((max(3, k//3), max(3, k//3)), np.uint8), iterations=1)
    return merged

# --- replace your extract_crops with this (note: returns only crops) ---
def extract_crops(img_bgr,
                  sat_thr: int,
                  v_min: int,
                  merge_pct: float,
                  min_area_ratio: float,
                  max_area_ratio: float,
                  text_threshold: float,
                  bottom_trim_pct: int,
                  pad: int = 6) -> list[np.ndarray]:
    """Return list of RGB crops using the Streamlit app's pipeline."""
    H, W = img_bgr.shape[:2]
    page_area = H * W

    mask = build_color_mask(img_bgr, sat_thr, v_min)
    merged = merge_mask(mask, merge_pct)

    cnts, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    crops: list[np.ndarray] = []
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
    return crops



# ------------------------------
# Google helpers (use lazy services)
# ------------------------------
def upload_png_to_drive(img_bytes: bytes, filename: str, folder_id_override: str | None = None) -> str:
    from googleapiclient.http import MediaIoBaseUpload
    gdrive_folder_id, drive_service, _ = get_services()
    folder_id = folder_id_override or gdrive_folder_id

    media = MediaIoBaseUpload(io.BytesIO(img_bytes), mimetype="image/png", resumable=False)
    meta  = {"name": filename, "parents": [folder_id]}
    created = drive_service.files().create(
        body=meta, media_body=media, fields="id, webViewLink, webContentLink"
    ).execute()

    # Optional: make public link (depends on your account settings)
    try:
        drive_service.permissions().create(
            fileId=created["id"], body={"role": "reader", "type": "anyone"}, fields="id"
        ).execute()
    except Exception:
        pass

    return created.get("webViewLink") or created.get("webContentLink")


def append_rows_to_sheet(rows: List[List[Any]]):
    if not rows:
        return
    _, _, ws = get_services()
    ws.append_rows(rows, value_input_option="RAW")


# ------------------------------
# FastAPI app
# ------------------------------
app = FastAPI(title="ConvertAPI → OpenCV → Drive/Sheets (lazy)")

@app.get("/healthz")
def health():
    """Basic health check + env presence."""
    return {
        "ok": True,
        "has_google_json": bool(os.getenv("GOOGLE_APPLICATION_CREDENTIALS")),
        "has_folder_id": bool(os.getenv("GDRIVE_FOLDER_ID")),
        "has_sheet_id": bool(os.getenv("GSHEET_ID")),
    }

@app.post("/process-convertapi-json")
def process_convertapi_json(payload: ConvertApiPayload = Body(...)) -> Dict[str, Any]:
    files = payload.Files or []
    if not files:
        raise HTTPException(status_code=400, detail="No files found in payload.")

    # Ensure Google services are ready (raises clear error if env missing)
    _ = get_services()

    pdf_name = payload.SourcePdfName or "input.pdf"
    s = payload.settings or ExtractionSettings()
    rows: List[List[Any]] = []
    total_crops = 0
    errors: List[str] = []

    start = time.time()
    for i, f in enumerate(files, start=1):
        try:
            page_png = _bytes_from_url(f.FileUrl)
            img_arr = np.frombuffer(page_png, dtype=np.uint8)
            img_bgr = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
            if img_bgr is None:
                raise RuntimeError("Failed to decode page image")

            crops = extract_crops(
                img_bgr, s.sat_thr, s.v_min, s.merge_pct,
                s.min_area_ratio, s.max_area_ratio, s.text_threshold, s.bottom_trim_pct, s.pad
            )

            page_num = _page_num_from_filename(f.FileName)
            if not crops:
                rows.append([pdf_name, page_num, -1, "NO_CROPS"])
                continue

            for ci, rgb in enumerate(crops, start=1):
                pil = Image.fromarray(rgb)
                crop_bytes = _pil_to_png_bytes(pil)
                out_name = f"{os.path.splitext(pdf_name)[0]}_p{page_num:02d}_img{ci:02d}.png"
                link = upload_png_to_drive(crop_bytes, out_name, None)
                rows.append([pdf_name, page_num, ci, link])
                total_crops += 1

            if i % 10 == 0:
                time.sleep(0.3)

        except Exception as e:
            errors.append(f"Page {i} ({f.FileName}): {e}")
            rows.append([pdf_name, _page_num_from_filename(f.FileName), -1, f"ERROR: {e}"])

    append_rows_to_sheet(rows)
    took = round(time.time() - start, 2)
    return {
        "ok": True,
        "pages_received": len(files),
        "crops_uploaded": total_crops,
        "rows_appended": len(rows),
        "seconds": took,
        "errors": errors,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("extractor_api_lazy:app", host="0.0.0.0", port=5000, reload=False)
