"""
extractor_api.py

FastAPI service that accepts the full ConvertAPI JSON (with Files[] of PNG page URLs),
extracts illustrations/diagrams from each page image using OpenCV + optional OCR-based
text filtering, uploads the extracted crops to a Google Drive folder, and appends
rows to a Google Sheet: [pdf_name, page_num, image_num, drive_file_url].

Environment variables required:
- GOOGLE_APPLICATION_CREDENTIALS=/path/to/service_account.json
- GDRIVE_FOLDER_ID=<target_google_drive_folder_id>
- GSHEET_ID=<target_google_sheet_id>
- (optional) SHEET_NAME=ExtractedImages
- (optional) TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe  # Windows

Run:
    uvicorn extractor_api:app --host 0.0.0.0 --port 5000

Install requirements (example):
    pip install fastapi uvicorn requests pillow opencv-python numpy pytesseract \
                gspread google-api-python-client google-auth google-auth-oauthlib google-auth-httplib2
"""

import io
import os
import re
import time
import traceback
from typing import List, Dict, Any, Optional

import cv2
import numpy as np
import requests
from PIL import Image

import pytesseract
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel

# --- Google APIs ---
import gspread
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload

from dotenv import load_dotenv

load_dotenv()  # loads variables from .env in the current working directory



# ------------------------------
# Optional: Tesseract path (Windows)
# ------------------------------
if os.getenv("TESSERACT_CMD"):
    pytesseract.pytesseract.tesseract_cmd = os.getenv("TESSERACT_CMD")


# ------------------------------
# Config via environment variables
# ------------------------------
GDRIVE_FOLDER_ID = os.getenv("GDRIVE_FOLDER_ID")
GSHEET_ID        = os.getenv("GSHEET_ID")
SHEET_NAME       = os.getenv("SHEET_NAME", "ExtractedImages")
GOOGLE_SCOPES    = [
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/spreadsheets"
]
SERVICE_ACCOUNT_JSON = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

if not (GDRIVE_FOLDER_ID and GSHEET_ID and SERVICE_ACCOUNT_JSON):
    raise RuntimeError(
        "Set GDRIVE_FOLDER_ID, GSHEET_ID, and GOOGLE_APPLICATION_CREDENTIALS environment variables."
    )

creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_JSON, scopes=GOOGLE_SCOPES)
drive_service = build("drive", "v3", credentials=creds)
gc = gspread.authorize(creds)
sh = gc.open_by_key(GSHEET_ID)
try:
    ws = sh.worksheet(SHEET_NAME)
except gspread.WorksheetNotFound:
    ws = sh.add_worksheet(title=SHEET_NAME, rows=2000, cols=10)
    ws.append_row(["pdf_name", "page_num", "image_num", "drive_file_url"])


# ------------------------------
# Data models
# ------------------------------
class ConvertFile(BaseModel):
    FileName: str
    FileUrl: str

class ExtractionSettings(BaseModel):
    # Controls for detection; reasonable defaults included
    sat_thr: int = 28                # Min saturation for color mask
    v_min: int = 80                  # Min brightness for color mask
    merge_pct: float = 2.0           # Merge distance (% of min page dim)
    min_area_ratio: float = 0.004    # Reject crops smaller than this fraction of page area
    max_area_ratio: float = 0.90     # Reject crops larger than this fraction of page area
    text_threshold: float = 0.08     # Filter crops that look too text-heavy (0..1)
    bottom_trim_pct: int = 8         # Trim this % from bottom to avoid captions
    pad: int = 6                     # Pad bounding boxes (pixels)

class ConvertApiPayload(BaseModel):
    Files: List[ConvertFile]
    SourcePdfName: Optional[str] = None
    settings: Optional[ExtractionSettings] = None  # Optional overrides


# ------------------------------
# OpenCV-based extraction helpers
# ------------------------------
def text_ratio(img_bgr: np.ndarray) -> float:
    """
    Estimate how text-heavy a crop is (0..1) using a simple OCR-based heuristic.
    If OCR fails/unavailable, returns 0.0 so it doesn't filter out crops.
    """
    try:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        _, bw = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        data = pytesseract.image_to_data(bw, output_type=pytesseract.Output.DICT, config="--psm 6")
        if len(data.get("text", [])) == 0:
            return 0.0
        H, W = bw.shape[:2]
        page_area = H * W
        total_text_area, char_boxes = 0, 0
        for i in range(len(data["text"])):
            conf_str = data["conf"][i]
            conf = int(conf_str) if conf_str not in ("-1", "", None) else -1
            txt = (data["text"][i] or "").strip()
            if conf >= 60 and any(c.isalnum() for c in txt):
                total_text_area += data["width"][i] * data["height"][i]
                char_boxes += 1
        coverage = total_text_area / (page_area + 1e-6)
        return float(0.7 * coverage + 0.3 * min(char_boxes / 50.0, 1.0))
    except Exception:
        return 0.0  # graceful fallback if OCR not present


def build_color_mask(img_bgr: np.ndarray, sat_thr=28, v_min=80) -> np.ndarray:
    """Create a mask of colored pixels to keep illustrations/photos and drop grey text."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    S, V = hsv[..., 1], hsv[..., 2]
    mask = ((S > sat_thr) & (V > v_min)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8), iterations=1)
    return mask


def merge_mask(mask: np.ndarray, merge_pct=2.0) -> np.ndarray:
    """
    Dilate/close to fuse nearby blobs into one region so grouped illustrations become one crop.
    merge_pct is a percentage of the minimum page dimension.
    """
    H, W = mask.shape[:2]
    k = max(9, int(min(H, W) * (merge_pct / 100.0)))
    if k % 2 == 0:
        k += 1
    merge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    merged = cv2.dilate(mask, merge_kernel, iterations=1)
    merged = cv2.morphologyEx(merged, cv2.MORPH_CLOSE, np.ones((max(3, k//3), max(3, k//3)), np.uint8), iterations=1)
    return merged


def extract_crops(
    img_bgr: np.ndarray,
    sat_thr: int,
    v_min: int,
    merge_pct: float,
    min_area_ratio: float,
    max_area_ratio: float,
    text_threshold: float,
    bottom_trim_pct: int,
    pad: int = 6,
) -> List[np.ndarray]:
    """
    Returns a list of RGB crops (np.ndarray) detected as non-text illustrations.
    """
    H, W = img_bgr.shape[:2]
    page_area = H * W

    mask = build_color_mask(img_bgr, sat_thr, v_min)
    merged = merge_mask(mask, merge_pct)

    cnts, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    crops: List[np.ndarray] = []
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

        # Trim a portion from bottom to reduce caption bleed
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
# Utility helpers (Drive + Sheets)
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

def upload_png_to_drive(img_bytes: bytes, filename: str, folder_id: str) -> str:
    media = MediaIoBaseUpload(io.BytesIO(img_bytes), mimetype="image/png", resumable=False)
    file_metadata = {"name": filename, "parents": [folder_id]}
    created = drive_service.files().create(
        body=file_metadata, media_body=media, fields="id, webViewLink, webContentLink"
    ).execute()
    # make it public (optional)
    try:
        drive_service.permissions().create(
            fileId=created["id"], body={"role": "reader", "type": "anyone"}, fields="id"
        ).execute()
    except Exception:
        pass
    return created.get("webViewLink") or created.get("webContentLink")

def append_rows_to_sheet(rows: List[List[Any]]):
    if rows:
        ws.append_rows(rows, value_input_option="RAW")


# ------------------------------
# FastAPI app
# ------------------------------
app = FastAPI(title="ConvertAPI → OpenCV Extractor → Drive/Sheets")

@app.post("/process-convertapi-json")
def process_convertapi_json(payload: ConvertApiPayload = Body(...)) -> Dict[str, Any]:
    """
    Accepts ConvertAPI JSON:
    {
      "Files": [{"FileName": "...", "FileUrl": "..."}, ...],
      "SourcePdfName": "original.pdf",
      "settings": { ... overrides ... }
    }
    """
    files = payload.Files or []
    if not files:
        raise HTTPException(status_code=400, detail="No files found in payload.")

    pdf_name = payload.SourcePdfName or "input.pdf"

    # Merge default settings with any overrides provided
    s = payload.settings or ExtractionSettings()
    rows: List[List[Any]] = []
    total_crops = 0
    errors: List[str] = []

    start = time.time()
    for idx, f in enumerate(files, start=1):
        try:
            page_png = _bytes_from_url(f.FileUrl)
            img_arr = np.frombuffer(page_png, dtype=np.uint8)
            img_bgr = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
            if img_bgr is None:
                raise RuntimeError("Failed to decode page image")

            crops = extract_crops(
                img_bgr=img_bgr,
                sat_thr=s.sat_thr,
                v_min=s.v_min,
                merge_pct=s.merge_pct,
                min_area_ratio=s.min_area_ratio,
                max_area_ratio=s.max_area_ratio,
                text_threshold=s.text_threshold,
                bottom_trim_pct=s.bottom_trim_pct,
                pad=s.pad,
            )

            page_num = _page_num_from_filename(f.FileName)
            if not crops:
                # still record a row with -1 to signal no crops
                rows.append([pdf_name, page_num, -1, "NO_CROPS"])
                continue

            for ci, rgb in enumerate(crops, start=1):
                pil = Image.fromarray(rgb)
                crop_bytes = _pil_to_png_bytes(pil)
                drive_name = f"{os.path.splitext(pdf_name)[0]}_p{page_num:02d}_img{ci:02d}.png"
                link = upload_png_to_drive(crop_bytes, drive_name, GDRIVE_FOLDER_ID)
                rows.append([pdf_name, page_num, ci, link])
                total_crops += 1

            # light pacing every 10 pages
            if idx % 10 == 0:
                time.sleep(0.3)

        except Exception as e:
            errors.append(f"Page {idx} ({f.FileName}): {e}")
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
    uvicorn.run("extractor_api:app", host="0.0.0.0", port=5000, reload=False)
