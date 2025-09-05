#!/usr/bin/env python3
# extractor_api_lazy.py — API that uses extract_core.py for cropping

import io
import os
import re
import ssl
import time
from functools import lru_cache
from typing import List, Dict, Any, Optional
import os
from fastapi import FastAPI, Body, Header, HTTPException

import cv2
import numpy as np
import requests
from PIL import Image
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel


import logging, sys, time, os, traceback, urllib.parse
from typing import Dict, Any, List
import numpy as np
import cv2
from PIL import Image

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from extract_core import configure_debug
configure_debug()  # ensures this module emits DEBUG logs

app = FastAPI()
API_TOKEN = os.environ.get("API_TOKEN", "change-me")

@app.get("/health")
def health():
    return {"ok": True}

def require_auth(auth: str | None):
    if not auth or not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    if auth.split(" ", 1)[1] != API_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token")

# Reuse the cropper module's logger if you like, or define a route-specific one:
ROUTE_LOGGER_NAME = "api.process_convertapi_json"
api_logger = logging.getLogger(ROUTE_LOGGER_NAME)
if not any(not isinstance(h, logging.NullHandler) for h in api_logger.handlers):
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.DEBUG)
    sh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s", "%H:%M:%S"))
    api_logger.addHandler(sh)
api_logger.setLevel(logging.DEBUG)

def _safe_url(u: str, keep_query=False, maxlen=120) -> str:
    """Mask URL to avoid leaking tokens; keep host + last path segment"""
    try:
        p = urllib.parse.urlsplit(u)
        host = p.netloc
        path = p.path or ""
        tail = path.rstrip("/").split("/")[-1]
        q = ("?" + p.query) if (keep_query and p.query) else ""
        s = f"https://{host}/.../{tail}{q}"
        return (s[:maxlen] + "…") if len(s) > maxlen else s
    except Exception:
        return "<unparseable-url>"

def _img_stats_arr(tag: str, arr: np.ndarray):
    if arr is None:
        api_logger.debug(f"[{tag}] arr=None")
        return
    h, w = arr.shape[:2]
    ch = 1 if arr.ndim == 2 else arr.shape[2]
    api_logger.debug(
        f"[{tag}] shape=({h}x{w}x{ch}) dtype={arr.dtype} "
        f"min={float(arr.min()) if arr.size else 'NA'} max={float(arr.max()) if arr.size else 'NA'} "
        f"mean={float(arr.mean()) if arr.size else 'NA'} nonzero={int(np.count_nonzero(arr))}"
    )

def _page_num_from_filename_safe(name: str) -> int:
    try:
        return _page_num_from_filename(name)
    except Exception:
        return -1

# 👉 All detection/cropping lives in extract_core.py
from extract_core import extract_crops_streamlit

# Google APIs (OAuth)
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
import gspread
from googleapiclient.http import MediaIoBaseUpload

# Robust downloader (retries + TLS fallback)
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import urllib.request

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"

_session = requests.Session()
_retry = Retry(
    total=5,
    backoff_factor=0.5,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"],
    respect_retry_after_header=True,
)
_adapter = HTTPAdapter(max_retries=_retry, pool_connections=20, pool_maxsize=50)
_session.mount("https://", _adapter)
_session.mount("http://", _adapter)

# ---------- Models ----------
class ConvertFile(BaseModel):
    FileName: str
    FileUrl: str

class ExtractionSettings(BaseModel):
    # Keep defaults identical to your Streamlit UI
    sat_thr: int = 28
    v_min: int = 80
    merge_pct: float = 2.0
    min_area_ratio: float = 0.004
    max_area_ratio: float = 0.90
    text_threshold: float = 0.08
    bottom_trim_pct: int = 8
    pad: int = 6

class ConvertApiPayload(BaseModel):
    Files: List[ConvertFile]
    SourcePdfName: Optional[str] = None
    settings: Optional[ExtractionSettings] = None

# ---------- OAuth clients (Drive + Sheets) ----------
SCOPES = [
    "https://www.googleapis.com/auth/drive.file",
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
            flow = InstalledAppFlow.from_client_secrets_file(client_file, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(token_file, "w", encoding="utf-8") as f:
            f.write(creds.to_json())
    return creds

@lru_cache(maxsize=1)
def get_services():
    folder_id = os.getenv("GDRIVE_FOLDER_ID")
    sheet_id  = os.getenv("GSHEET_ID")
    sheet_tab = os.getenv("SHEET_NAME", "ExtractedImages")
    if not (folder_id and sheet_id):
        raise RuntimeError("Set GDRIVE_FOLDER_ID and GSHEET_ID in your .env")

    creds = _get_oauth_creds()
    drive_service = build("drive", "v3", credentials=creds)
    gc = gspread.authorize(creds)
    sh = gc.open_by_key(sheet_id)
    try:
        ws = sh.worksheet(sheet_tab)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=sheet_tab, rows=2000, cols=10)
        ws.append_row(["pdf_name", "page_num", "image_num", "drive_file_url"])
    return folder_id, drive_service, ws

# ---------- Utils ----------
def _page_num_from_filename(name: str) -> int:
    m = re.findall(r"(\d+)", name or "")
    return int(m[-1]) if m else 1

def _bytes_from_url(url: str, timeout=120) -> bytes:
    """Robust downloader: retries + TLS1.3 fallback for EOF-in-protocol issues."""
    try:
        r = _session.get(url, timeout=timeout, headers={"User-Agent": UA}, allow_redirects=True)
        r.raise_for_status()
        return r.content
    except requests.exceptions.SSLError:
        ctx = ssl.create_default_context()
        if hasattr(ssl, "OP_NO_TLSv1_3"):
            try:
                ctx.options |= ssl.OP_NO_TLSv1_3
            except Exception:
                pass
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, context=ctx, timeout=timeout) as resp:
            return resp.read()

def _pil_to_png_bytes(pil_img: Image.Image) -> bytes:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()

def upload_png_to_drive(img_bytes: bytes, filename: str, folder_id_override: Optional[str] = None) -> str:
    folder_id, drive_service, _ = get_services()
    target_folder = folder_id_override or folder_id
    media = MediaIoBaseUpload(io.BytesIO(img_bytes), mimetype="image/png", resumable=False)
    meta  = {"name": filename, "parents": [target_folder]}
    created = drive_service.files().create(
        body=meta, media_body=media, fields="id, webViewLink, webContentLink"
    ).execute()
    # Optional: make public link (may be restricted by your account policy)
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

# ---------- FastAPI ----------
app = FastAPI(title="ConvertAPI → extract_core → Drive/Sheets")

@app.get("/healthz")
def health():
    return {
        "ok": True,
        "has_oauth_client": bool(os.getenv("OAUTH_CLIENT_FILE")),
        "has_oauth_token": bool(os.getenv("OAUTH_TOKEN_FILE") and os.path.exists(os.getenv("OAUTH_TOKEN_FILE"))),
        "has_folder_id": bool(os.getenv("GDRIVE_FOLDER_ID")),
        "has_sheet_id": bool(os.getenv("GSHEET_ID")),
    }

# --- instrumented route ---
@app.post("/process-convertapi-json")
def process_convertapi_json(payload: ConvertApiPayload = Body(...),authorization: str | None = Header(default=None)) -> Dict[str, Any]:
    
    run_id = f"run-{int(time.time())}"
    api_logger.debug(f"[BEGIN] {run_id} Received payload")

    files = payload.Files or []
    api_logger.debug(f"[PAYLOAD] files_count={len(files)} "
                     f"pdf_name={payload.SourcePdfName!r} "
                     f"settings_present={bool(payload.settings)}")

    if not files:
        api_logger.error(f"[ERROR] {run_id} No files found in payload.")
        raise HTTPException(status_code=400, detail="No files found in payload.")

    # Ensure Google services ready (OAuth flow triggers on first run)
    t0 = time.time()
    api_logger.debug(f"[GOOGLE] {run_id} Initializing services…")
    try:
        _ = get_services()
        api_logger.debug(f"[GOOGLE] {run_id} Services ready in {time.time()-t0:.2f}s")
    except Exception as e:
        api_logger.exception(f"[GOOGLE] {run_id} get_services failed: {e}")
        raise

    pdf_name = payload.SourcePdfName or "input.pdf"
    s = payload.settings or ExtractionSettings()

    # Echo effective settings
    api_logger.debug(
        f"[SETTINGS] {run_id} "
        f"sat_thr={s.sat_thr} v_min={s.v_min} merge_pct={s.merge_pct} "
        f"min_area_ratio={s.min_area_ratio} max_area_ratio={s.max_area_ratio} "
        f"text_threshold={s.text_threshold} bottom_trim_pct={s.bottom_trim_pct} pad={s.pad}"
    )

    rows: List[List[Any]] = []
    total_crops = 0
    errors: List[str] = []

    start = time.time()
    api_logger.debug(f"[LOOP] {run_id} Start processing {len(files)} pages")

    for i, f in enumerate(files, start=1):
        pstart = time.time()
        page_num_guess = _page_num_from_filename_safe(getattr(f, "FileName", ""))
        safe_url = _safe_url(getattr(f, "FileUrl", ""))
        api_logger.debug(
            f"[PAGE] {run_id} i={i} "
            f"fileName={getattr(f, 'FileName', '')!r} page_num_guess={page_num_guess} url={safe_url}"
        )
        try:
            # Download & decode page PNG
            t_dl = time.time()
            page_png = _bytes_from_url(f.FileUrl)
            dl_ms = (time.time() - t_dl) * 1000.0
            api_logger.debug(f"[DOWNLOAD] {run_id} i={i} bytes={len(page_png)} dl_ms={dl_ms:.1f}")

            img_arr = np.frombuffer(page_png, dtype=np.uint8)
            api_logger.debug(f"[DECODE] {run_id} i={i} np_frombuffer len={img_arr.size}")
            img_bgr = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
            if img_bgr is None:
                raise RuntimeError("Failed to decode page image")
            _img_stats_arr(f"PAGE[{i}].bgr", img_bgr)

            # 👉 call shared cropper (exact Streamlit behavior)
            api_logger.debug(
                f"[CROPPER] {run_id} i={i} calling extract_crops_streamlit "
                f"(sat_thr={s.sat_thr}, v_min={s.v_min}, merge_pct={s.merge_pct}, "
                f"min_area_ratio={s.min_area_ratio}, max_area_ratio={s.max_area_ratio}, "
                f"text_threshold={s.text_threshold}, bottom_trim_pct={s.bottom_trim_pct}, pad={s.pad})"
            )
            t_crop = time.time()
            crops = extract_crops_streamlit(
                img_bgr,
                28, 80, 2.0,
                0.004, 0.90,
                0.08, 8,
                6,
            )
            crop_ms = (time.time() - t_crop) * 1000.0
            api_logger.debug(f"[CROPPER] {run_id} i={i} crops_found={len(crops)} crop_ms={crop_ms:.1f}")

            page_num = _page_num_from_filename_safe(f.FileName)
            if not crops:
                rows.append([pdf_name, page_num, -1, "NO_CROPS"])
                api_logger.debug(f"[ROWS] {run_id} i={i} appended NO_CROPS row (page={page_num})")
            else:
                # Upload each crop, add row to sheet
                for ci, rgb in enumerate(crops, start=1):
                    # log crop stats without dumping pixel values
                    ch = 1 if rgb.ndim == 2 else rgb.shape[2]
                    api_logger.debug(
                        f"[CROP] {run_id} i={i} ci={ci} shape=({rgb.shape[0]}x{rgb.shape[1]}x{ch}) dtype={rgb.dtype}"
                    )
                    pil = Image.fromarray(rgb)
                    t_png = time.time()
                    crop_bytes = _pil_to_png_bytes(pil)
                    png_ms = (time.time() - t_png) * 1000.0

                    out_name = f"{os.path.splitext(pdf_name)[0]}_p{page_num:02d}_img{ci:02d}.png"
                    api_logger.debug(
                        f"[UPLOAD] {run_id} i={i} ci={ci} out_name={out_name!r} png_bytes={len(crop_bytes)} png_ms={png_ms:.1f}"
                    )

                    t_up = time.time()
                    link = upload_png_to_drive(crop_bytes, out_name, None)
                    up_ms = (time.time() - t_up) * 1000.0
                    api_logger.debug(f"[UPLOAD] {run_id} i={i} ci={ci} drive_link={_safe_url(link, keep_query=False)} up_ms={up_ms:.1f}")

                    rows.append([pdf_name, page_num, ci, link])
                    total_crops += 1

                # Optional micro-throttle
                if i % 10 == 0:
                    api_logger.debug(f"[THROTTLE] {run_id} i={i} sleeping 0.3s to ease API quotas")
                    time.sleep(0.3)

            api_logger.debug(f"[PAGE] {run_id} i={i} done in {(time.time()-pstart)*1000.0:.1f} ms")

        except Exception as e:
            err = f"Page {i} ({getattr(f, 'FileName', '')}): {e}"
            errors.append(err)
            rows.append([pdf_name, page_num_guess, -1, f"ERROR: {e}"])
            api_logger.error(f"[EXC] {run_id} i={i} {err}")
            api_logger.debug("[TRACE]\n" + "".join(traceback.format_exc()))

    # Append rows to sheet
    t_sheet = time.time()
    api_logger.debug(f"[SHEET] {run_id} appending {len(rows)} rows to sheet…")
    append_rows_to_sheet(rows)
    sheet_ms = (time.time() - t_sheet) * 1000.0
    took = round(time.time() - start, 2)

    api_logger.debug(
        f"[END] {run_id} pages_received={len(files)} crops_uploaded={total_crops} "
        f"rows_appended={len(rows)} seconds={took} sheet_ms={sheet_ms:.1f} "
        f"errors_count={len(errors)}"
    )

    return {
        "ok": True,
        "pages_received": len(files),
        "crops_uploaded": total_crops,
        "rows_appended": len(rows),
        "seconds": took,
        "errors": errors,
        "run_id": run_id,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("extractor_api_lazy:app", host="0.0.0.0", port=5000, reload=False)
