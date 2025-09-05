# extractor_api_lazy.py
# FastAPI service with Google OAuth (user consent), image crop extraction, Drive upload, Sheets append.
# Works on Render/Railway. Pair with your extract_core.py (must be in the same directory).
#
# Required env vars (Render/Railway -> Environment):
#   GOOGLE_CLIENT_ID            (from Google Cloud Console - OAuth client, Web)
#   GOOGLE_CLIENT_SECRET        (from Google Cloud Console)
#   GOOGLE_REDIRECT_URI         (e.g., https://<service>.onrender.com/oauth2/callback)
#   API_TOKEN                   (any long random string; used as Bearer auth for /process-convertapi-json)
#   GOOGLE_REFRESH_TOKEN        (set AFTER first OAuth; see /google/auth + /oauth2/callback)
# Optional:
#   SHEET_ID                    (Google Sheets spreadsheet ID to append rows)
#   SHEET_RANGE                 (e.g. "Sheet1!A:D"; default "Sheet1!A:D")
#   DRIVE_FOLDER_ID             (target Drive folder ID for uploads; default: root)
#   MAKE_PUBLIC                 ("true"/"1" to add anyoneWithLink reader permission)
#   LOG_LEVEL                   (DEBUG/INFO/WARN; default DEBUG)
#
# Run locally:
#   uvicorn extractor_api_lazy:app --host 0.0.0.0 --port 5000 --reload
#
# Docker CMD (example):
#   uvicorn extractor_api_lazy:app --host 0.0.0.0 --port ${PORT:-8000} --workers 2

import os
import re
import io
import sys
import json
import time
import base64
import logging
import urllib.request
from typing import Any, Dict, List, Optional

import numpy as np
import cv2
from PIL import Image

from fastapi import FastAPI, Body, Header, HTTPException, Request
from fastapi.responses import RedirectResponse, PlainTextResponse

from pydantic import BaseModel, Field

from google_auth_oauthlib.flow import Flow
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request as GoogleRequest
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload

# --- import your cropper (instrumented) ---
from extract_core import extract_crops_streamlit, configure_debug as core_configure_debug

# If you're using a non-HTTPS redirect for local dev, force OAuthlib to allow it.
if os.getenv("GOOGLE_REDIRECT_URI", "").startswith(("http://localhost", "http://127.0.0.1")):
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"  # allow http
    # optional, but handy: avoid scope strictness during dev
    os.environ.setdefault("OAUTHLIB_RELAX_TOKEN_SCOPE", "1")

# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------
LOG_LEVEL = os.environ.get("LOG_LEVEL", "DEBUG").upper()
api_logger = logging.getLogger("api.extractor")
if not any(not isinstance(h, logging.NullHandler) for h in api_logger.handlers):
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.DEBUG)
    sh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s", "%H:%M:%S"))
    api_logger.addHandler(sh)
api_logger.setLevel(LOG_LEVEL)

# Also enable extract_core module debug once here
try:
    core_configure_debug(logging.getLevelName(LOG_LEVEL))
except Exception:
    pass

# ------------------------------------------------------------------------------
# FastAPI app
# ------------------------------------------------------------------------------
app = FastAPI(title="OCR Crop Extractor API (OAuth)")

# ------------------------------------------------------------------------------
# OAuth (user-consent) configuration
# ------------------------------------------------------------------------------
SCOPES = [
    "https://www.googleapis.com/auth/drive.file",
    "https://www.googleapis.com/auth/spreadsheets",
]
GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET")
GOOGLE_REDIRECT_URI = os.environ.get("GOOGLE_REDIRECT_URI")

def _client_cfg() -> Dict[str, Any]:
    if not (GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET and GOOGLE_REDIRECT_URI):
        raise RuntimeError("Missing Google OAuth env vars (GOOGLE_CLIENT_ID / GOOGLE_CLIENT_SECRET / GOOGLE_REDIRECT_URI)")
    return {
        "web": {
            "client_id": GOOGLE_CLIENT_ID,
            "client_secret": GOOGLE_CLIENT_SECRET,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": [GOOGLE_REDIRECT_URI],
        }
    }

def _get_refresh_token_from_env() -> Optional[str]:
    rt = os.environ.get("GOOGLE_REFRESH_TOKEN")
    return rt or None

def _require_refresh_token() -> str:
    rt = _get_refresh_token_from_env()
    if not rt:
        raise RuntimeError("No Google refresh token set. Visit /google/auth once, then set GOOGLE_REFRESH_TOKEN and redeploy.")
    return rt

def _make_credentials_from_refresh() -> Credentials:
    rt = _require_refresh_token()
    creds = Credentials(
        token=None,
        refresh_token=rt,
        client_id=GOOGLE_CLIENT_ID,
        client_secret=GOOGLE_CLIENT_SECRET,
        token_uri="https://oauth2.googleapis.com/token",
        scopes=SCOPES,
    )
    creds.refresh(GoogleRequest())
    return creds

def get_services():
    """Create Drive/Sheets clients using an OAuth refresh token (from env)."""
    creds = _make_credentials_from_refresh()
    drive = build("drive", "v3", credentials=creds, cache_discovery=False)
    sheets = build("sheets", "v4", credentials=creds, cache_discovery=False)
    return drive, sheets

# ------------------------------------------------------------------------------
# Security (Bearer token) for your processing endpoint
# ------------------------------------------------------------------------------
API_TOKEN = os.environ.get("API_TOKEN", "change-me")

def _require_auth(authorization: Optional[str]):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = authorization.split(" ", 1)[1]
    if token != API_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token")

# ------------------------------------------------------------------------------
# Models
# ------------------------------------------------------------------------------
class FileItem(BaseModel):
    FileName: str
    FileUrl: str

class ExtractionSettings(BaseModel):
    sat_thr: int = 28
    v_min: int = 80
    merge_pct: float = 2.0
    min_area_ratio: float = 0.004
    max_area_ratio: float = 0.90
    text_threshold: float = 0.08
    bottom_trim_pct: int = 8
    pad: int = 6

class ConvertApiPayload(BaseModel):
    Files: List[FileItem] = Field(default_factory=list)
    SourcePdfName: Optional[str] = "input.pdf"
    settings: Optional[ExtractionSettings] = None

# ------------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------------
def _safe_url(u: str, keep_query: bool = False, maxlen: int = 140) -> str:
    try:
        # Keep host + tail segment; hide tokens
        from urllib.parse import urlsplit
        p = urlsplit(u)
        host = p.netloc
        path = p.path or ""
        tail = path.rstrip("/").split("/")[-1]
        q = ("?" + p.query) if (keep_query and p.query) else ""
        s = f"https://{host}/.../{tail}{q}"
        return (s[:maxlen] + "…") if len(s) > maxlen else s
    except Exception:
        return "<url>"

def _bytes_from_url(url: str, timeout: int = 60) -> bytes:
    t0 = time.time()
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
    api_logger.debug(f"[HTTP] GET { _safe_url(url) } -> {len(data)} bytes in {(time.time()-t0)*1000.0:.1f} ms")
    return data

def _pil_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

_page_re = re.compile(r"(?:p|page)?\s*(\d+)", re.IGNORECASE)
def _page_num_from_filename(name: str) -> int:
    # try to find a number in the filename; fallback -1
    if not name:
        return -1
    # common patterns: p01.png, page_12.png, file-1-70-69.png (take last number)
    nums = re.findall(r"(\d+)", name)
    if nums:
        try:
            return int(nums[-1])
        except Exception:
            return int(nums[0])
    return -1

def _img_stats_arr(tag: str, arr: np.ndarray):
    try:
        h, w = arr.shape[:2]
        ch = 1 if arr.ndim == 2 else arr.shape[2]
        api_logger.debug(
            f"[{tag}] shape=({h}x{w}x{ch}) dtype={arr.dtype} "
            f"min={float(arr.min()) if arr.size else 'NA'} "
            f"max={float(arr.max()) if arr.size else 'NA'} "
            f"mean={float(arr.mean()) if arr.size else 'NA'} "
            f"nonzero={int(np.count_nonzero(arr))}"
        )
    except Exception:
        api_logger.debug(f"[{tag}] <stats unavailable>")

# ------------------------------------------------------------------------------
# Google helpers (Drive upload + Sheets append)
# ------------------------------------------------------------------------------
SHEET_ID = os.environ.get("SHEET_ID", "").strip()
SHEET_RANGE = os.environ.get("SHEET_RANGE", "Sheet1!A:D")
DRIVE_FOLDER_ID = os.environ.get("DRIVE_FOLDER_ID", "").strip()
MAKE_PUBLIC = os.environ.get("MAKE_PUBLIC", "").lower() in ("1", "true", "yes")

def upload_png_to_drive(png_bytes: bytes, filename: str, folder_id: Optional[str]) -> str:
    drive, _ = get_services()
    media = MediaIoBaseUpload(io.BytesIO(png_bytes), mimetype="image/png", resumable=False)
    metadata: Dict[str, Any] = {"name": filename, "mimeType": "image/png"}
    fid = folder_id or DRIVE_FOLDER_ID
    if fid:
        metadata["parents"] = [fid]
    file = drive.files().create(body=metadata, media_body=media, fields="id,webViewLink,webContentLink").execute()
    file_id = file.get("id")
    link = file.get("webViewLink") or file.get("webContentLink") or f"https://drive.google.com/file/d/{file_id}/view"

    if MAKE_PUBLIC:
        try:
            drive.permissions().create(
                fileId=file_id,
                body={"role": "reader", "type": "anyone"},
            ).execute()
        except Exception as e:
            api_logger.warning(f"[DRIVE] Could not make public: {e}")

    api_logger.debug(f"[DRIVE] uploaded {filename} -> {link}")
    return link

def append_rows_to_sheet(rows: List[List[Any]]):
    if not SHEET_ID:
        api_logger.warning("[SHEETS] SHEET_ID not set; skipping append.")
        return
    _, sheets = get_services()
    body = {"values": rows}
    res = sheets.spreadsheets().values().append(
        spreadsheetId=SHEET_ID,
        range=SHEET_RANGE,
        valueInputOption="RAW",
        insertDataOption="INSERT_ROWS",
        body=body,
    ).execute()
    api_logger.debug(f"[SHEETS] append rows={len(rows)} -> updatedRange={res.get('updates', {}).get('updatedRange')}")

# ------------------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------------------
@app.get("/health")
def health():
    ok = True
    has_rt = bool(_get_refresh_token_from_env())
    return {"ok": ok, "has_refresh_token": has_rt, "sheet": bool(SHEET_ID)}

@app.get("/google/auth")
def google_auth():
    """Kick off OAuth flow (one-time). After consent, /oauth2/callback will print refresh token to logs."""
    flow = Flow.from_client_config(_client_cfg(), scopes=SCOPES, redirect_uri=GOOGLE_REDIRECT_URI)
    url, _state = flow.authorization_url(
        access_type="offline", include_granted_scopes="true", prompt="consent"
    )
    return RedirectResponse(url)

@app.get("/oauth2/callback")
def oauth2_callback(request: Request):
    """Handle Google's redirect, exchange code for tokens, and print refresh_token once."""
    flow = Flow.from_client_config(_client_cfg(), scopes=SCOPES, redirect_uri=GOOGLE_REDIRECT_URI)
    flow.fetch_token(authorization_response=str(request.url))
    creds = flow.credentials
    if not creds.refresh_token:
        raise HTTPException(
            status_code=400,
            detail="No refresh_token returned. Revoke app at https://myaccount.google.com/permissions and retry.",
        )
    # Print once; copy into service env as GOOGLE_REFRESH_TOKEN, then redeploy.
    print("=== COPY & SAVE AS GOOGLE_REFRESH_TOKEN ===")
    print(creds.refresh_token)
    print("==========================================")
    return PlainTextResponse("Google connected. Copy refresh token from logs, set GOOGLE_REFRESH_TOKEN, then redeploy.")

# ------------------------------------------------------------------------------
# Main processing route (called from n8n)
# ------------------------------------------------------------------------------
@app.post("/process-convertapi-json")
def process_convertapi_json(
    payload: ConvertApiPayload = Body(...),
    authorization: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    _require_auth(authorization)

    run_id = f"run-{int(time.time())}"
    files = payload.Files or []
    api_logger.debug(f"[BEGIN] {run_id} files_count={len(files)} pdf_name={payload.SourcePdfName!r} settings_present={bool(payload.settings)}")

    if not files:
        raise HTTPException(status_code=400, detail="No files found in payload.")

    # Ensure Google services ready (OAuth refresh works)
    t0 = time.time()
    try:
        _ = get_services()
        api_logger.debug(f"[GOOGLE] {run_id} services ready in {time.time()-t0:.2f}s")
    except Exception as e:
        api_logger.exception(f"[GOOGLE] {run_id} get_services failed: {e}")
        raise HTTPException(status_code=500, detail=f"Google auth error: {e}")

    pdf_name = payload.SourcePdfName or "input.pdf"
    s = ExtractionSettings()  # defaults

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
        try:
            page_num_guess = _page_num_from_filename(getattr(f, "FileName", ""))
            safe_url = _safe_url(getattr(f, "FileUrl", ""))

            api_logger.debug(f"[PAGE] {run_id} i={i} fileName={f.FileName!r} page_num_guess={page_num_guess} url={safe_url}")

            # Download & decode page PNG
            page_png = _bytes_from_url(f.FileUrl)
            img_arr = np.frombuffer(page_png, dtype=np.uint8)
            img_bgr = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
            if img_bgr is None:
                raise RuntimeError("Failed to decode page image")
            _img_stats_arr(f"PAGE[{i}].bgr", img_bgr)

            # Call shared cropper (named args: bullet-proof)
            api_logger.debug(
                f"[CROPPER] {run_id} i={i} calling extract_crops_streamlit "
                f"(sat_thr={s.sat_thr}, v_min={s.v_min}, merge_pct={s.merge_pct}, "
                f"min_area_ratio={s.min_area_ratio}, max_area_ratio={s.max_area_ratio}, "
                f"text_threshold={s.text_threshold}, bottom_trim_pct={s.bottom_trim_pct}, pad={s.pad})"
            )
            t_crop = time.time()
            crops = extract_crops_streamlit(
                img_bgr,
                sat_thr=s.sat_thr,
                v_min=s.v_min,
                merge_pct=s.merge_pct,
                min_area_ratio=s.min_area_ratio,
                max_area_ratio=s.max_area_ratio,
                text_threshold=s.text_threshold,
                bottom_trim_pct=s.bottom_trim_pct,
                pad=s.pad,
            )
            crop_ms = (time.time() - t_crop) * 1000.0
            api_logger.debug(f"[CROPPER] {run_id} i={i} crops_found={len(crops)} crop_ms={crop_ms:.1f}")

            page_num = _page_num_from_filename(f.FileName)
            if not crops:
                rows.append([pdf_name, page_num, -1, "NO_CROPS"])
                api_logger.debug(f"[ROWS] {run_id} i={i} appended NO_CROPS row (page={page_num})")
            else:
                for ci, rgb in enumerate(crops, start=1):
                    ch = 1 if rgb.ndim == 2 else rgb.shape[2]
                    api_logger.debug(f"[CROP] {run_id} i={i} ci={ci} shape=({rgb.shape[0]}x{rgb.shape[1]}x{ch}) dtype={rgb.dtype}")
                    pil = Image.fromarray(rgb)
                    crop_bytes = _pil_to_png_bytes(pil)

                    out_name = f"{os.path.splitext(pdf_name)[0]}_p{page_num:02d}_img{ci:02d}.png"
                    t_up = time.time()
                    link = upload_png_to_drive(crop_bytes, out_name, None)
                    up_ms = (time.time() - t_up) * 1000.0
                    api_logger.debug(f"[UPLOAD] {run_id} i={i} ci={ci} link={_safe_url(link)} up_ms={up_ms:.1f}")

                    rows.append([pdf_name, page_num, ci, link])
                    total_crops += 1

                # Optional small throttle for big batches
                if i % 10 == 0:
                    api_logger.debug(f"[THROTTLE] {run_id} i={i} sleeping 0.3s")
                    time.sleep(0.3)

            api_logger.debug(f"[PAGE] {run_id} i={i} done in {(time.time()-pstart)*1000.0:.1f} ms")

        except Exception as e:
            err = f"Page {i} ({getattr(f, 'FileName', '')}): {e}"
            errors.append(err)
            rows.append([pdf_name, _page_num_from_filename(getattr(f, 'FileName', '')), -1, f"ERROR: {e}"])
            api_logger.error(f"[EXC] {run_id} i={i} {err}")

    # Append rows to sheet (if configured)
    t_sheet = time.time()
    append_rows_to_sheet(rows)
    sheet_ms = (time.time() - t_sheet) * 1000.0

    took = round(time.time() - start, 2)
    api_logger.debug(
        f"[END] {run_id} pages_received={len(files)} crops_uploaded={total_crops} "
        f"rows_appended={len(rows)} seconds={took} sheet_ms={sheet_ms:.1f} errors_count={len(errors)}"
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
