# extractor_api_lazyprod_ready.py
# FastAPI service that:
# 1) Receives ConvertAPI-style payload (page images).
# 2) Extracts image crops using extract_core.extract_crops_streamlit.
# 3) Uploads each crop to Google Drive (optionally makes them public).
# 4) Appends rows to Google Sheet (file_name, page_number, image_number, image_url).
# 5) Returns **no content (204)** by default when server writes the Sheet, so n8n
#    does not need to parse JSON. This can be toggled via env vars below.
#
# ── Required ENV ──────────────────────────────────────────────────────────────
#   GOOGLE_CLIENT_ID
#   GOOGLE_CLIENT_SECRET
#   GOOGLE_REDIRECT_URI                # e.g. https://<host>/oauth2/callback
#   GOOGLE_REFRESH_TOKEN               # set after completing /google/auth
#   API_TOKEN                          # Bearer token for /process-convertapi-json
#
# ── Optional ENV ─────────────────────────────────────────────────────────────
#   SHEET_ID                           # Google Sheet ID
#   SHEET_RANGE                        # default "Sheet1!A:D"
#   DRIVE_FOLDER_ID                    # target Drive folder
#   MAKE_PUBLIC                        # "true" to share uploaded images to anyone with link
#   LOG_LEVEL                          # default DEBUG
#   WRITE_TO_SHEETS_FROM_API           # default TRUE (this build writes rows itself)
#   NO_CONTENT_RESPONSE                # default TRUE if WRITE_TO_SHEETS_FROM_API=true, else FALSE
#   RETURN_UPLOADED_IMAGES             # default FALSE; when TRUE, include array in JSON response
#
# Notes:
# - If you set WRITE_TO_SHEETS_FROM_API=false, the endpoint will **not** append
#   to Sheets; it will then return a small JSON summary (and, optionally,
#   `uploaded_images` when RETURN_UPLOADED_IMAGES=true) so that n8n can handle
#   sheet writes in a different workflow.

import os
import re
import io
import sys
import time
import logging
import urllib.request
from typing import Any, Dict, List, Optional

import numpy as np
import cv2
from PIL import Image

from fastapi import FastAPI, Body, Header, HTTPException, Request
from fastapi.responses import RedirectResponse, PlainTextResponse, Response

from pydantic import BaseModel, Field

from google_auth_oauthlib.flow import Flow
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request as GoogleRequest
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload
from googleapiclient.errors import HttpError

# Import your cropper from the shared module
from extract_core import extract_crops_streamlit, configure_debug as core_configure_debug

# Allow http redirect for local dev
if os.getenv("GOOGLE_REDIRECT_URI", "").startswith(("http://localhost", "http://127.0.0.1")):
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"
    os.environ.setdefault("OAUTHLIB_RELAX_TOKEN_SCOPE", "1")

# ── Logging ──────────────────────────────────────────────────────────────────
LOG_LEVEL = os.environ.get("LOG_LEVEL", "DEBUG").upper()
api_logger = logging.getLogger("api.extractor")
if not any(not isinstance(h, logging.NullHandler) for h in api_logger.handlers):
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.DEBUG)
    sh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s", "%H:%M:%S"))
    api_logger.addHandler(sh)
api_logger.setLevel(LOG_LEVEL)
try:
    core_configure_debug(logging.getLevelName(LOG_LEVEL))
except Exception:
    pass

# ── App ─────────────────────────────────────────────────────────────────────
app = FastAPI(title="OCR Crop Extractor API (OAuth)")

# ── OAuth client config ──────────────────────────────────────────────────────
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
    return os.environ.get("GOOGLE_REFRESH_TOKEN") or None


def _require_refresh_token() -> str:
    rt = _get_refresh_token_from_env()
    if not rt:
        raise RuntimeError("No Google refresh token set. Visit /google/auth, copy refresh token from logs, set GOOGLE_REFRESH_TOKEN, then redeploy.")
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
    creds = _make_credentials_from_refresh()
    drive = build("drive", "v3", credentials=creds, cache_discovery=False)
    sheets = build("sheets", "v4", credentials=creds, cache_discovery=False)
    return drive, sheets

# ── Security ─────────────────────────────────────────────────────────────────
API_TOKEN = os.environ.get("API_TOKEN", "change-me")


def _require_auth(authorization: Optional[str]):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = authorization.split(" ", 1)[1]
    if token != API_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token")

# ── Models ───────────────────────────────────────────────────────────────────
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

# ── Utils ────────────────────────────────────────────────────────────────────

def _safe_url(u: str, keep_query: bool = False, maxlen: int = 140) -> str:
    try:
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


def _bytes_from_url(url: str, timeout: int = 15) -> bytes:
    safe = _safe_url(url)
    api_logger.debug(f"[HTTP] GET start {safe} timeout={timeout}s")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        api_logger.debug(f"[HTTP] GET ok {safe} -> {len(data)} bytes")
        return data
    except Exception as e:
        api_logger.error(f"[HTTP] GET fail {safe}: {e}")
        raise



def _pil_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _page_num_from_filename(name: str) -> int:
    if not name:
        return -1
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

# ── Google helpers (Drive upload + Sheets append) ────────────────────────────
SHEET_ID = os.environ.get("SHEET_ID", "").strip()
SHEET_RANGE = os.environ.get("SHEET_RANGE", "Sheet1!A:D")
DRIVE_FOLDER_ID = os.environ.get("DRIVE_FOLDER_ID", "").strip()
MAKE_PUBLIC = os.environ.get("MAKE_PUBLIC", "").lower() in ("1", "true", "yes")

# Defaults for this build: API writes the Sheet & returns 204 No Content
WRITE_TO_SHEETS_FROM_API = os.environ.get("WRITE_TO_SHEETS_FROM_API", "true").lower() in ("1", "true", "yes")
NO_CONTENT_RESPONSE = os.environ.get(
    "NO_CONTENT_RESPONSE",
    "true" if WRITE_TO_SHEETS_FROM_API else "false",
).lower() in ("1", "true", "yes")
RETURN_UPLOADED_IMAGES = os.environ.get("RETURN_UPLOADED_IMAGES", "false").lower() in ("1", "true", "yes")


def upload_png_to_drive(png_bytes: bytes, filename: str, folder_id: Optional[str]) -> str:
    drive, _ = get_services()
    media = MediaIoBaseUpload(io.BytesIO(png_bytes), mimetype="image/png", resumable=False)

    parent_id = (folder_id or DRIVE_FOLDER_ID).strip() or None
    metadata: Dict[str, Any] = {"name": filename, "mimeType": "image/png"}
    if parent_id:
        metadata["parents"] = [parent_id]

    try:
        file = drive.files().create(
            body=metadata,
            media_body=media,
            fields="id,name,parents,owners(emailAddress,displayName),driveId,webViewLink,webContentLink",
            supportsAllDrives=True,
        ).execute()
    except HttpError as e:
        api_logger.error(f"[DRIVE] create failed for {filename} in parent={parent_id!r}: {e}")
        raise

    file_id = file.get("id")
    owners = file.get("owners", [])
    owner_email = owners[0].get("emailAddress") if owners else "?"
    link = file.get("webViewLink") or file.get("webContentLink") or f"https://drive.google.com/file/d/{file_id}/view"

    api_logger.debug(
        "[DRIVE] uploaded name=%r id=%s parents=%s driveId=%s owner=%s link=%s",
        file.get("name"), file_id, file.get("parents"), file.get("driveId"), owner_email, _safe_url(link)
    )

    if MAKE_PUBLIC:
        try:
            drive.permissions().create(
                fileId=file_id,
                body={"role": "reader", "type": "anyone"},
                supportsAllDrives=True,
            ).execute()
            api_logger.debug("[DRIVE] made public (anyone with link)")
        except HttpError as e:
            api_logger.warning(f"[DRIVE] make public failed: {e}")

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

# ── Routes ───────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    ok = True
    has_rt = bool(_get_refresh_token_from_env())
    return {"ok": ok, "has_refresh_token": has_rt, "sheet": bool(SHEET_ID)}


@app.get("/google/auth")
def google_auth():
    flow = Flow.from_client_config(_client_cfg(), scopes=SCOPES, redirect_uri=GOOGLE_REDIRECT_URI)
    url, _state = flow.authorization_url(access_type="offline", include_granted_scopes="true", prompt="consent")
    return RedirectResponse(url)


@app.get("/oauth2/callback")
def oauth2_callback(request: Request):
    flow = Flow.from_client_config(_client_cfg(), scopes=SCOPES, redirect_uri=GOOGLE_REDIRECT_URI)
    flow.fetch_token(authorization_response=str(request.url))
    creds = flow.credentials
    if not creds.refresh_token:
        raise HTTPException(
            status_code=400,
            detail="No refresh_token returned. Revoke app at https://myaccount.google.com/permissions and retry.",
        )
    print("=== COPY & SAVE AS GOOGLE_REFRESH_TOKEN ===")
    print(creds.refresh_token)
    print("==========================================")
    return PlainTextResponse("Google connected. Copy refresh token from logs, set GOOGLE_REFRESH_TOKEN, then redeploy.")


@app.post("/process-convertapi-json")
def process_convertapi_json(
    payload: ConvertApiPayload = Body(...),
    authorization: Optional[str] = Header(default=None),
) -> Any:
    _require_auth(authorization)

    run_id = f"run-{int(time.time())}"
    files = payload.Files or []
    api_logger.debug(f"[BEGIN] {run_id} files_count={len(files)} pdf_name={payload.SourcePdfName!r} settings_present={bool(payload.settings)}")

    if not files:
        raise HTTPException(status_code=400, detail="No files found in payload.")

    # Warm up Google clients (refresh token flow)
    t0 = time.time()
    try:
        _ = get_services()
        api_logger.debug(f"[GOOGLE] {run_id} services ready in {time.time()-t0:.2f}s")
    except Exception as e:
        api_logger.exception(f"[GOOGLE] {run_id} get_services failed: {e}")
        raise HTTPException(status_code=500, detail=f"Google auth error: {e}")

    pdf_name = payload.SourcePdfName or "input.pdf"
    s = ExtractionSettings()  # (use defaults for now)

    rows: List[List[Any]] = []
    total_crops = 0
    errors: List[str] = []

    # Optional payload for response (kept minimal by default)
    uploaded_images: List[Dict[str, Any]] = []

    api_logger.debug(f"[LOOP] {run_id} Start processing {len(files)} pages")

    for i, f in enumerate(files, start=1):
        pstart = time.time()
        try:
            page_num_guess = _page_num_from_filename(getattr(f, "FileName", ""))
            api_logger.debug(f"[PAGE] {run_id} i={i} fileName={f.FileName!r} page_num_guess={page_num_guess} url={_safe_url(getattr(f, 'FileUrl', ''))}")

            # Download page image
            page_png = _bytes_from_url(f.FileUrl)
            img_arr = np.frombuffer(page_png, dtype=np.uint8)
            img_bgr = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
            if img_bgr is None:
                raise RuntimeError("Failed to decode page image")
            _img_stats_arr(f"PAGE[{i}].bgr", img_bgr)

            # Crop extraction
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
            api_logger.debug(f"[CROPPER] {run_id} i={i} crops_found={len(crops)} crop_ms={(time.time()-t_crop)*1000.0:.1f}")

            page_num = _page_num_from_filename(f.FileName)
            if not crops:
                continue 
            else:
                for ci, rgb in enumerate(crops, start=1):
                    pil = Image.fromarray(rgb)
                    crop_bytes = _pil_to_png_bytes(pil)
                    out_name = f"{os.path.splitext(pdf_name)[0]}_p{page_num:02d}_img{ci:02d}.png"
                    link = upload_png_to_drive(crop_bytes, out_name, None)

                    rows.append([pdf_name, page_num, ci, link])
                    total_crops += 1

                    if RETURN_UPLOADED_IMAGES:
                        uploaded_images.append({
                            "file_name": out_name,
                            "page_number": int(page_num),
                            "image_number": int(ci),
                            "image_url": link,
                        })

            api_logger.debug(f"[PAGE] {run_id} i={i} done in {(time.time()-pstart)*1000.0:.1f} ms")
        except Exception as e:
            err = f"Page {i} ({getattr(f, 'FileName', '')}): {e}"
            errors.append(err)
            rows.append([pdf_name, _page_num_from_filename(getattr(f, 'FileName', '')), -1, f"ERROR: {e}"])
            api_logger.error(f"[EXC] {run_id} i={i} {err}")

    # Sheet write (server-side by default)
    if WRITE_TO_SHEETS_FROM_API:
        append_rows_to_sheet(rows)
    else:
        api_logger.debug("[SHEETS] Skipped server-side append (WRITE_TO_SHEETS_FROM_API is false)")

    took = round(time.time() - t0, 2)
    api_logger.debug(
        f"[END] {run_id} pages_received={len(files)} crops_uploaded={total_crops} rows_appended={len(rows)} seconds={took} errors_count={len(errors)}"
    )

    # If NO_CONTENT_RESPONSE is true, respond 204 so n8n doesn't parse JSON
    if NO_CONTENT_RESPONSE and WRITE_TO_SHEETS_FROM_API:
        return Response(status_code=204)

    # Otherwise a compact JSON summary (optionally with uploaded_images)
    resp: Dict[str, Any] = {
        "ok": True,
        "pages_received": len(files),
        "crops_uploaded": total_crops,
        "rows_appended": len(rows),
        "seconds": took,
        "errors": errors,
        "run_id": run_id,
        "source_pdf_name": pdf_name,
    }
    if RETURN_UPLOADED_IMAGES:
        resp["uploaded_images"] = uploaded_images
    return resp
