import io
import os
import re
import time
import math
import requests
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel

# Google APIs
import gspread
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload

# ------------------------------
# Config via environment variables
# ------------------------------
GDRIVE_FOLDER_ID = os.getenv("GDRIVE_FOLDER_ID")            # target folder for uploads
GSHEET_ID        = os.getenv("GSHEET_ID")                   # spreadsheet id
SHEET_NAME       = os.getenv("SHEET_NAME", "ExtractedImages")
GOOGLE_SCOPES    = [
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/spreadsheets"
]
SERVICE_ACCOUNT_JSON = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")  # path to service-account json

if not (GDRIVE_FOLDER_ID and GSHEET_ID and SERVICE_ACCOUNT_JSON):
    raise RuntimeError("Set GDRIVE_FOLDER_ID, GSHEET_ID, and GOOGLE_APPLICATION_CREDENTIALS env vars.")

creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_JSON, scopes=GOOGLE_SCOPES)
drive_service = build("drive", "v3", credentials=creds)
gc = gspread.authorize(creds)
sh = gc.open_by_key(GSHEET_ID)
try:
    ws = sh.worksheet(SHEET_NAME)
except gspread.WorksheetNotFound:
    ws = sh.add_worksheet(title=SHEET_NAME, rows=1000, cols=10)
    ws.append_row(["pdf_name", "page_num", "image_num", "drive_file_url"])

app = FastAPI(title="ConvertAPI Post-Processor")

# ------------------------------
# Models
# ------------------------------
class ConvertFile(BaseModel):
    FileName: str
    FileUrl: str

class ConvertApiPayload(BaseModel):
    # Typical ConvertAPI response includes:
    # { "ConversionCost": 1, "Files": [ { "FileName": "...", "FileUrl": "..." }, ... ] }
    Files: List[ConvertFile]
    SourcePdfName: str | None = None

# ------------------------------
# Helpers
# ------------------------------
def _get_page_num_from_filename(name: str) -> int:
    # tries output-12.png -> 12; output.png -> 1
    m = re.findall(r"(\d+)", name)
    return int(m[-1]) if m else 1

def download_png(url: str, timeout=120) -> bytes:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.content

def upload_to_drive_png(data: bytes, filename: str, folder_id: str) -> str:
    media = MediaIoBaseUpload(io.BytesIO(data), mimetype="image/png", resumable=False)
    file_metadata = {"name": filename, "parents": [folder_id]}
    created = drive_service.files().create(
        body=file_metadata, media_body=media, fields="id, webViewLink, webContentLink"
    ).execute()
    # Make it viewable (optional)
    try:
        drive_service.permissions().create(
            fileId=created["id"],
            body={"role": "reader", "type": "anyone"},
            fields="id"
        ).execute()
    except Exception:
        pass
    return created.get("webViewLink") or created.get("webContentLink")

def extract_subimages(page_png_bytes: bytes, pdf_name: str, page_num: int) -> List[bytes]:
    """
    TODO: plug your real extraction here.
    For now, we return the single page image to demonstrate the pipeline.
    Replace this with your OpenCV/Pillow logic that returns a list of sub-image bytes.
    """
    # Example placeholder: pretend we detected 1 sub-image == the full page.
    return [page_png_bytes]

def append_rows(rows: List[List[Any]]):
    # batch append to reduce API calls
    if rows:
        ws.append_rows(rows, value_input_option="RAW")

# ------------------------------
# Endpoint
# ------------------------------
@app.post("/process-convertapi-json")
def process_convertapi_json(payload: ConvertApiPayload = Body(...)):
    files = payload.Files or []
    if not files:
        raise HTTPException(status_code=400, detail="No files found in payload.")

    pdf_name = payload.SourcePdfName or "input.pdf"
    rows_to_append: List[List[Any]] = []

    # simple throttle so we don't hammer ConvertAPI links
    start = time.time()
    for i, f in enumerate(files, start=1):
        try:
            page_num = _get_page_num_from_filename(f.FileName)
            page_png = download_png(f.FileUrl)
            # Extract sub-images (list of bytes)
            subimgs = extract_subimages(page_png, pdf_name, page_num)

            # Save each sub-image to Drive
            for idx, img_bytes in enumerate(subimgs, start=1):
                drive_filename = f"{os.path.splitext(pdf_name)[0]}_p{page_num}_img{idx}.png"
                url = upload_to_drive_png(img_bytes, drive_filename, GDRIVE_FOLDER_ID)
                rows_to_append.append([pdf_name, page_num, idx, url])

            # Optional: light pacing
            if i % 10 == 0:
                time.sleep(0.5)

        except Exception as e:
            # log a row noting failure
            rows_to_append.append([pdf_name, _get_page_num_from_filename(f.FileName), -1, f"ERROR: {e}"])

    append_rows(rows_to_append)
    took = f"{round(time.time()-start, 2)}s"
    return {"ok": True, "pages": len(files), "rows_added": len(rows_to_append), "time": took}
