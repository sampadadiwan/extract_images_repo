#!/usr/bin/env python3
"""
env_check.py

Purpose:
  Quick diagnostics to verify your environment variables and Google credentials
  before running extractor_api.py.

What it does:
  1) Loads .env (unless --no-dotenv) and prints a redacted summary
  2) Verifies service account JSON path exists and parses it
  3) Initializes Google Drive + Sheets clients
  4) Reads Drive folder metadata and Spreadsheet title (non-destructive)
  5) (Optional) --write-test: creates the worksheet if missing and appends a test row
  6) Prints Tesseract availability

Usage:
  python env_check.py
  python env_check.py --no-dotenv
  python env_check.py --write-test

Required env keys (typically placed in .env):
  GOOGLE_APPLICATION_CREDENTIALS=path/to/service_account.json
  GDRIVE_FOLDER_ID=<drive_folder_id>
  GSHEET_ID=<spreadsheet_id>
  SHEET_NAME=ExtractedImages   # (optional)
  TESSERACT_CMD=...            # (optional, Windows)

Exit codes:
  0 = all good
  1 = some checks failed
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# Load .env first (unless disabled)
try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

def red(s): return f"\033[31m{s}\033[0m"
def green(s): return f"\033[32m{s}\033[0m"
def yellow(s): return f"\033[33m{s}\033[0m"
def blue(s): return f"\033[34m{s}\033[0m"


def print_header(title: str):
    print("\n" + "="*80)
    print(title)
    print("="*80)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-dotenv", action="store_true", help="Do not load .env automatically")
    parser.add_argument("--write-test", action="store_true", help="Append a test row to the Sheet (creates worksheet if missing)")
    args = parser.parse_args()

    if not args.no_dotenv:
        if load_dotenv is None:
            print(yellow("python-dotenv not installed; skipping .env load. (pip install python-dotenv)"))
        else:
            env_loaded = load_dotenv()
            print(blue(f".env loaded: {env_loaded}"))

    # 1) Collect env vars
    GJSON = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    GDRIVE_FOLDER_ID = os.getenv("GDRIVE_FOLDER_ID")
    GSHEET_ID = os.getenv("GSHEET_ID")
    SHEET_NAME = os.getenv("SHEET_NAME", "ExtractedImages")
    TESSERACT_CMD = os.getenv("TESSERACT_CMD")

    print_header("Env summary")
    for k, v in [
        ("GOOGLE_APPLICATION_CREDENTIALS", GJSON),
        ("GDRIVE_FOLDER_ID", GDRIVE_FOLDER_ID),
        ("GSHEET_ID", GSHEET_ID),
        ("SHEET_NAME", SHEET_NAME),
        ("TESSERACT_CMD", TESSERACT_CMD),
    ]:
        shown = v if not v else (v if len(v) < 80 else v[:37] + "..." + v[-20:])
        print(f"  {k}: {shown}")

    ok = True
    # 2) Validate service account JSON path
    print_header("Service account JSON")
    if not GJSON:
        print(red("GOOGLE_APPLICATION_CREDENTIALS is not set.")); ok = False
    else:
        p = Path(GJSON)
        if not p.exists():
            print(red(f"Service account file not found: {p}")); ok = False
        else:
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                client_email = data.get("client_email")
                proj = data.get("project_id")
                print(green("Parsed service account JSON OK"))
                print(f"  client_email: {client_email}")
                print(f"  project_id  : {proj}")
            except Exception as e:
                print(red(f"Failed to parse service account JSON: {e}")); ok = False

    # 3) Initialize Google clients
    drive, sheet = None, None
    try:
        from google.oauth2.service_account import Credentials
        from googleapiclient.discovery import build
        import gspread
        SCOPES = [
            "https://www.googleapis.com/auth/drive",
            "https://www.googleapis.com/auth/spreadsheets",
        ]
        if GJSON:
            creds = Credentials.from_service_account_file(GJSON, scopes=SCOPES)
            drive = build("drive", "v3", credentials=creds)
            gc = gspread.authorize(creds)
            sheet = gc.open_by_key(GSHEET_ID) if GSHEET_ID else None
            print(green("Google Drive/Sheets clients created."))
        else:
            print(red("Skipping Google client init (no GOOGLE_APPLICATION_CREDENTIALS).")); ok = False
    except Exception as e:
        print(red(f"Failed to init Google clients: {e}")); ok = False

    # 4) Drive folder metadata (non-destructive)
    print_header("Drive folder access")
    if drive and GDRIVE_FOLDER_ID:
        try:
            meta = drive.files().get(fileId=GDRIVE_FOLDER_ID, fields="id, name, mimeType").execute()
            print(green(f"Drive folder OK: {meta.get('name')} ({meta.get('id')})"))
        except Exception as e:
            print(red(f"Drive folder access error (share the folder with the service account): {e}")); ok = False
    else:
        print(yellow("Skipping Drive folder check (missing drive client or GDRIVE_FOLDER_ID)."))

    # 5) Sheet metadata (non-destructive)
    print_header("Google Sheet access")
    if sheet:
        try:
            title = sheet.title
            print(green(f"Sheet OK: {title} ({GSHEET_ID})"))
        except Exception as e:
            print(red(f"Sheet access error (share the sheet with the service account): {e}")); ok = False
    else:
        print(yellow("Skipping Sheet check (missing sheet client or GSHEET_ID)."))

    # 6) Optional write test
    if args.write_test and sheet:
        print_header("Write test (append a row)")
        try:
            try:
                ws = sheet.worksheet(SHEET_NAME)
            except Exception:
                ws = sheet.add_worksheet(title=SHEET_NAME, rows=1000, cols=10)
                ws.append_row(["pdf_name", "page_num", "image_num", "drive_file_url"])
            row = ["ENV_CHECK", 0, 0, f"checked at {datetime.utcnow().isoformat()}Z"]
            ws.append_row(row, value_input_option="RAW")
            print(green("Appended a test row successfully."))
        except Exception as e:
            print(red(f"Write test failed: {e}")); ok = False

    # 7) Tesseract visibility
    print_header("Tesseract check")
    try:
        import pytesseract
        if TESSERACT_CMD:
            pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
        ver = pytesseract.get_tesseract_version()
        print(green(f"Tesseract version: {ver}"))
    except Exception as e:
        print(yellow(f"Tesseract not available or not configured: {e}"))

    print_header("Result")
    if ok:
        print(green("All checks passed ✓"))
        sys.exit(0)
    else:
        print(red("Some checks failed. Please fix the above issues."))
        sys.exit(1)


if __name__ == "__main__":
    main()
