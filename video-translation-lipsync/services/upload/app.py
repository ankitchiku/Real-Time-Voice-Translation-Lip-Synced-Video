import os
import uuid
import sqlite3
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import aiofiles
from datetime import datetime

app = FastAPI(title="Upload Service")

# Configuration
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "/app/uploads"))
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", 104857600))  # 100MB
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 8388608))  # 8MB
DB_PATH = UPLOAD_DIR / "uploads.db"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Initialize database
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS uploads (
            id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            filepath TEXT NOT NULL,
            size INTEGER NOT NULL,
            target_language TEXT,
            status TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

init_db()

# Allowed file extensions
ALLOWED_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.mp3', '.wav', '.m4a'}

def get_db_connection():
    return sqlite3.connect(DB_PATH)

@app.get("/")
async def root():
    return {"service": "Upload Service", "status": "running"}

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    target_language: str = Form("es")
):
    """
    Upload a video or audio file with chunked upload support
    """
    # Validate file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File type not supported. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    # Generate unique upload ID
    upload_id = str(uuid.uuid4())
    safe_filename = f"{upload_id}{file_ext}"
    filepath = UPLOAD_DIR / safe_filename
    
    # Stream file to disk in chunks
    total_size = 0
    try:
        async with aiofiles.open(filepath, 'wb') as f:
            while chunk := await file.read(CHUNK_SIZE):
                total_size += len(chunk)
                if total_size > MAX_FILE_SIZE:
                    await f.close()
                    filepath.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large. Maximum size: {MAX_FILE_SIZE} bytes"
                    )
                await f.write(chunk)
    except Exception as e:
        filepath.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")
    
    # Save metadata to database
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("""
        INSERT INTO uploads (id, filename, filepath, size, target_language, status)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (upload_id, file.filename, str(filepath), total_size, target_language, "uploaded"))
    conn.commit()
    conn.close()
    
    return JSONResponse({
        "upload_id": upload_id,
        "filename": file.filename,
        "size": total_size,
        "target_language": target_language,
        "status": "uploaded",
        "message": "File uploaded successfully"
    })

@app.get("/upload/{upload_id}/status")
async def get_upload_status(upload_id: str):
    """
    Check the status of an uploaded file
    """
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM uploads WHERE id = ?", (upload_id,))
    row = c.fetchone()
    conn.close()
    
    if not row:
        raise HTTPException(status_code=404, detail="Upload not found")
    
    return JSONResponse({
        "upload_id": row[0],
        "filename": row[1],
        "filepath": row[2],
        "size": row[3],
        "target_language": row[4],
        "status": row[5],
        "created_at": row[6]
    })

@app.get("/upload/{upload_id}/info")
async def get_upload_info(upload_id: str):
    """
    Get detailed information about an upload
    """
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM uploads WHERE id = ?", (upload_id,))
    row = c.fetchone()
    conn.close()
    
    if not row:
        raise HTTPException(status_code=404, detail="Upload not found")
    
    filepath = Path(row[2])
    file_exists = filepath.exists()
    
    return JSONResponse({
        "upload_id": row[0],
        "filename": row[1],
        "filepath": row[2],
        "size": row[3],
        "target_language": row[4],
        "status": row[5],
        "created_at": row[6],
        "file_exists": file_exists
    })

@app.delete("/upload/{upload_id}")
async def delete_upload(upload_id: str):
    """
    Delete an uploaded file
    """
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT filepath FROM uploads WHERE id = ?", (upload_id,))
    row = c.fetchone()
    
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Upload not found")
    
    # Delete file
    filepath = Path(row[0])
    if filepath.exists():
        filepath.unlink()
    
    # Delete from database
    c.execute("DELETE FROM uploads WHERE id = ?", (upload_id,))
    conn.commit()
    conn.close()
    
    return JSONResponse({
        "upload_id": upload_id,
        "message": "Upload deleted successfully"
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)