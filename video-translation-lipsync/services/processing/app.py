import os
import uuid
import sqlite3
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import asyncio
from concurrent.futures import ThreadPoolExecutor
from pipeline import TranslationPipeline

app = FastAPI(title="Processing Service")

# Configuration
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/app/outputs"))
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "/app/uploads"))
DB_PATH = OUTPUT_DIR / "jobs.db"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Thread pool for async processing
executor = ThreadPoolExecutor(max_workers=2)

# Initialize database
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            job_id TEXT PRIMARY KEY,
            upload_id TEXT NOT NULL,
            target_language TEXT NOT NULL,
            status TEXT NOT NULL,
            progress INTEGER DEFAULT 0,
            output_path TEXT,
            error_message TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

init_db()

# Initialize pipeline
pipeline = TranslationPipeline()

def get_db_connection():
    return sqlite3.connect(DB_PATH)

class ProcessRequest(BaseModel):
    upload_id: str
    target_language: str = "es"

def update_job_status(job_id: str, status: str, progress: int = 0, 
                      output_path: str = None, error_message: str = None):
    """Update job status in database"""
    conn = get_db_connection()
    c = conn.cursor()
    
    if status == "completed" or status == "failed":
        c.execute("""
            UPDATE jobs 
            SET status = ?, progress = ?, output_path = ?, 
                error_message = ?, completed_at = CURRENT_TIMESTAMP
            WHERE job_id = ?
        """, (status, progress, output_path, error_message, job_id))
    else:
        c.execute("""
            UPDATE jobs 
            SET status = ?, progress = ?, output_path = ?, error_message = ?
            WHERE job_id = ?
        """, (status, progress, output_path, error_message, job_id))
    
    conn.commit()
    conn.close()

def process_video_task(job_id: str, upload_id: str, input_path: str, 
                       output_path: str, target_language: str):
    """Background task to process video"""
    try:
        # Update status: ASR
        update_job_status(job_id, "processing", 20, None, None)
        
        # Run pipeline
        pipeline.process(
            input_path=input_path,
            output_path=output_path,
            target_language=target_language,
            job_id=job_id,
            status_callback=update_job_status
        )
        
        # Mark as completed
        update_job_status(job_id, "completed", 100, output_path, None)
        
    except Exception as e:
        error_msg = str(e)
        update_job_status(job_id, "failed", 0, None, error_msg)

@app.get("/")
async def root():
    return {"service": "Processing Service", "status": "running"}

@app.post("/process")
async def start_processing(request: ProcessRequest):
    """
    Start processing pipeline for an uploaded file
    """
    # Check if upload exists
    upload_db = UPLOAD_DIR / "uploads.db"
    if not upload_db.exists():
        raise HTTPException(status_code=404, detail="Upload service database not found")
    
    conn = sqlite3.connect(upload_db)
    c = conn.cursor()
    c.execute("SELECT filepath, filename FROM uploads WHERE id = ?", (request.upload_id,))
    row = c.fetchone()
    conn.close()
    
    if not row:
        raise HTTPException(status_code=404, detail="Upload not found")
    
    input_path = row[0]
    if not Path(input_path).exists():
        raise HTTPException(status_code=404, detail="Uploaded file not found")
    
    # Create job
    job_id = str(uuid.uuid4())
    output_filename = f"{job_id}_output.mp4"
    output_path = str(OUTPUT_DIR / output_filename)
    
    # Save job to database
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("""
        INSERT INTO jobs (job_id, upload_id, target_language, status, progress)
        VALUES (?, ?, ?, ?, ?)
    """, (job_id, request.upload_id, request.target_language, "queued", 0))
    conn.commit()
    conn.close()
    
    # Start processing in background
    loop = asyncio.get_event_loop()
    loop.run_in_executor(
        executor,
        process_video_task,
        job_id,
        request.upload_id,
        input_path,
        output_path,
        request.target_language
    )
    
    return JSONResponse({
        "job_id": job_id,
        "upload_id": request.upload_id,
        "target_language": request.target_language,
        "status": "queued",
        "message": "Processing started"
    })

@app.get("/process/{job_id}/status")
async def get_job_status(job_id: str):
    """
    Check the status of a processing job
    """
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,))
    row = c.fetchone()
    conn.close()
    
    if not row:
        raise HTTPException(status_code=404, detail="Job not found")
    
    response = {
        "job_id": row[0],
        "upload_id": row[1],
        "target_language": row[2],
        "status": row[3],
        "progress": row[4],
        "output_path": row[5],
        "error_message": row[6],
        "created_at": row[7],
        "completed_at": row[8]
    }
    
    return JSONResponse(response)

@app.get("/process/jobs/list")
async def list_jobs(limit: int = 10):
    """
    List recent processing jobs
    """
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("""
        SELECT job_id, upload_id, target_language, status, progress, created_at
        FROM jobs
        ORDER BY created_at DESC
        LIMIT ?
    """, (limit,))
    rows = c.fetchall()
    conn.close()
    
    jobs = []
    for row in rows:
        jobs.append({
            "job_id": row[0],
            "upload_id": row[1],
            "target_language": row[2],
            "status": row[3],
            "progress": row[4],
            "created_at": row[5]
        })
    
    return JSONResponse({"jobs": jobs, "count": len(jobs)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)