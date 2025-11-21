import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
import aiofiles

app = FastAPI(title="Media Service")

# Configuration
MEDIA_DIR = Path(os.getenv("MEDIA_DIR", "/app/outputs"))
MEDIA_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 1024 * 1024  # 1MB chunks for streaming

@app.get("/")
async def root():
    return {"service": "Media Service", "status": "running"}

@app.get("/media/{job_id}")
async def get_media(job_id: str, stream: bool = False):
    """
    Download or stream the processed video
    """
    # Look for output file
    output_file = MEDIA_DIR / f"{job_id}_output.mp4"
    
    if not output_file.exists():
        # Try alternative naming
        for file in MEDIA_DIR.glob(f"{job_id}*"):
            if file.suffix in ['.mp4', '.avi', '.mov']:
                output_file = file
                break
    
    if not output_file.exists():
        raise HTTPException(status_code=404, detail="Media file not found")
    
    if stream:
        # Stream the file
        async def iterfile():
            async with aiofiles.open(output_file, 'rb') as f:
                while chunk := await f.read(CHUNK_SIZE):
                    yield chunk
        
        return StreamingResponse(
            iterfile(),
            media_type="video/mp4",
            headers={
                "Content-Disposition": f"inline; filename={output_file.name}"
            }
        )
    else:
        # Direct download
        return FileResponse(
            path=output_file,
            media_type="video/mp4",
            filename=output_file.name
        )

@app.get("/media/{job_id}/info")
async def get_media_info(job_id: str):
    """
    Get information about a media file
    """
    output_file = MEDIA_DIR / f"{job_id}_output.mp4"
    
    if not output_file.exists():
        for file in MEDIA_DIR.glob(f"{job_id}*"):
            if file.suffix in ['.mp4', '.avi', '.mov']:
                output_file = file
                break
    
    if not output_file.exists():
        raise HTTPException(status_code=404, detail="Media file not found")
    
    stat = output_file.stat()
    
    return {
        "job_id": job_id,
        "filename": output_file.name,
        "size": stat.st_size,
        "size_mb": round(stat.st_size / (1024 * 1024), 2),
        "created": stat.st_ctime,
        "modified": stat.st_mtime,
        "exists": True
    }

@app.get("/media/list")
async def list_media(limit: int = 10):
    """
    List available media files
    """
    files = []
    for file in sorted(MEDIA_DIR.glob("*.mp4"), key=lambda x: x.stat().st_mtime, reverse=True)[:limit]:
        stat = file.stat()
        job_id = file.stem.replace("_output", "")
        files.append({
            "job_id": job_id,
            "filename": file.name,
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
            "modified": stat.st_mtime
        })
    
    return {
        "files": files,
        "count": len(files)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)