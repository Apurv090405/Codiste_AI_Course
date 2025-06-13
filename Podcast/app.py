import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from podcast_generator import generate_podcast
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Podcast Generator API")

class PodcastRequest(BaseModel):
    topic: str
    description: str
    duration: int
    speakers: int
    location: str

@app.post("/generate_podcast")
async def generate_podcast_endpoint(request: PodcastRequest):
    try:
        if not all([request.topic, request.description, request.location]):
            raise HTTPException(status_code=400, detail="All fields are required.")
        if not 10 <= request.duration <= 120:
            raise HTTPException(status_code=400, detail="Duration must be between 10 and 120 minutes.")
        if request.speakers not in [1, 2, 3]:
            raise HTTPException(status_code=400, detail="Speakers must be 1, 2, or 3.")
        
        result = generate_podcast(
            topic=request.topic,
            description=request.description,
            duration=request.duration,
            speakers=request.speakers,
            location=request.location
        )
        
        if result.get("error"):
            raise HTTPException(status_code=500, detail=f"Error: {result['error']}")
        
        return {"filename": result["filename"], "message": f"Podcast saved to {result['filename']}"}
    except Exception as e:
        logger.error(f"API error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)