from dotenv import load_dotenv
import os

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file.")

MODEL_CONFIG = {
    "model_name": "gemini-1.5-flash",
    "generation_config": {"temperature": 0.7, "max_output_tokens": 4000}
}

GNEWS_CONFIG = {
    "language": "en",
    "max_results": 15
}