"""
Central config loaded from environment.
Used by server.py and test_pipeline.py so S3 bucket and API URL are not passed everywhere.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# S3: default bucket for video input/output (used when not provided per-request)
S3_VIDEO_BUCKET = os.getenv("S3_VIDEO_BUCKET", "youdescribe-downloaded-youtube-videos")

# API: base URL of the YouDescribe Node.js API (for forwarding results)
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:4001")
