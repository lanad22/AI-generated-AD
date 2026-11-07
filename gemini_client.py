"""Shared factory for a Gemini client backed by Vertex AI.

Auth comes from Application Default Credentials (the service account in
GOOGLE_APPLICATION_CREDENTIALS). The project is taken from GOOGLE_CLOUD_PROJECT,
falling back to the project_id inside the credentials file. The region comes
from GOOGLE_CLOUD_LOCATION (default "global").
"""
import json
import os

from dotenv import load_dotenv
from google import genai

load_dotenv()

DEFAULT_LOCATION = "global"


def _project_from_credentials():
    cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not cred_path or not os.path.exists(cred_path):
        return None
    try:
        with open(cred_path) as f:
            return json.load(f).get("project_id")
    except (OSError, ValueError):
        return None


def get_gemini_client():
    """Return a google-genai Client that talks to Vertex AI.

    Resolution order:
      1. GOOGLE_CLOUD_PROJECT set -> service-account / ADC auth in that project.
      2. GEMINI_API_KEY set       -> Vertex AI express mode (API key, no project).
      3. project_id in GOOGLE_APPLICATION_CREDENTIALS -> ADC auth in that project.
    """
    location = os.getenv("GOOGLE_CLOUD_LOCATION", DEFAULT_LOCATION)

    explicit_project = os.getenv("GOOGLE_CLOUD_PROJECT")
    if explicit_project:
        return genai.Client(vertexai=True, project=explicit_project, location=location)

    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        return genai.Client(vertexai=True, api_key=api_key)

    cred_project = _project_from_credentials()
    if cred_project:
        return genai.Client(vertexai=True, project=cred_project, location=location)

    raise RuntimeError(
        "No Vertex AI credentials found: set GOOGLE_CLOUD_PROJECT with "
        "GOOGLE_APPLICATION_CREDENTIALS, or GEMINI_API_KEY for express mode."
    )
