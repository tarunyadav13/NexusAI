import os
import time
import logging

from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import errors


ROOT_DIR = Path(__file__).resolve().parents[3]

load_dotenv(ROOT_DIR / ".env")

logger = logging.getLogger(__name__)


class GeminiLLM:

    def __init__(self):

        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError("GEMINI_API_KEY is missing")

        self.client = genai.Client(api_key=api_key)

        self.model = os.getenv(
            "GEMINI_MODEL",
            "gemini-3.8-flash"
        )

        self.max_retries = 3

    def generate(self, prompt: str) -> str:

        for attempt in range(self.max_retries):

            try:

                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt
                )

                if not response.text:
                    raise RuntimeError("Gemini returned an empty response")

                return response.text.strip()

            except errors.APIError as exc:

                status_code = getattr(exc, "code", None)

                # Retry temporary server and rate-limit errors.
                if status_code not in (429, 500, 502, 503, 504):
                    raise

                if attempt == self.max_retries - 1:
                    logger.error(
                        "Gemini request failed after %s attempts",
                        self.max_retries
                    )
                    raise

                wait_time = 2 ** attempt

                logger.warning(
                    "Gemini temporarily unavailable. "
                    "Retrying in %s seconds...",
                    wait_time
                )

                time.sleep(wait_time)

        raise RuntimeError("Gemini request failed")