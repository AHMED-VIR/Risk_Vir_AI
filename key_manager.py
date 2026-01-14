import os
import logging
from typing import Optional
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

class KeyManager:
    def __init__(self):
        self.api_key: str = self._load_key()
        self.client: OpenAI = OpenAI(api_key=self.api_key)
        
        if not self.api_key:
            raise ValueError("No API key found in .env under OPENAI_API_KEY")

        logging.info("KeyManager initialized with OpenAI client.")

    def _load_key(self) -> str:
        """
        Loads key from OPENAI_API_KEY.
        """
        key = os.getenv("OPENAI_API_KEY")
        if key:
            return key.strip()
        return ""

    def get_client(self) -> OpenAI:
        """Returns the OpenAI client."""
        return self.client
