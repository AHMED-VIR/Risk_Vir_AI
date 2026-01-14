import os
import logging
from typing import Optional
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

class KeyManager:
    def __init__(self):
        self.api_keys = self._load_keys()
        self.current_key_index = 0
        self.clients = [OpenAI(api_key=key) for key in self.api_keys]
        
        if not self.clients:
            raise ValueError("No API keys found in .env under OPENAI_API_KEY")

        logging.info(f"KeyManager initialized with {len(self.clients)} OpenAI clients.")

    def _load_keys(self) -> list[str]:
        """
        Loads keys from OPENAI_API_KEY (comma separated).
        """
        keys_str = os.getenv("OPENAI_API_KEY")
        if keys_str:
            # Split by comma and strip whitespace
            return [k.strip() for k in keys_str.split(',') if k.strip()]
        return []

    def get_client(self) -> OpenAI:
        """Returns the next OpenAI client in rotation."""
        if not self.clients:
             raise ValueError("No OpenAI clients available.")
        
        client = self.clients[self.current_key_index]
        # Rotate index
        self.current_key_index = (self.current_key_index + 1) % len(self.clients)
        return client
