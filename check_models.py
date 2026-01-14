import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

api_key = os.getenv("OPENAI_API_KEY")
print(f"Checking key: {api_key[:10]}...")

try:
    client = OpenAI(api_key=api_key)
    models = client.models.list()
    print("Available models:")
    for model in models:
        print(f"- {model.id}")
except Exception as e:
    print(f"Error listing models: {e}")
