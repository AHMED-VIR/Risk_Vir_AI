from google import genai
import os

# Gemini API Key
GEMINI_API_KEY = "AIzaSyBTJdeA1Hp9v9me5fpRL1j4Yt5WHjO-3Wg"

client = genai.Client(api_key=GEMINI_API_KEY)

try:
    print("Listing models...")
    # List models that support generateContent
    for model in client.models.list(config={"query_base": True}):
        print(f"Model: {model.name}")
        print(f"  DisplayName: {model.display_name}")
        print(f"  Supported Actions: {model.supported_actions}")
        print("-" * 20)
        
except Exception as e:
    print(f"Error listing models: {e}")
