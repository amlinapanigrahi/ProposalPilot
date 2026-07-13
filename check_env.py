# check_env.py
from dotenv import load_dotenv
import os

load_dotenv()
key = os.getenv("GEMINI_API_KEY")

print(f"Key found: {bool(key)}")
if key:
    print(f"Length: {len(key)}")
    print(f"Starts with AIza: {key.startswith('AIza')}")
    print(f"First 10 chars: {key[:10]}")
    print(f"Last 5 chars: {key[-5:]}")
else:
    print("Key is None — .env not being read correctly.")