# test_gemini_key.py
import os
import requests
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash:generateContent?key={api_key}"
payload = {"contents": [{"parts": [{"text": "Say hello in one word."}]}]}

response = requests.post(url, json=payload, timeout=45)
print("Status:", response.status_code)
print(response.json())