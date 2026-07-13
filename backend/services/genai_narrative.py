# backend/services/genai_narrative.py
import os
import requests

class GeminiExplainer:
    @staticmethod
    def generate_narrative(proposal_title: str, metrics: dict, extracted_keywords: list) -> str:
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key or api_key.strip() == "" or "Gemini_Key" in api_key:
            return (
                f"The system has processed '{proposal_title}' with a novelty score of {metrics['novelty_score']}/100 "
                f"and a financial feasibility score of {metrics['financial_score']}/100. Key focus areas detected "
                f"include: {', '.join(extracted_keywords)}. Highly recommended for priority human review."
            )

        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-flash-lite:generateContent?key={api_key}"

        prompt = (
            f"You are an expert technical risk auditor evaluating an R&D proposal titled '{proposal_title}'.\n"
            f"The quantitative machine learning pipeline returned these metrics:\n"
            f"- Novelty Score: {metrics['novelty_score']}/100\n"
            f"- Financial Feasibility Score: {metrics['financial_score']}/100\n"
            f"- Extracted Technical Elements: {', '.join(extracted_keywords)}\n\n"
            f"Write a concise, professional 5 to 10 sentence executive summary explaining the verdict. "
            f"Highlight why it was flagged or recommended based on the scores. Be direct and objective."
        )

        payload = {"contents": [{"parts": [{"text": prompt}]}]}

        try:
            response = requests.post(url, json=payload, timeout=60)
        except Exception as e:
            print(f"\n[Gemini Network Exception] {str(e)}\n")
            return "Executive summary generation timed out. Classification models processed successfully."

        if response.status_code != 200:
            print(f"\n[Gemini API Error] Status Code: {response.status_code}")
            print(f"[Gemini API Error] Response Body: {response.text}\n")
            return "Executive summary generation timed out. Classification models processed successfully."

        try:
            result = response.json()
            return result['candidates'][0]['content']['parts'][0]['text'].strip()
        
        except (KeyError, IndexError, ValueError) as e:
            print(f"\n[Gemini Response Parsing Error] {str(e)}")
            print(f"[Gemini Response Parsing Error] Raw response: {response.text}\n")
            return "Executive summary generation timed out. Classification models processed successfully."