# backend/services/genai_narrative.py
import os
import requests

class GeminiExplainer:
    @staticmethod
    def generate_narrative(proposal_title: str, metrics: dict, extracted_keywords: list) -> str:
        api_key = os.getenv("GEMINI_API_KEY")
        
        # Checking if the key is structurally missing or blank
        if not api_key or api_key.strip() == "" or "Your_Actual_Gemini_Key_Here" in api_key:
            return (
                f"The system has processed '{proposal_title}' with a novelty score of {metrics['novelty_score']}% "
                f"and a financial feasibility rating of {metrics['financial_score']}%. Key focus areas detected "
                f"include: {', '.join(extracted_keywords)}. Highly recommended for priority human review."
            )

        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
        
        prompt = (
            f"You are an expert technical risk auditor evaluating an R&D proposal titled '{proposal_title}'.\n"
            f"The quantitative machine learning pipeline returned these metrics:\n"
            f"- Novelty Score: {metrics['novelty_score']}/100\n"
            f"- Financial Feasibility Score: {metrics['financial_score']}/100\n"
            f"- Extracted Technical Elements: {', '.join(extracted_keywords)}\n\n"
            f"Write a concise, professional 3-sentence executive summary explaining the verdict. "
            f"Highlight why it was flagged or recommended based on the scores. Be direct and objective."
        )

        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        
        try:
            response = requests.post(url, json=payload, timeout=25)
            if response.status_code == 200:
                result = response.json()
                return result['candidates'][0]['content']['parts'][0]['text'].strip()
            else:
                # 🚨 PRINT THE EXACT ERROR FROM GOOGLE TO THE TERMINAL CONSOLE
                print(f"\n[Gemini API Error] Status Code: {response.status_code}")
                print(f"[Gemini API Error] Response Body: {response.text}\n")
        except Exception as e:
            print(f"\n[Network Exception] {str(e)}\n")
            
        return "Executive summary generation timed out. Classification models processed successfully."