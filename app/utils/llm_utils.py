import os
import requests
import json
from typing import Optional
from ..config import settings


def generate_summary(content: str) -> str:
    """
    Generate a summary of SEC filing content using Groq LLM API.
    """
    
    deepseek_url = settings.DEEPSEEK_API_URL
    headers = {"Content-Type": "application/json"}
    
    prompt = f"Summarize the following SEC filing content concisely, focusing on key financial information and risks. Keep the summary under 3 sentences:\n\n{content}"
    
    data = {
        "prompt": prompt,
        "stream": False
    }
    
    try:
        response = requests.post(deepseek_url, headers=headers, json=data)
        response.raise_for_status()
        summary = response.json()["response"]
        return summary
    except Exception as e:
        return f"Error generating summary: {str(e)}"
