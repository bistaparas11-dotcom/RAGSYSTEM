import os
import re
import requests
import json
from typing import Optional

from ..config import settings


def extract_sec_data(content: str, context: str):
    """
    Extract entity relationships from data along with their summary in structured format.
    """
    deepseek_url = settings.DEEPSEEK_API_URL
    headers = {"Content-Type": "application/json"}

    prompt = f"""
    ### Role
    You are an expert financial data extractor specialized in SEC filing analysis.

    ### Task
    Analyze the following SEC filing content and:
    1. provide a 3-sentence summary for the content
    2. extract key entities
    3. extract relationships between entities and create triplets (Source, Relationship, Target)

    ### Output Format (Strict JSON)
    Return only a JSON object with this structure:
    {
        "summary": "3-sentence summary here",
        "entities": [
            {"name": "Entity Name", "type": "Entity Type", "description": "Brief context"}
        ],
        "relationships": [
            {"source": "Entity A", "relationship": "Relationship Type", "target": "Entity B", "description": "Supporting evidence from text"}
        ]
    }

    ### Content to Analyze
    Context: {context}
    Content: {content}
    """

    data = {"prompt": prompt, "stream": False}

    try:
        response = requests.post(deepseek_url, headers=headers, json=data)
        response.raise_for_status()
        response_text = response.json()["response"]

        json_match = re.search(r"\{[\s\S]*\}", response_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        return {"error": "No JSON found in response"}
    except Exception as e:
        return {"error": str(e)}
