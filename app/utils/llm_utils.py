import os
import re
import requests
import json
from typing import Optional

from app.config import settings


def build_prompt(content: str, section: str) -> str:
    return (
        "### Role\n"
        "You are an expert financial data extractor specialized in SEC filing analysis.\n\n"
        "### Task\n"
        "Analyze the following SEC filing content and:\n"
        "1. Provide a 3-sentence summary for the content\n"
        "2. Extract key entities\n"
        "3. Extract relationships between entities and create triplets (Source, Relationship, Target)\n\n"
        "### Output Format (Strict JSON, No Markdown, No explanation)\n"
        "Return only a JSON object with this structure:\n"
        '{\n'
        '    "summary": "3-sentence summary here",\n'
        '    "entities": [\n'
        '        {"name": "Entity Name", "type": "COMPANY|PERSON|PRODUCT|REGULATION|METRIC|RISK", "description": "Brief context"}\n'
        '    ],\n'
        '    "relationships": [\n'
        '        {"source": "Entity A", "relationship": "Relationship Type", "target": "Entity B", "description": "Supporting evidence"}\n'
        '    ]\n'
        '}\n\n'
        "### Content to Analyze\n"
        f"Section: {section}\n"
        f"Content: {content}"
    )


def extract_sec_data(content: str, section: str):
    deepseek_url = settings.DEEPSEEK_API_URL
    headers = {"Content-Type": "application/json"}
    data = {"prompt": build_prompt(content, section), "stream": False}

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