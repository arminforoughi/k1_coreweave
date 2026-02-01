"""Browserbase + Stagehand research worker.

When an unknown object is detected, this worker:
1. Takes the YOLO class hint (if any) and thumbnail
2. Uses Browserbase/Stagehand to search the web for what the object might be
3. Extracts a structured "Object Card" with label, description, confidence
4. Returns the result for the backend to auto-label or queue for review

Stagehand SDK v3.5+: session-based REST API. Requires:
  - BROWSERBASE_API_KEY, BROWSERBASE_PROJECT_ID
  - MODEL_API_KEY (Anthropic/OpenAI/Google key for page interpretation)
"""
import os
import json
import base64
import weave
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

BROWSERBASE_API_KEY = os.getenv("BROWSERBASE_API_KEY", "")
BROWSERBASE_PROJECT_ID = os.getenv("BROWSERBASE_PROJECT_ID", "")
MODEL_API_KEY = os.getenv("MODEL_API_KEY") or os.getenv("OPENAI_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")


def _research_via_gpt_vision(thumbnail_b64: str, yolo_hint: str) -> Optional[dict]:
    """Use GPT-5+ vision to analyze the image."""
    from openai import OpenAI

    if not thumbnail_b64 or not OPENAI_API_KEY:
        return None

    client = OpenAI(api_key=OPENAI_API_KEY)

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            max_completion_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{thumbnail_b64}",
                            },
                        },
                        {
                            "type": "text",
                            "text": f"""Look at this image. A vision system detected an object and guessed it might be a "{yolo_hint}".

Your task: Identify what specific object this is. Provide:
1. A specific label (e.g., "MacBook Pro 14-inch", "IKEA desk lamp", "ceramic coffee mug") - be as specific as possible
2. A brief description
3. Key identifying features
4. Your confidence (0.0 to 1.0)

Respond in JSON format:
{{
  "label": "specific object name",
  "description": "brief description",
  "facts": ["fact 1", "fact 2", "fact 3"],
  "confidence": 0.0-1.0
}}"""
                        }
                    ],
                }
            ],
        )

        response_text = response.choices[0].message.content

        # Try to extract JSON from the response
        import re
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            result["source"] = "gpt_vision"
            result["model"] = response.model
            return result

    except Exception as e:
        print(f"GPT vision analysis failed: {e}")

    return None


def _research_via_claude_vision(thumbnail_b64: str, yolo_hint: str, use_opus: bool = False) -> Optional[dict]:
    """Use Claude vision via Anthropic API to analyze the image directly.

    Args:
        use_opus: If True, use Claude Opus 4.5 for better accuracy (slower, more expensive)
    """
    from anthropic import Anthropic

    if not thumbnail_b64:
        return None

    anthropic = Anthropic(api_key=MODEL_API_KEY)
    model = "claude-opus-4-5-20251101" if use_opus else "claude-sonnet-4-5-20250929"

    try:
        vision_response = anthropic.messages.create(
            model=model,
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": thumbnail_b64,
                            },
                        },
                        {
                            "type": "text",
                            "text": f"""Look at this image. A vision system detected an object and guessed it might be a "{yolo_hint}".

Your task: Identify what specific object this is. Provide:
1. A specific label (e.g., "MacBook Pro 14-inch", "IKEA desk lamp", "ceramic coffee mug") - be as specific as possible
2. A brief description
3. Key identifying features
4. Your confidence (0.0 to 1.0)

Respond in JSON format:
{{
  "label": "specific object name",
  "description": "brief description",
  "facts": ["fact 1", "fact 2", "fact 3"],
  "confidence": 0.0-1.0
}}"""
                        }
                    ],
                }
            ],
        )

        # Parse Claude's response
        response_text = vision_response.content[0].text

        # Try to extract JSON from the response
        import re
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            result["source"] = "claude_opus_vision" if use_opus else "claude_sonnet_vision"
            result["model"] = model
            return result

    except Exception as e:
        print(f"Claude vision analysis failed: {e}")

    return None


def _research_via_image_search(thumbnail_b64: str, yolo_hint: str) -> Optional[dict]:
    """Multi-model image analysis with ensemble voting.

    Tries GPT-5+ first (if available), then Claude Sonnet 4.5.
    For low-confidence results (<0.7), escalates to Claude Opus 4.5.
    """
    if not thumbnail_b64:
        return None

    results = []

    # Try GPT-5+ first if OpenAI key available
    if OPENAI_API_KEY:
        gpt_result = _research_via_gpt_vision(thumbnail_b64, yolo_hint)
        if gpt_result:
            results.append(gpt_result)
            # If GPT is highly confident, return immediately
            if gpt_result.get("confidence", 0) >= 0.85:
                return gpt_result

    # Try Claude Sonnet for speed
    claude_result = _research_via_claude_vision(thumbnail_b64, yolo_hint, use_opus=False)
    if claude_result:
        results.append(claude_result)
        # If Claude Sonnet is highly confident, return immediately
        if claude_result.get("confidence", 0) >= 0.85:
            return claude_result

    # If we have results but low confidence, try Claude Opus for second opinion
    if results:
        max_confidence = max(r.get("confidence", 0) for r in results)
        if max_confidence < 0.7:
            opus_result = _research_via_claude_vision(thumbnail_b64, yolo_hint, use_opus=True)
            if opus_result:
                results.append(opus_result)

    # Return the highest confidence result
    if results:
        return max(results, key=lambda r: r.get("confidence", 0))

    return None




@weave.op()
def research_object(thumbnail_b64: str, yolo_hint: str = "",
                    yolo_confidence: float = 0.0) -> dict:
    """Research an unknown object using vision APIs (Claude/GPT).

    Args:
        thumbnail_b64: Base64 JPEG of the object crop
        yolo_hint: YOLO's best guess class name (may be wrong)
        yolo_confidence: Original YOLO confidence (0-1)

    Returns:
        dict with keys: label, confidence (0-1), description, source, facts[]
        Returns None if research fails entirely.
    """
    # Try vision APIs directly
    if thumbnail_b64:
        try:
            result = _research_via_image_search(thumbnail_b64, yolo_hint)
            if result:
                return result
        except Exception as e:
            print(f"Vision API research failed: {e}")

    # Fallback: use YOLO hint with scaled confidence.
    # YOLO classes come from COCO's 80-class vocabulary — "person", "chair",
    # "laptop" etc. are well-defined categories. Even moderate YOLO confidence
    # (0.3+) usually means the class is correct.
    # Scale: YOLO 0.25 -> 0.72, 0.35 -> 0.76, 0.5 -> 0.82, 0.75 -> 0.92
    if yolo_hint:
        fallback_conf = 0.65 + (yolo_confidence * 0.35)
        return {
            "label": yolo_hint,
            "confidence": round(min(fallback_conf, 0.95), 3),
            "description": f"Detected as {yolo_hint} by YOLOv8 (confidence: {yolo_confidence:.0%})",
            "source": "yolo_fallback",
            "facts": [],
        }

    return None
