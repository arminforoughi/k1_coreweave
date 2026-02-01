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
from dotenv import load_dotenv

load_dotenv()

BROWSERBASE_API_KEY = os.getenv("BROWSERBASE_API_KEY", "")
BROWSERBASE_PROJECT_ID = os.getenv("BROWSERBASE_PROJECT_ID", "")
MODEL_API_KEY = os.getenv("MODEL_API_KEY") or os.getenv("OPENAI_API_KEY", "")


def _research_via_stagehand(yolo_hint: str) -> dict | None:
    """Use Stagehand v3.5 session API to search the web."""
    from stagehand import Stagehand

    client = Stagehand(
        browserbase_api_key=BROWSERBASE_API_KEY,
        browserbase_project_id=BROWSERBASE_PROJECT_ID,
        model_api_key=MODEL_API_KEY,
    )

    search_query = f"what is a {yolo_hint} object identify" if yolo_hint else "identify unknown household object"

    session_response = client.sessions.start(model_name="anthropic/claude-sonnet-4-5-20250929")
    session_id = session_response.data.session_id

    try:
        client.sessions.navigate(session_id, url="https://www.google.com")
        client.sessions.act(session_id, input=f'type "{search_query}" in the search box and press Enter to search')

        result = client.sessions.extract(
            session_id,
            instruction=(
                f"Look at the search results. I'm trying to identify an object that "
                f"a camera detected. The detection system guessed it might be a '{yolo_hint}'. "
                f"Extract the most likely identity of this object."
            ),
            schema={
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "description": {"type": "string"},
                    "facts": {"type": "array", "items": {"type": "string"}},
                    "confidence": {"type": "number"},
                },
            },
        )

        if result and hasattr(result, "data"):
            # result.data is a Data object with a 'result' attribute containing the extracted dict
            extracted = result.data.result if hasattr(result.data, 'result') else result.data
            if isinstance(extracted, dict):
                extracted["source"] = "browserbase_search"
                return extracted
            else:
                data = json.loads(str(extracted))
                data["source"] = "browserbase_search"
                return data
    finally:
        try:
            client.sessions.end(session_id)
        except Exception:
            pass

    return None


@weave.op()
def research_object(thumbnail_b64: str, yolo_hint: str = "",
                    yolo_confidence: float = 0.0) -> dict:
    """Research an unknown object using Browserbase + Stagehand.

    Args:
        thumbnail_b64: Base64 JPEG of the object crop
        yolo_hint: YOLO's best guess class name (may be wrong)
        yolo_confidence: Original YOLO confidence (0-1)

    Returns:
        dict with keys: label, confidence (0-1), description, source, facts[]
        Returns None if research fails entirely.
    """
    # Try Stagehand if all keys are present
    if BROWSERBASE_API_KEY and MODEL_API_KEY:
        try:
            result = _research_via_stagehand(yolo_hint)
            if result:
                return result
        except Exception as e:
            print(f"Browserbase research failed: {e}")

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
