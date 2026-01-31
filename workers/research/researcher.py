"""Browserbase + Stagehand research worker.

When an unknown object is detected, this worker:
1. Takes the YOLO class hint (if any) and thumbnail
2. Uses Browserbase/Stagehand to search the web for what the object might be
3. Extracts a structured "Object Card" with label, description, confidence
4. Returns the result for the backend to auto-label or queue for review
"""
import os
import json
import base64
import weave
from dotenv import load_dotenv

load_dotenv()

BROWSERBASE_API_KEY = os.getenv("BROWSERBASE_API_KEY", "")
BROWSERBASE_PROJECT_ID = os.getenv("BROWSERBASE_PROJECT_ID", "")


@weave.op()
def research_object(thumbnail_b64: str, yolo_hint: str = "") -> dict:
    """Research an unknown object using Browserbase + Stagehand.

    Args:
        thumbnail_b64: Base64 JPEG of the object crop
        yolo_hint: YOLO's best guess class name (may be wrong)

    Returns:
        dict with keys: label, confidence (0-1), description, source, facts[]
        Returns None if research fails entirely.
    """
    if not BROWSERBASE_API_KEY:
        # Fallback: use YOLO hint if available
        if yolo_hint:
            return {
                "label": yolo_hint,
                "confidence": 0.3,
                "description": f"Possibly a {yolo_hint} (based on visual similarity, no web research available)",
                "source": "yolo_hint",
                "facts": [],
            }
        return None

    try:
        from stagehand import Stagehand

        # Build search query from YOLO hint
        if yolo_hint:
            search_query = f"what is a {yolo_hint} object identify"
        else:
            search_query = "identify unknown household object"

        with Stagehand(
            api_key=BROWSERBASE_API_KEY,
            project_id=BROWSERBASE_PROJECT_ID,
            env="BROWSERBASE",
        ) as stagehand:
            page = stagehand.page

            # Navigate to Google and search
            page.goto("https://www.google.com")

            stagehand.act(f'search for "{search_query}"')

            # Wait for results
            page.wait_for_timeout(2000)

            # Extract information from search results
            result = stagehand.extract({
                "instruction": (
                    f"Look at the search results. I'm trying to identify an object that "
                    f"a camera detected. The detection system guessed it might be a '{yolo_hint}'. "
                    f"Extract the most likely identity of this object. "
                    f"Return a JSON object with: "
                    f"'label' (the object name, 1-3 words), "
                    f"'description' (one sentence about what it is), "
                    f"'facts' (array of 2-3 interesting facts), "
                    f"'confidence' (0.0 to 1.0, how sure you are this is correct)"
                ),
                "schema": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string"},
                        "description": {"type": "string"},
                        "facts": {"type": "array", "items": {"type": "string"}},
                        "confidence": {"type": "number"},
                    },
                },
            })

            if result:
                result["source"] = "browserbase_search"
                return result

    except ImportError:
        print("Stagehand not installed. Install with: pip install stagehand")
    except Exception as e:
        print(f"Browserbase research failed: {e}")

    # Fallback
    if yolo_hint:
        return {
            "label": yolo_hint,
            "confidence": 0.3,
            "description": f"Possibly a {yolo_hint} (web research failed, using detection hint)",
            "source": "yolo_fallback",
            "facts": [],
        }

    return None
