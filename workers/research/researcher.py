"""Vision + Browserbase research worker.

When an unknown object is detected, this worker:
1. Uses vision APIs (GPT-4o / Claude) to identify the object from its thumbnail
2. Dispatches Browserbase/Stagehand to deep-research the identified object:
   - Product details (manufacturer, model, specs)
   - Pricing and availability
   - Safety/hazard information
3. Returns an enriched "Object Card" with all gathered data

Requires:
  - ANTHROPIC_API_KEY / MODEL_API_KEY (for vision + Stagehand page interpretation)
  - BROWSERBASE_API_KEY, BROWSERBASE_PROJECT_ID (for web research)
"""
import os
import json
import base64
import asyncio
import re
import time
import threading
import weave
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

BROWSERBASE_API_KEY = os.getenv("BROWSERBASE_API_KEY", "")
BROWSERBASE_PROJECT_ID = os.getenv("BROWSERBASE_PROJECT_ID", "")
MODEL_API_KEY = os.getenv("MODEL_API_KEY") or os.getenv("OPENAI_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Browserbase rate limiter: max 4 sessions per 60s (plan limit is 5/min)
_BB_MAX_PER_WINDOW = 4
_BB_WINDOW_SECONDS = 65  # slightly over 1 min for safety
_bb_timestamps: list[float] = []
_bb_lock = threading.Lock()


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


async def _deep_research_via_browserbase(label: str) -> Optional[dict]:
    """Use Browserbase/Stagehand to deep-research an identified object.

    After vision API identifies an object (e.g., "Boston Dynamics Spot"),
    this searches the web to gather:
    - Product details (manufacturer, model, specs, dimensions)
    - Pricing and availability
    - Safety/hazard information

    Returns enriched data dict or None if Browserbase unavailable.
    """
    if not BROWSERBASE_API_KEY or not BROWSERBASE_PROJECT_ID:
        print("Browserbase not configured, skipping deep research")
        return None

    from stagehand import AsyncStagehand

    client = AsyncStagehand(
        browserbase_api_key=BROWSERBASE_API_KEY,
        browserbase_project_id=BROWSERBASE_PROJECT_ID,
        model_api_key=MODEL_API_KEY,
    )

    session = None
    enrichment = {
        "product_url": None,
        "manufacturer": None,
        "price": None,
        "specs": [],
        "safety_info": None,
        "web_description": None,
        "search_sources": [],
    }

    try:
        session = await client.sessions.start(model_name="anthropic/claude-sonnet-4-5-20250929")
        print(f"Browserbase session started: {session.id}")

        # Step 1: Search for product details
        search_query = f"{label} product specifications price"
        await session.navigate(url=f"https://www.google.com/search?q={search_query.replace(' ', '+')}")
        print(f"  Searching: {search_query}")

        product_data = await session.extract(
            instruction=f"""Extract information about "{label}" from the search results.
Find the most relevant product listing or information page.
Extract what you can see on this search results page.""",
            schema={
                "type": "object",
                "properties": {
                    "product_name": {"type": "string", "description": "Full product name"},
                    "manufacturer": {"type": "string", "description": "Company that makes it"},
                    "description": {"type": "string", "description": "Brief product description from search snippets"},
                    "price_range": {"type": "string", "description": "Price or price range if visible"},
                    "top_result_url": {"type": "string", "description": "URL of the most relevant result"},
                },
                "required": ["product_name"],
            },
        )

        if product_data and product_data.data and product_data.data.result:
            data = product_data.data.result
            if isinstance(data, dict):
                enrichment["manufacturer"] = data.get("manufacturer")
                enrichment["price"] = data.get("price_range")
                enrichment["web_description"] = data.get("description")
                top_url = data.get("top_result_url")
                if top_url:
                    enrichment["product_url"] = top_url
                    enrichment["search_sources"].append(top_url)
                print(f"  Found: {data.get('product_name', '?')} by {data.get('manufacturer', '?')}")

        # Step 2: Navigate to top result for detailed specs
        if enrichment.get("product_url"):
            try:
                await session.navigate(url=enrichment["product_url"])
                print(f"  Visiting: {enrichment['product_url']}")

                specs_data = await session.extract(
                    instruction=f"""Extract detailed product specifications for "{label}".
Look for: dimensions, weight, technical specs, features, pricing.""",
                    schema={
                        "type": "object",
                        "properties": {
                            "specs": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of key specifications (e.g., 'Weight: 32 kg', 'Battery: 5 hours')",
                            },
                            "price": {"type": "string", "description": "Price if found on this page"},
                            "detailed_description": {"type": "string", "description": "Detailed product description"},
                        },
                    },
                )

                if specs_data and specs_data.data and specs_data.data.result:
                    sdata = specs_data.data.result
                    if isinstance(sdata, dict):
                        if sdata.get("specs"):
                            enrichment["specs"] = sdata["specs"]
                        if sdata.get("price") and not enrichment.get("price"):
                            enrichment["price"] = sdata["price"]
                        if sdata.get("detailed_description"):
                            enrichment["web_description"] = sdata["detailed_description"]
                        print(f"  Specs: {len(enrichment['specs'])} items found")
            except Exception as e:
                print(f"  Specs extraction failed: {e}")

        # Step 3: Search for safety/hazard information
        safety_query = f"{label} safety hazard warnings precautions"
        await session.navigate(url=f"https://www.google.com/search?q={safety_query.replace(' ', '+')}")
        print(f"  Safety search: {safety_query}")

        safety_data = await session.extract(
            instruction=f"""Extract any safety information, hazard warnings, or precautions
related to "{label}" from these search results. Include regulatory info if available.""",
            schema={
                "type": "object",
                "properties": {
                    "safety_summary": {
                        "type": "string",
                        "description": "Summary of safety concerns, hazards, or precautions",
                    },
                    "hazard_level": {
                        "type": "string",
                        "description": "General hazard level: 'none', 'low', 'moderate', 'high'",
                    },
                    "warnings": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Specific safety warnings or precautions",
                    },
                },
            },
        )

        if safety_data and safety_data.data and safety_data.data.result:
            sdata = safety_data.data.result
            if isinstance(sdata, dict):
                parts = []
                if sdata.get("hazard_level"):
                    parts.append(f"Hazard level: {sdata['hazard_level']}")
                if sdata.get("safety_summary"):
                    parts.append(sdata["safety_summary"])
                if sdata.get("warnings"):
                    parts.extend(sdata["warnings"])
                if parts:
                    enrichment["safety_info"] = " | ".join(parts)
                    print(f"  Safety: {sdata.get('hazard_level', '?')} hazard level")

        print(f"  Deep research complete.")
        return enrichment

    except Exception as e:
        print(f"Browserbase deep research failed: {e}")
        return enrichment if any(v for v in enrichment.values() if v) else None

    finally:
        if session:
            try:
                await session.end()
            except Exception:
                pass


def _bb_rate_limit_wait():
    """Block until we have capacity under the Browserbase burst rate limit."""
    with _bb_lock:
        now = time.time()
        # Prune timestamps older than the window
        cutoff = now - _BB_WINDOW_SECONDS
        while _bb_timestamps and _bb_timestamps[0] < cutoff:
            _bb_timestamps.pop(0)

        if len(_bb_timestamps) >= _BB_MAX_PER_WINDOW:
            # Wait until the oldest timestamp expires
            wait_time = _bb_timestamps[0] + _BB_WINDOW_SECONDS - now + 1
            print(f"  Browserbase rate limit: waiting {wait_time:.0f}s "
                  f"({len(_bb_timestamps)}/{_BB_MAX_PER_WINDOW} sessions in window)")
            return wait_time
    return 0


def _run_deep_research(label: str) -> Optional[dict]:
    """Sync wrapper for async Browserbase research with rate limiting."""
    # Wait for rate limit capacity
    wait = _bb_rate_limit_wait()
    if wait > 0:
        time.sleep(wait)

    # Record this session timestamp
    with _bb_lock:
        _bb_timestamps.append(time.time())

    try:
        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(_deep_research_via_browserbase(label))
        loop.close()
        return result
    except Exception as e:
        print(f"Deep research wrapper failed: {e}")
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
    """Research an unknown object using vision APIs + Browserbase deep research.

    Pipeline:
    1. Vision APIs (GPT-4o / Claude) identify the object from its thumbnail
    2. Browserbase/Stagehand deep-researches the identified object:
       - Product details, specs, manufacturer
       - Pricing and availability
       - Safety/hazard information
    3. Returns enriched result with all gathered data

    Args:
        thumbnail_b64: Base64 JPEG of the object crop
        yolo_hint: YOLO's best guess class name (may be wrong)
        yolo_confidence: Original YOLO confidence (0-1)

    Returns:
        dict with keys: label, confidence, description, source, facts[],
        plus enrichment fields: product_url, manufacturer, price, specs[],
        safety_info, web_description, search_sources[]
    """
    result = None

    # Stage 1: Vision API identification
    if thumbnail_b64:
        try:
            result = _research_via_image_search(thumbnail_b64, yolo_hint)
            if result:
                print(f"Vision API identified: {result.get('label')} "
                      f"(conf={result.get('confidence')}, source={result.get('source')})")
        except Exception as e:
            print(f"Vision API research failed: {e}")

    if not result:
        return None

    # Stage 2: Browserbase deep research (enrichment)
    label = result.get("label", "")
    if label and BROWSERBASE_API_KEY:
        try:
            print(f"Starting Browserbase deep research for: {label}")
            enrichment = _run_deep_research(label)
            if enrichment:
                result["product_url"] = enrichment.get("product_url")
                result["manufacturer"] = enrichment.get("manufacturer")
                result["price"] = enrichment.get("price")
                result["specs"] = enrichment.get("specs", [])
                result["safety_info"] = enrichment.get("safety_info")
                result["web_description"] = enrichment.get("web_description")
                result["search_sources"] = enrichment.get("search_sources", [])
                result["source"] = result.get("source", "") + "+browserbase"
                print(f"Deep research enriched: manufacturer={enrichment.get('manufacturer')}, "
                      f"price={enrichment.get('price')}, "
                      f"specs={len(enrichment.get('specs', []))} items, "
                      f"safety={'yes' if enrichment.get('safety_info') else 'no'}")
        except Exception as e:
            print(f"Browserbase deep research failed: {e}")

    return result
