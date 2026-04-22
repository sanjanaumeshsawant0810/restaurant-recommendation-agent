import json
import math
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import requests

try:
    from google import genai
except ImportError:
    genai = None


PLACES_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
PLACES_TEXT_SEARCH_URL = "https://places.googleapis.com/v1/places:searchText"
PLACES_DETAILS_URL = "https://places.googleapis.com/v1/places/{place_id}"
NYC_BOUNDS = {
    "lat_min": 40.45,
    "lat_max": 40.95,
    "lng_min": -74.30,
    "lng_max": -73.65,
}


CUISINE_KEYWORDS = {
    "indian": ["indian", "south indian", "north indian"],
    "italian": ["italian"],
    "chinese": ["chinese"],
    "thai": ["thai"],
    "korean": ["korean"],
    "japanese": ["japanese", "sushi"],
    "mexican": ["mexican"],
    "american": ["american", "burger", "steakhouse"],
    "coffee": ["coffee", "cafe", "espresso"],
}

MEAL_TYPE_KEYWORDS = {
    "snack": ["snack", "light", "small bite", "not that hungry"],
    "meal": ["meal", "full meal", "lunch", "dinner", "hungry", "very hungry"],
    "dessert": ["dessert", "sweet", "sweets", "ice cream"],
    "coffee": ["coffee", "cafe", "espresso"],
}

PLACE_TYPE_KEYWORDS = {
    "restaurant": ["restaurant", "dine in", "sit down"],
    "cafe": ["cafe", "coffee shop"],
    "dessert_shop": ["dessert shop", "ice cream place", "bakery"],
    "takeout": ["to go", "takeout", "pickup"],
    "luxury": ["luxury", "fancy", "fine dining", "upscale"],
}

AMBIENCE_KEYWORDS = {
    "romantic": ["romantic", "date", "cozy", "intimate"],
    "italian": ["italian ambience", "italian vibe"],
    "casual": ["casual", "relax", "laid back"],
    "family": ["family", "kids", "good for kids"],
    "trendy": ["trendy", "modern", "stylish"],
    "quiet": ["quiet", "peaceful"],
}


def normalize_text(text: str) -> str:
    text = str(text or "").strip().lower()
    text = re.sub(r"[^a-z0-9\s.,-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def gemini_enabled() -> bool:
    return bool(GEMINI_API_KEY and genai is not None)


def _gemini_client():
    if not gemini_enabled():
        return None
    return genai.Client(api_key=GEMINI_API_KEY)


def extract_json_object(text: str) -> Dict[str, Any]:
    cleaned = str(text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"Could not find JSON object in Gemini response: {cleaned}")
    return json.loads(cleaned[start : end + 1])


APP_STATE_SYSTEM_PROMPT = """
You are a restaurant recommendation conversation planner for a web app.

Return JSON only.

Use the current state and the latest user message to update this exact shape:
{
  "when": null,
  "cuisine": null,
  "dish": null,
  "location_mode": null,
  "manual_location": null,
  "travel_mode": null,
  "min_travel_minutes": null,
  "travel_minutes": null,
  "max_price_level": null,
  "min_rating": null,
  "next_question": null
}

Rules:
- The latest user message can override earlier preferences if it clearly changes the request.
- Time means whether the user wants to eat now or later.
- Infer cuisine from strong dishes when obvious, like pizza -> italian, dosa -> south indian, sushi -> japanese, tacos -> mexican, coffee -> coffee.
- Only ask for cuisine when the food request is vague, such as breakfast, lunch, dinner, snack, dessert, sweets, or food.
- If the user mentions a concrete place name or neighborhood, store it in manual_location and set location_mode to manual.
- If the user says use my location, current location, or my location, set location_mode to current.
- If the user gives travel time, capture travel_minutes and travel_mode when stated.
- If the user gives a range like 5 to 10 minutes, store min_travel_minutes=5 and travel_minutes=10.
- If the user gives multiple acceptable travel modes, preserve that flexibility instead of collapsing it to just one mode.
- If the user gives a budget or price range like $, $$, $$$, $$$$, cheap, moderate, or expensive, capture max_price_level as 1 to 4.
- Time is optional. Do not ask for it unless the user is clearly asking for time-sensitive recommendations and that detail is missing.
- Ask only one short natural next question at a time.
- Leave fields null when the value is unknown.
"""


FINAL_RESPONSE_SYSTEM_PROMPT = """
You are a warm restaurant recommendation assistant.

Write a concise recommendation list using only the provided data.
- Keep it scannable.
- Use plain text only, not Markdown.
- Do not use asterisk bullets or bold markers.
- Prefer variety across recommendations and avoid repeating the same brand name unless there are not enough strong alternatives.
- Write exactly the requested number of recommendations when that many results are provided.
- Mention every place in the provided results list once, in the same order.
- Use numbered items like 1., 2., 3.
- For every recommendation, explicitly say whether it fits the user's criteria well or does not fully fit.
- Mention distance, travel time, rating, and open-now status when available.
- If a place misses a user preference, mention that briefly.
- If a place is still good but within 10 minutes beyond the user's travel preference, say that clearly instead of rejecting it harshly.
- If the user asked for more results, keep both strong matches and weaker matches, but clearly label the weaker ones.
- Do not invent facts.
"""


def split_terms(text: str) -> List[str]:
    cleaned = normalize_text(text)
    if not cleaned:
        return []
    return [part.strip() for part in re.split(r",|/|\band\b", cleaned) if part.strip()]


def extract_min_rating(text: str) -> Optional[float]:
    cleaned = normalize_text(text)
    match = re.search(r"(?:at least|min(?:imum)?|least)\s*(\d(?:\.\d)?)", cleaned)
    if match:
        return float(match.group(1))
    if re.fullmatch(r"\d(?:\.\d)?", cleaned):
        value = float(cleaned)
        if 1.0 <= value <= 5.0:
            return value
    match = re.search(r"\b([1-5](?:\.\d)?)\b", cleaned)
    if match and any(token in cleaned for token in ["rating", "rated", "stars", "star"]):
        return float(match.group(1))
    return None


def keyword_lookup(text: str, keyword_map: Dict[str, List[str]]) -> Optional[str]:
    cleaned = normalize_text(text)
    for canonical, keywords in keyword_map.items():
        if any(keyword in cleaned for keyword in keywords):
            return canonical
    return None


def infer_dish(text: str) -> Optional[str]:
    cleaned = normalize_text(text)
    dish_patterns = [
        r"crave (.+)",
        r"want to eat (.+)",
        r"want (.+)",
        r"eat (.+)",
    ]
    for pattern in dish_patterns:
        match = re.search(pattern, cleaned)
        if match:
            value = match.group(1).strip(" .")
            if value and len(value) < 80:
                return value
    return None


def is_in_nyc(lat: float, lng: float) -> bool:
    return NYC_BOUNDS["lat_min"] <= lat <= NYC_BOUNDS["lat_max"] and NYC_BOUNDS["lng_min"] <= lng <= NYC_BOUNDS["lng_max"]


def haversine_miles(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    radius_miles = 3958.8
    d_lat = math.radians(lat2 - lat1)
    d_lng = math.radians(lng2 - lng1)
    a = (
        math.sin(d_lat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(d_lng / 2) ** 2
    )
    return 2 * radius_miles * math.asin(math.sqrt(a))


def places_text_search(text_query: str, field_mask: str, max_result_count: int = 10, location_bias: Optional[dict] = None) -> List[dict]:
    if not PLACES_API_KEY:
        raise ValueError("Set GOOGLE_MAPS_API_KEY before running this script.")

    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": PLACES_API_KEY,
        "X-Goog-FieldMask": field_mask,
    }
    payload = {
        "textQuery": text_query,
        "maxResultCount": max_result_count,
    }
    if location_bias:
        payload["locationBias"] = location_bias

    response = requests.post(PLACES_TEXT_SEARCH_URL, headers=headers, json=payload, timeout=30)
    response.raise_for_status()
    return response.json().get("places", [])


def place_details(place_id: str, field_mask: str) -> dict:
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": PLACES_API_KEY,
        "X-Goog-FieldMask": field_mask,
    }
    response = requests.get(PLACES_DETAILS_URL.format(place_id=place_id), headers=headers, timeout=30)
    response.raise_for_status()
    return response.json()


def geocode_location(location_text: str) -> dict:
    places = places_text_search(
        text_query=location_text,
        field_mask="places.id,places.displayName,places.formattedAddress,places.location",
        max_result_count=1,
    )
    if not places:
        raise ValueError(f"Could not resolve location: {location_text}")

    place = places[0]
    lat = place["location"]["latitude"]
    lng = place["location"]["longitude"]
    return {
        "label": place.get("displayName", {}).get("text", location_text),
        "address": place.get("formattedAddress", location_text),
        "lat": lat,
        "lng": lng,
    }


def analyze_app_turn_with_gemini(state: Dict[str, Any], message: str) -> Optional[Dict[str, Any]]:
    client = _gemini_client()
    if client is None:
        return None

    prompt = (
        APP_STATE_SYSTEM_PROMPT
        + "\nCurrent state:\n"
        + json.dumps(
            {
                "when": state.get("when"),
                "cuisine": state.get("cuisine"),
                "dish": state.get("dish"),
                "location_mode": state.get("location_mode"),
                "manual_location": state.get("manual_location"),
                "travel_mode": state.get("travel_mode"),
                "travel_minutes": state.get("travel_minutes"),
                "min_rating": state.get("min_rating"),
            },
            ensure_ascii=False,
        )
        + "\nLatest user message:\n"
        + json.dumps(message, ensure_ascii=False)
    )
    response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    return extract_json_object(response.text)


def build_final_response_with_gemini(state: Dict[str, Any], results: List[Dict[str, Any]], limit: int) -> Optional[str]:
    client = _gemini_client()
    if client is None:
        return None

    payload = {
        "state": {
            "when": state.get("when"),
            "cuisine": state.get("cuisine"),
            "dish": state.get("dish"),
            "location_mode": state.get("location_mode"),
            "manual_location": state.get("manual_location"),
            "travel_mode": state.get("travel_mode"),
            "travel_minutes": state.get("travel_minutes"),
            "min_rating": state.get("min_rating"),
        },
        "results": results[:limit],
        "limit": limit,
    }
    prompt = (
        FINAL_RESPONSE_SYSTEM_PROMPT
        + "\nRecommendation payload:\n"
        + json.dumps(payload, ensure_ascii=False)
    )
    response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    return str(response.text or "").strip() or None


def get_browser_location_colab() -> dict:
    try:
        from google.colab import output
    except ImportError as exc:
        raise RuntimeError("Current browser location works only in Google Colab.") from exc

    js = """
    async function getLocation() {
      return await new Promise((resolve, reject) => {
        navigator.geolocation.getCurrentPosition(
          (position) => resolve({lat: position.coords.latitude, lng: position.coords.longitude}),
          (error) => reject(error.message),
          {enableHighAccuracy: true, timeout: 10000}
        );
      });
    }
    """
    coords = output.eval_js(js + "getLocation()")
    lat = float(coords["lat"])
    lng = float(coords["lng"])
    return {"label": "Current location", "address": "Current browser location", "lat": lat, "lng": lng}


@dataclass
class PreferenceState:
    dish: Optional[str] = None
    cuisine: Optional[str] = None
    meal_type: Optional[str] = None
    place_type: Optional[str] = None
    ambience: Optional[str] = None
    min_rating: Optional[float] = None
    hunger_level: Optional[str] = None
    when: Optional[str] = None
    location_mode: Optional[str] = None
    manual_location: Optional[str] = None
    user_location: Optional[dict] = None
    confirmations: List[str] = field(default_factory=list)


class RestaurantRecommenderBot:
    def __init__(self) -> None:
        self.state = PreferenceState()

    def update_from_message(self, message: str) -> None:
        cleaned = normalize_text(message)

        if not self.state.min_rating:
            self.state.min_rating = extract_min_rating(message)

        if not self.state.cuisine:
            self.state.cuisine = keyword_lookup(message, CUISINE_KEYWORDS)

        if not self.state.meal_type:
            self.state.meal_type = keyword_lookup(message, MEAL_TYPE_KEYWORDS)

        if not self.state.place_type:
            self.state.place_type = keyword_lookup(message, PLACE_TYPE_KEYWORDS)

        if not self.state.ambience:
            self.state.ambience = keyword_lookup(message, AMBIENCE_KEYWORDS)

        if not self.state.hunger_level:
            if "very hungry" in cleaned or "hungry right now" in cleaned:
                self.state.hunger_level = "very hungry"
            elif "not that hungry" in cleaned or "light" in cleaned:
                self.state.hunger_level = "light hunger"
            elif "hungry" in cleaned:
                self.state.hunger_level = "hungry"

        if not self.state.when:
            if "right now" in cleaned or "now" in cleaned:
                self.state.when = "now"
            elif "later" in cleaned or "tomorrow" in cleaned:
                self.state.when = "later"

        if not self.state.location_mode:
            if "current location" in cleaned or "use my location" in cleaned or "access my location" in cleaned:
                self.state.location_mode = "current"
            elif "manual" in cleaned or "enter location" in cleaned:
                self.state.location_mode = "manual"

        if self.state.location_mode == "manual" and not self.state.manual_location:
            if len(cleaned) > 2 and all(token not in cleaned for token in ["manual", "location", "enter"]):
                self.state.manual_location = message.strip()

        if not self.state.dish:
            self.state.dish = infer_dish(message)

    def next_question(self) -> Optional[str]:
        if not self.state.hunger_level and not self.state.meal_type:
            return "Are you hungry right now, planning a date, or just looking to relax?"
        if not self.state.location_mode and not self.state.user_location:
            return "Can I use your current location to recommend places nearby, or do you want to enter the location manually?"
        if self.state.location_mode == "manual" and not self.state.manual_location:
            return "Please enter the location you want me to search near, like Central Park or Times Square."
        if not self.state.cuisine and self.state.dish:
            return f"Just to confirm, what cuisine should I focus on for {self.state.dish}?"
        if not self.state.meal_type:
            return "Is this more of a snack, a full meal, dessert, or coffee?"
        if not self.state.place_type:
            return "What type of place do you want: restaurant, cafe, dessert shop, takeout spot, or something more upscale?"
        if not self.state.min_rating:
            return "What is the minimum Google rating you want, like 4.0 or 4.5?"
        if not self.state.ambience:
            return "What ambience are you in the mood for, like romantic, casual, cozy, trendy, quiet, or family-friendly?"
        return None

    def resolve_location(self) -> None:
        if self.state.user_location:
            return
        if self.state.location_mode == "current":
            self.state.user_location = get_browser_location_colab()
        elif self.state.location_mode == "manual" and self.state.manual_location:
            self.state.user_location = geocode_location(self.state.manual_location)
        else:
            raise ValueError("Location is still missing.")

    def build_search_query(self) -> str:
        parts = []
        if self.state.dish:
            parts.append(self.state.dish)
        if self.state.cuisine and self.state.cuisine not in " ".join(parts):
            parts.append(self.state.cuisine)

        if self.state.place_type == "cafe":
            parts.append("cafe")
        elif self.state.place_type == "dessert_shop":
            parts.append("dessert shop")
        else:
            parts.append("restaurant")

        if self.state.ambience:
            parts.append(self.state.ambience)

        return " ".join(parts).strip()

    def fetch_recommendations(self, limit: int = 5) -> List[dict]:
        self.resolve_location()
        location_bias = {
            "circle": {
                "center": {
                    "latitude": self.state.user_location["lat"],
                    "longitude": self.state.user_location["lng"],
                },
                "radius": 6000.0,
            }
        }

        places = places_text_search(
            text_query=self.build_search_query(),
            field_mask=(
                "places.id,places.displayName,places.formattedAddress,places.location,"
                "places.rating,places.userRatingCount,places.priceLevel,places.primaryTypeDisplayName"
            ),
            max_result_count=10,
            location_bias=location_bias,
        )

        scored = []
        for place in places:
            rating = place.get("rating")
            if self.state.min_rating and (rating is None or rating < self.state.min_rating):
                continue

            details = place_details(
                place["id"],
                (
                    "id,displayName,formattedAddress,location,rating,userRatingCount,priceLevel,"
                    "primaryTypeDisplayName,regularOpeningHours,currentOpeningHours,"
                    "reviewSummary,editorialSummary,takeout,delivery,dineIn,servesCoffee,"
                    "servesDessert,servesLunch,servesDinner,websiteUri,googleMapsUri"
                ),
            )
            location = details.get("location", {})
            distance_miles = None
            if location.get("latitude") is not None and location.get("longitude") is not None:
                distance_miles = round(
                    haversine_miles(
                        self.state.user_location["lat"],
                        self.state.user_location["lng"],
                        location["latitude"],
                        location["longitude"],
                    ),
                    2,
                )

            score = float(details.get("rating", 0)) * 2.0
            score += min(details.get("userRatingCount", 0), 3000) / 1000
            if distance_miles is not None:
                score += max(0.0, 4.0 - min(distance_miles, 4.0))
            if self.state.place_type == "cafe" and details.get("servesCoffee"):
                score += 1.0
            if self.state.meal_type == "dessert" and details.get("servesDessert"):
                score += 1.0
            if self.state.place_type == "takeout" and details.get("takeout"):
                score += 1.0

            scored.append(
                {
                    "name": details.get("displayName", {}).get("text"),
                    "address": details.get("formattedAddress"),
                    "rating": details.get("rating"),
                    "user_rating_count": details.get("userRatingCount"),
                    "distance_miles": distance_miles,
                    "price_level": details.get("priceLevel"),
                    "primary_type": details.get("primaryTypeDisplayName", {}).get("text"),
                    "review_summary": details.get("reviewSummary", {}).get("text"),
                    "editorial_summary": details.get("editorialSummary", {}).get("text"),
                    "open_now": details.get("currentOpeningHours", {}).get("openNow"),
                    "takeout": details.get("takeout"),
                    "delivery": details.get("delivery"),
                    "dine_in": details.get("dineIn"),
                    "google_maps_url": details.get("googleMapsUri"),
                    "score": round(score, 2),
                }
            )

        return sorted(scored, key=lambda item: item["score"], reverse=True)[:limit]

    def recommendation_paragraph(self, results: List[dict]) -> str:
        if not results:
            return (
                "I could not find a strong live match with your current filters, so try lowering the minimum rating, "
                "broadening the cuisine, or choosing a larger search area."
            )

        intro_bits = []
        if self.state.dish:
            intro_bits.append(self.state.dish)
        elif self.state.cuisine:
            intro_bits.append(self.state.cuisine)
        if self.state.ambience:
            intro_bits.append(f"{self.state.ambience} ambience")
        intro = ", ".join(intro_bits) if intro_bits else "your preferences"

        sentences = [
            f"Based on {intro} near {self.state.user_location['label']}, these are the strongest live matches I found."
        ]

        for place in results[:3]:
            reasons = []
            if place["rating"] is not None:
                reasons.append(f"rated {place['rating']}")
            if place["distance_miles"] is not None:
                reasons.append(f"about {place['distance_miles']} miles away")
            if place["primary_type"]:
                reasons.append(place["primary_type"].lower())
            if place["open_now"] is True:
                reasons.append("currently open")

            summary = place["review_summary"] or place["editorial_summary"]
            reason_text = ", ".join(reasons)
            sentence = f"{place['name']} at {place['address']} stands out because it is {reason_text}."
            if summary:
                sentence += f" Review summary: {summary}"
            if place["google_maps_url"]:
                sentence += f" Maps: {place['google_maps_url']}"
            sentences.append(sentence)

        sentences.append("These suggestions are ranked from live Places data using rating, review volume, and distance.")
        return " ".join(sentences)

    def handle_message(self, message: str) -> str:
        self.update_from_message(message)

        if self.state.location_mode == "manual" and self.state.manual_location and not self.state.user_location:
            self.resolve_location()

        question = self.next_question()
        if question:
            return question

        results = self.fetch_recommendations(limit=5)
        return self.recommendation_paragraph(results)


def run_cli_chat() -> None:
    bot = RestaurantRecommenderBot()
    print("Restaurant recommender is ready. Tell me what you want to eat.")
    while True:
        user_message = input("You: ").strip()
        if user_message.lower() in {"quit", "exit"}:
            print("Bot: See you next time.")
            break
        try:
            reply = bot.handle_message(user_message)
        except Exception as exc:
            reply = f"Sorry, I hit an error: {exc}"
        print(f"Bot: {reply}")


if __name__ == "__main__":
    run_cli_chat()
