import os
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from threading import Lock
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from flask import Flask, jsonify, render_template, request

from places_restaurant_chatbot import (
    analyze_app_turn_with_gemini,
    build_final_response_with_gemini,
    gemini_enabled,
    geocode_location,
    haversine_miles,
    normalize_text,
    place_details,
    places_text_search,
    verify_dish_availability,
)


app = Flask(__name__)

TRAVEL_TIME_LEEWAY_MINUTES = 10
APP_TIMEZONE = ZoneInfo("America/New_York")


TOP_K_PATTERNS = {
    10: [
        r"\btop\s*10\b",
        r"\btop ten\b",
        r"\bten recommendations\b",
        r"\bshow me more\b",
        r"\bmore recommendations\b",
    ],
}

DISH_TO_CUISINE = {
    "pizza": "italian",
    "margherita pizza": "italian",
    "chicago pizza": "italian",
    "hawaiian pizza": "italian",
    "pasta": "italian",
    "dosa": "south indian",
    "idli": "south indian",
    "vada": "south indian",
    "biryani": "indian",
    "sushi": "japanese",
    "ramen": "japanese",
    "taco": "mexican",
    "tacos": "mexican",
    "burrito": "mexican",
    "coffee": "coffee",
}

VAGUE_FOOD_INTENTS = {
    "breakfast",
    "brunch",
    "lunch",
    "dinner",
    "snack",
    "dessert",
    "sweets",
    "ice cream",
    "food",
    "eat",
}


@dataclass
class ConversationState:
    when: Optional[str] = None
    cuisine: Optional[str] = None
    dish: Optional[str] = None
    location_mode: Optional[str] = None
    manual_location: Optional[str] = None
    user_location: Optional[dict] = None
    travel_mode: Optional[str] = None
    min_travel_minutes: Optional[int] = None
    travel_minutes: Optional[int] = None
    search_radius_meters: Optional[int] = None
    max_price_level: Optional[int] = None
    min_rating: Optional[float] = None
    requested_day_offset: Optional[int] = None
    requested_hour: Optional[int] = None
    requested_minute: Optional[int] = None
    llm_next_question: Optional[str] = None
    last_results: List[dict] = field(default_factory=list)
    last_limit: int = 5
    conversation_turns: int = 0


@dataclass
class AgentTrace:
    agent: str
    action: str
    details: str


SESSIONS: Dict[str, ConversationState] = {}
SESSIONS_LOCK = Lock()


def get_state(session_id: str) -> ConversationState:
    with SESSIONS_LOCK:
        return SESSIONS.setdefault(session_id, ConversationState())


def parse_top_k_request(message: str) -> Optional[int]:
    cleaned = normalize_text(message)
    for top_k, patterns in TOP_K_PATTERNS.items():
        if any(re.search(pattern, cleaned) for pattern in patterns):
            return top_k
    return None


def parse_min_rating(message: str) -> Optional[float]:
    cleaned = normalize_text(message)
    match = re.search(r"(?:at least|min(?:imum)?|rating)\s*(\d(?:\.\d)?)", cleaned)
    if match:
        return float(match.group(1))
    if re.fullmatch(r"\d(?:\.\d)?", cleaned):
        value = float(cleaned)
        if 1.0 <= value <= 5.0:
            return value
    return None


def parse_price_preference(message: str) -> Optional[int]:
    cleaned = normalize_text(message)
    if "$$$$" in message:
        return 4
    if "$$$" in message:
        return 3
    if "$$" in message:
        return 2
    if "$" in message:
        return 1

    patterns = [
        (1, ["cheap", "budget", "affordable", "inexpensive", "low cost"]),
        (2, ["moderate", "mid range", "midrange", "reasonable", "not too expensive"]),
        (3, ["expensive", "upscale", "fancy"]),
        (4, ["very expensive", "luxury", "high end", "fine dining"]),
    ]
    for level, keywords in patterns:
        if any(keyword in cleaned for keyword in keywords):
            return level

    match = re.search(r"(?:price|budget|spend|cost)\s*(?:level)?\s*(\d)", cleaned)
    if match:
        level = int(match.group(1))
        if 1 <= level <= 4:
            return level
    return None


def format_price_level(price_level: Optional[int]) -> Optional[str]:
    if price_level is None:
        return None
    return "$" * max(1, min(price_level, 4))


def infer_when(message: str) -> Optional[str]:
    cleaned = normalize_text(message)
    if "right now" in cleaned or cleaned == "now" or "now" in cleaned:
        return "now"
    if "later" in cleaned or "tomorrow" in cleaned:
        return "later"
    if re.search(r"\bin\s+\d+\s*(?:hr|hour|hours)\b", cleaned):
        return "later"
    if re.search(r"\bin\s+\d+\s*(?:min|mins|minute|minutes)\b", cleaned):
        return "later"
    return None


def infer_requested_time_details(message: str) -> Dict[str, Optional[int]]:
    cleaned = normalize_text(message)
    details = {
        "requested_day_offset": None,
        "requested_hour": None,
        "requested_minute": None,
    }

    if "tomorrow" in cleaned:
        details["requested_day_offset"] = 1
    elif any(phrase in cleaned for phrase in ["today", "this afternoon", "this evening", "tonight"]):
        details["requested_day_offset"] = 0

    time_match = re.search(r"\bat\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b", cleaned)
    if time_match:
        hour = int(time_match.group(1))
        minute = int(time_match.group(2) or 0)
        meridiem = time_match.group(3)
        if meridiem == "pm" and hour != 12:
            hour += 12
        if meridiem == "am" and hour == 12:
            hour = 0
        details["requested_hour"] = hour
        details["requested_minute"] = minute
    elif "afternoon" in cleaned:
        details["requested_hour"] = 13
        details["requested_minute"] = 0
    elif "evening" in cleaned or "tonight" in cleaned:
        details["requested_hour"] = 19
        details["requested_minute"] = 0
    elif "lunch" in cleaned:
        details["requested_hour"] = 12
        details["requested_minute"] = 0
    elif "dinner" in cleaned:
        details["requested_hour"] = 19
        details["requested_minute"] = 0

    return details


def infer_location_mode(message: str) -> Optional[str]:
    cleaned = normalize_text(message)
    if any(phrase in cleaned for phrase in ["current location", "use my location", "my location"]):
        return "current"
    if any(phrase in cleaned for phrase in ["manual", "enter location", "type location"]):
        return "manual"
    return None


def infer_manual_location_text(message: str) -> Optional[str]:
    raw = (message or "").strip()
    cleaned = normalize_text(raw)

    coordinate_location = parse_coordinate_pair(raw)
    if coordinate_location:
        return raw

    patterns = [
        r"\bnear\s+(.+?)(?:\s+with\b|\s+and\b|$)",
        r"\bat\s+(.+?)(?:\s+with\b|\s+and\b|$)",
        r"\bin\s+(.+?)(?:\s+with\b|\s+and\b|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, cleaned)
        if match:
            location_text = match.group(1).strip(" ,.")
            if location_text:
                return location_text
    return None


def infer_cuisine_from_dish(dish: Optional[str]) -> Optional[str]:
    cleaned = normalize_text(dish or "")
    if not cleaned:
        return None
    for key, cuisine in sorted(DISH_TO_CUISINE.items(), key=lambda item: len(item[0]), reverse=True):
        if key in cleaned:
            return cuisine
    return None


def is_vague_food_request(dish: Optional[str]) -> bool:
    return normalize_text(dish or "") in VAGUE_FOOD_INTENTS


def infer_dish(message: str) -> Optional[str]:
    cleaned = normalize_text(message)
    candidates = [
        r"i want to eat (.+)",
        r"i want (.+)",
        r"i crave (.+)",
        r"eat (.+)",
    ]
    for pattern in candidates:
        match = re.search(pattern, cleaned)
        if match:
            return match.group(1).strip(" .")
    if len(cleaned.split()) <= 4:
        return cleaned
    return None


def allowed_travel_modes(travel_mode: Optional[str]) -> List[str]:
    mapping = {
        "walk": ["walk"],
        "car": ["car"],
        "transit": ["transit"],
        "walk_or_transit": ["walk", "transit"],
        "walk_or_car": ["walk", "car"],
        "transit_or_car": ["transit", "car"],
        "walk_or_transit_or_car": ["walk", "transit", "car"],
    }
    return mapping.get(travel_mode or "", [])


def travel_mode_label(travel_mode: Optional[str]) -> str:
    labels = {
        "walk": "walk",
        "car": "car",
        "transit": "public transport",
        "walk_or_transit": "walk or public transport",
        "walk_or_car": "walk or car",
        "transit_or_car": "public transport or car",
        "walk_or_transit_or_car": "walk, public transport, or car",
    }
    return labels.get(travel_mode or "", travel_mode or "your allowed travel options")


def format_travel_window(min_minutes: Optional[int], max_minutes: Optional[int]) -> str:
    if min_minutes and max_minutes and min_minutes != max_minutes:
        return f"{min_minutes} to {max_minutes} minutes"
    if max_minutes:
        return f"{max_minutes} minutes"
    if min_minutes:
        return f"{min_minutes} minutes"
    return "an unspecified amount of time"


def infer_travel_preferences(message: str) -> Dict[str, Optional[int]]:
    cleaned = normalize_text(message)
    walk_requested = any(word in cleaned for word in ["walk", "walking", "by foot", "on foot"])
    car_requested = any(word in cleaned for word in ["car", "drive", "driving"])
    transit_requested = any(
        phrase in cleaned for phrase in ["public transport", "public transit", "subway", "train", "bus", "transit"]
    )

    travel_mode = None
    requested_modes = [mode for mode, requested in [("walk", walk_requested), ("transit", transit_requested), ("car", car_requested)] if requested]
    if requested_modes == ["walk"]:
        travel_mode = "walk"
    elif requested_modes == ["transit"]:
        travel_mode = "transit"
    elif requested_modes == ["car"]:
        travel_mode = "car"
    elif requested_modes == ["walk", "transit"]:
        travel_mode = "walk_or_transit"
    elif requested_modes == ["walk", "car"]:
        travel_mode = "walk_or_car"
    elif requested_modes == ["transit", "car"]:
        travel_mode = "transit_or_car"
    elif requested_modes == ["walk", "transit", "car"]:
        travel_mode = "walk_or_transit_or_car"

    min_travel_minutes = None
    travel_minutes = None
    range_match = re.search(r"(\d{1,3})\s*(?:to|-)\s*(\d{1,3})\s*(minute|min|minutes)", cleaned)
    if range_match:
        min_travel_minutes = int(range_match.group(1))
        travel_minutes = int(range_match.group(2))
    else:
        hour_match = re.search(r"(\d{1,2})\s*(hr|hour|hours)", cleaned)
        if hour_match:
            travel_minutes = int(hour_match.group(1)) * 60
    match = re.search(r"(\d{1,3})\s*(minute|min|minutes)", cleaned)
    if match:
        travel_minutes = int(match.group(1))
    elif re.fullmatch(r"\d{1,3}", cleaned):
        travel_minutes = int(cleaned)

    search_radius_meters = None
    if travel_minutes is not None:
        radius_candidates = []
        for mode in allowed_travel_modes(travel_mode):
            if mode == "walk":
                radius_candidates.append(max(500, min(travel_minutes * 80, 5000)))
            elif mode == "car":
                radius_candidates.append(max(1500, min(travel_minutes * 700, 30000)))
            elif mode == "transit":
                radius_candidates.append(max(1500, min(travel_minutes * 400, 20000)))
        if radius_candidates:
            search_radius_meters = max(radius_candidates)
        else:
            search_radius_meters = max(1000, min(travel_minutes * 250, 15000))

    return {
        "travel_mode": travel_mode,
        "min_travel_minutes": min_travel_minutes,
        "travel_minutes": travel_minutes,
        "search_radius_meters": search_radius_meters,
    }


def parse_coordinate_pair(text: str) -> Optional[dict]:
    cleaned = str(text or "").strip()
    decimal_match = re.search(r"(-?\d{1,3}(?:\.\d+)?)\s*,\s*(-?\d{1,3}(?:\.\d+)?)", cleaned)
    if decimal_match:
        lat = float(decimal_match.group(1))
        lng = float(decimal_match.group(2))
        if -90 <= lat <= 90 and -180 <= lng <= 180:
            return {
                "label": f"{lat:.6f}, {lng:.6f}",
                "address": f"Coordinates: {lat:.6f}, {lng:.6f}",
                "lat": lat,
                "lng": lng,
            }
    dms_match = re.search(
        r"(\d{1,3})[^0-9A-Za-z]+(\d{1,2})[^0-9A-Za-z]+(\d{1,2}(?:\.\d+)?)[^0-9A-Za-z]*([NS])\D+"
        r"(\d{1,3})[^0-9A-Za-z]+(\d{1,2})[^0-9A-Za-z]+(\d{1,2}(?:\.\d+)?)[^0-9A-Za-z]*([EW])",
        cleaned,
        re.IGNORECASE,
    )
    if dms_match:
        lat = float(dms_match.group(1)) + float(dms_match.group(2)) / 60 + float(dms_match.group(3)) / 3600
        lng = float(dms_match.group(5)) + float(dms_match.group(6)) / 60 + float(dms_match.group(7)) / 3600
        if dms_match.group(4).upper() == "S":
            lat *= -1
        if dms_match.group(8).upper() == "W":
            lng *= -1
        return {
            "label": f"{lat:.6f}, {lng:.6f}",
            "address": f"Coordinates: {lat:.6f}, {lng:.6f}",
            "lat": lat,
            "lng": lng,
        }
    return None


def estimate_travel_minutes_for_mode(distance_miles: Optional[float], travel_mode: str) -> Optional[int]:
    if distance_miles is None:
        return None
    if travel_mode == "walk":
        return max(1, int(round((distance_miles / 3.0) * 60)))
    if travel_mode == "car":
        return max(1, int(round((distance_miles / 20.0) * 60)))
    if travel_mode == "transit":
        return max(1, int(round((distance_miles / 12.0) * 60)))
    return max(1, int(round((distance_miles / 10.0) * 60)))


def estimate_travel_minutes(distance_miles: Optional[float], travel_mode: Optional[str]) -> Optional[int]:
    estimates = [
        estimate_travel_minutes_for_mode(distance_miles, mode)
        for mode in allowed_travel_modes(travel_mode)
    ]
    estimates = [estimate for estimate in estimates if estimate is not None]
    if estimates:
        return min(estimates)
    if travel_mode:
        return estimate_travel_minutes_for_mode(distance_miles, travel_mode)
    return estimate_travel_minutes_for_mode(distance_miles, "walk")


def build_travel_estimates(distance_miles: Optional[float], travel_mode: Optional[str]) -> Dict[str, int]:
    estimates: Dict[str, int] = {}
    for mode in allowed_travel_modes(travel_mode):
        estimate = estimate_travel_minutes_for_mode(distance_miles, mode)
        if estimate is not None:
            estimates[mode] = estimate
    return estimates


def format_travel_estimates(estimates: Dict[str, int]) -> Optional[str]:
    if not estimates:
        return None

    labels = {
        "walk": "on foot",
        "transit": "by public transport",
        "car": "by car",
    }
    ordered_modes = [mode for mode in ["transit", "walk", "car"] if mode in estimates]
    parts = [f"{estimates[mode]} minutes {labels[mode]}" for mode in ordered_modes]
    if len(parts) == 1:
        return f"about {parts[0]}"
    if len(parts) == 2:
        return f"about {parts[0]} or {parts[1]}"
    return "about " + ", ".join(parts[:-1]) + f", or {parts[-1]}"


def build_search_query(state: ConversationState) -> str:
    parts = []
    if state.dish:
        parts.append(state.dish)
    if state.cuisine and state.cuisine not in " ".join(parts):
        parts.append(state.cuisine)
    parts.append("cafe" if state.cuisine == "coffee" else "restaurant")
    return " ".join(parts).strip()


def canonicalize_place_name(name: Optional[str]) -> str:
    cleaned = normalize_text(name or "")
    cleaned = re.sub(r"\b(co|coffee co|llc|inc|restaurant|cafe)\b", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def build_directions_url(user_location: dict, destination: dict, travel_mode: Optional[str]) -> str:
    allowed_modes = allowed_travel_modes(travel_mode)
    if "transit" in allowed_modes:
        mode = "transit"
    elif "walk" in allowed_modes:
        mode = "walking"
    else:
        mode = "driving"
    origin = f"{user_location['lat']},{user_location['lng']}"
    dest = f"{destination['lat']},{destination['lng']}"
    return f"https://www.google.com/maps/dir/?api=1&origin={origin}&destination={dest}&travelmode={mode}"


def requested_datetime_for_state(state: ConversationState) -> Optional[datetime]:
    if state.requested_hour is None:
        return None
    base = datetime.now(APP_TIMEZONE).replace(second=0, microsecond=0)
    target = base + timedelta(days=state.requested_day_offset or 0)
    return target.replace(
        hour=state.requested_hour,
        minute=state.requested_minute or 0,
    )


def is_place_open_at_requested_time(details: dict, state: ConversationState) -> Optional[bool]:
    target_dt = requested_datetime_for_state(state)
    if target_dt is None:
        return None

    periods = details.get("regularOpeningHours", {}).get("periods") or []
    if not periods:
        return None

    google_day = (target_dt.weekday() + 1) % 7
    target_minutes = target_dt.hour * 60 + target_dt.minute

    for period in periods:
        open_info = period.get("open")
        close_info = period.get("close")
        if not open_info or not close_info:
            continue

        open_day = open_info.get("day")
        close_day = close_info.get("day")
        open_minutes = int(open_info.get("hour", 0)) * 60 + int(open_info.get("minute", 0))
        close_minutes = int(close_info.get("hour", 0)) * 60 + int(close_info.get("minute", 0))

        if open_day == close_day:
            if google_day == open_day and open_minutes <= target_minutes < close_minutes:
                return True
            continue

        if google_day == open_day and target_minutes >= open_minutes:
            return True
        if google_day == close_day and target_minutes < close_minutes:
            return True

    return False


def unmet_criteria_for_place(place: dict, state: ConversationState) -> List[str]:
    unmet = []
    if state.dish and place.get("menu_verification", {}).get("status") == "not_verified":
        unmet.append(f"could not verify {state.dish} from the restaurant website menu")
    if state.max_price_level is not None and place.get("price_level") is not None and int(place["price_level"]) > int(state.max_price_level):
        unmet.append(
            f"price level {format_price_level(place['price_level'])} is above your preferred {format_price_level(state.max_price_level)}"
        )
    if state.min_rating is not None and place.get("rating") is not None and float(place["rating"]) < float(state.min_rating):
        unmet.append(f"rating below your minimum ({state.min_rating})")
    if state.travel_minutes and place.get("estimated_travel_minutes") and place["estimated_travel_minutes"] > state.travel_minutes:
        extra_minutes = place["estimated_travel_minutes"] - state.travel_minutes
        if extra_minutes <= TRAVEL_TIME_LEEWAY_MINUTES:
            unmet.append(
                f"still a good option, but about {extra_minutes} minutes farther than your preferred {format_travel_window(state.min_travel_minutes, state.travel_minutes)}"
            )
        else:
            unmet.append(
                f"estimated travel time is {place['estimated_travel_minutes']} minutes, which is {extra_minutes} minutes above your preferred {format_travel_window(state.min_travel_minutes, state.travel_minutes)}"
            )
    if state.when == "now" and place.get("open_now") is False:
        unmet.append("not open right now")
    if state.when == "later" and place.get("open_at_requested_time") is False:
        target_dt = requested_datetime_for_state(state)
        if target_dt is not None:
            unmet.append(f"not likely to be open around {target_dt.strftime('%-I:%M %p').lower()} on {target_dt.strftime('%A')}")
    return unmet


def verification_status_for_place(place: dict, state: ConversationState) -> str:
    menu_verification = place.get("menu_verification") or {}
    if state.dish:
        if menu_verification.get("verified"):
            return "verified"
        if menu_verification.get("status") == "not_verified":
            return "not_verified"
    if place.get("unmet_criteria"):
        return "likely"
    return "likely"


def confidence_score_for_place(place: dict, state: ConversationState) -> int:
    score = 45
    menu_verification = place.get("menu_verification") or {}

    if state.dish:
        if menu_verification.get("verified"):
            score += 30
        elif menu_verification.get("status") == "not_verified":
            score -= 20
        elif menu_verification.get("status") in {"website_unreachable", "no_website"}:
            score -= 10

    if place.get("open_now") is True:
        score += 10
    elif state.when == "now" and place.get("open_now") is False:
        score -= 20

    if state.when == "later":
        if place.get("open_at_requested_time") is True:
            score += 12
        elif place.get("open_at_requested_time") is False:
            score -= 18

    if state.max_price_level is not None and place.get("price_level") is not None:
        if int(place["price_level"]) <= int(state.max_price_level):
            score += 5
        else:
            score -= 10

    if state.min_rating is not None and place.get("rating") is not None:
        if float(place["rating"]) >= float(state.min_rating):
            score += 6
        else:
            score -= 8

    if state.travel_minutes and place.get("estimated_travel_minutes") is not None:
        if place["estimated_travel_minutes"] <= state.travel_minutes:
            score += 8
        elif place["estimated_travel_minutes"] - state.travel_minutes <= TRAVEL_TIME_LEEWAY_MINUTES:
            score += 2
        else:
            score -= 10

    return max(0, min(100, score))


def confidence_label(score: int) -> str:
    if score >= 80:
        return "high confidence"
    if score >= 60:
        return "medium confidence"
    return "low confidence"


def fit_label_for_place(place: dict) -> str:
    status = place.get("verification_status")
    unmet = place.get("unmet_criteria") or []
    if status == "verified" and not unmet:
        return "fits your criteria well."
    if status == "not_verified":
        return "looks relevant, but does not fully fit your criteria."
    if unmet:
        return "does not fully fit your criteria."
    return "looks like a likely match, but is not fully verified."


class IntentAgent:
    name = "Intent Agent"

    def run(self, state: ConversationState, message: str, browser_location: Optional[dict]) -> List[AgentTrace]:
        traces: List[AgentTrace] = []
        cleaned = normalize_text(message)
        state.llm_next_question = None

        gemini_result = None
        if gemini_enabled():
            try:
                gemini_result = analyze_app_turn_with_gemini(asdict(state), message)
            except Exception as exc:
                traces.append(AgentTrace(self.name, "llm_fallback", f"Gemini analysis failed, using local rules instead: {exc}."))

        if gemini_result:
            for field_name in [
                "when",
                "cuisine",
                "dish",
                "location_mode",
                "manual_location",
                "travel_mode",
                "min_travel_minutes",
                "travel_minutes",
                "max_price_level",
                "min_rating",
            ]:
                value = gemini_result.get(field_name)
                if value not in (None, ""):
                    setattr(state, field_name, value)
            if gemini_result.get("next_question"):
                state.llm_next_question = str(gemini_result["next_question"]).strip()
            traces.append(AgentTrace(self.name, "llm_slot_fill", "Gemini analyzed the turn and updated the conversation state."))

        inferred_when = infer_when(message)
        if inferred_when:
            state.when = inferred_when
            if inferred_when == "now":
                state.requested_day_offset = None
                state.requested_hour = None
                state.requested_minute = None
            traces.append(AgentTrace(self.name, "slot_fill", f"Captured time preference: {inferred_when}."))

        inferred_time_details = infer_requested_time_details(message)
        if any(value is not None for value in inferred_time_details.values()):
            if inferred_time_details["requested_day_offset"] is not None:
                state.requested_day_offset = inferred_time_details["requested_day_offset"]
            if inferred_time_details["requested_hour"] is not None:
                state.requested_hour = inferred_time_details["requested_hour"]
                state.requested_minute = inferred_time_details["requested_minute"] or 0
            traces.append(AgentTrace(self.name, "time_parse", "Updated the requested future time from the latest message."))

        inferred_mode = infer_location_mode(message)
        if inferred_mode:
            state.location_mode = inferred_mode
            if inferred_mode == "manual":
                state.user_location = None
            traces.append(AgentTrace(self.name, "slot_fill", f"Captured location mode: {inferred_mode}."))

        parsed_coords = parse_coordinate_pair(message)
        if parsed_coords:
            state.location_mode = "manual"
            state.manual_location = message.strip()
            state.user_location = parsed_coords
            traces.append(AgentTrace(self.name, "location_parse", "Parsed coordinates directly from the message."))
        elif not state.user_location:
            inferred_manual_location = infer_manual_location_text(message)
            if inferred_manual_location:
                state.location_mode = "manual"
                state.manual_location = inferred_manual_location
                traces.append(
                    AgentTrace(
                        self.name,
                        "location_inference",
                        f"Inferred manual location from the message: {inferred_manual_location}.",
                    )
                )
            elif state.location_mode == "current" and browser_location:
                state.user_location = {
                    "label": "Current location",
                    "address": "Current browser location",
                    "lat": browser_location["lat"],
                    "lng": browser_location["lng"],
                }
                traces.append(AgentTrace(self.name, "location_use", "Used the browser-provided current location."))
            elif state.location_mode == "manual" and not state.manual_location:
                if len(cleaned) > 2 and all(token not in cleaned for token in ["manual", "location", "enter"]):
                    state.manual_location = message.strip()
                    traces.append(AgentTrace(self.name, "location_capture", "Stored manual location text for geocoding."))

        inferred_dish = infer_dish(message)
        if inferred_dish:
            state.dish = inferred_dish
            traces.append(AgentTrace(self.name, "slot_fill", f"Captured food intent: {state.dish}."))

        inferred_cuisine = infer_cuisine_from_dish(state.dish)
        if inferred_cuisine:
            state.cuisine = inferred_cuisine
            traces.append(AgentTrace(self.name, "inference", f"Inferred cuisine from dish: {state.cuisine}."))

        inferred_travel = infer_travel_preferences(message)
        if inferred_travel["travel_mode"]:
            state.travel_mode = inferred_travel["travel_mode"]
        if inferred_travel["min_travel_minutes"]:
            state.min_travel_minutes = inferred_travel["min_travel_minutes"]
        if inferred_travel["travel_minutes"]:
            state.travel_minutes = inferred_travel["travel_minutes"]
        if inferred_travel["search_radius_meters"]:
            state.search_radius_meters = inferred_travel["search_radius_meters"]
        if inferred_travel["travel_minutes"] or inferred_travel["travel_mode"]:
            traces.append(
                AgentTrace(
                    self.name,
                    "slot_fill",
                    f"Captured travel preference: {format_travel_window(state.min_travel_minutes, state.travel_minutes)} by {travel_mode_label(state.travel_mode)}.",
                )
            )

        inferred_rating = parse_min_rating(message)
        if inferred_rating is not None:
            state.min_rating = inferred_rating
            traces.append(AgentTrace(self.name, "slot_fill", f"Captured minimum rating: {inferred_rating}."))

        inferred_price_level = parse_price_preference(message)
        if inferred_price_level is not None:
            state.max_price_level = inferred_price_level
            traces.append(
                AgentTrace(
                    self.name,
                    "slot_fill",
                    f"Captured price preference: up to {format_price_level(inferred_price_level)}.",
                )
            )

        if state.location_mode == "manual" and state.manual_location and not state.user_location:
            state.user_location = geocode_location(state.manual_location)
            traces.append(AgentTrace(self.name, "geocode", "Resolved the manual location with Places text search."))

        state.conversation_turns += 1
        return traces


class ClarificationAgent:
    name = "Clarification Agent"

    def next_question(self, state: ConversationState) -> Optional[str]:
        if state.llm_next_question:
            return state.llm_next_question
        if not state.cuisine and is_vague_food_request(state.dish):
            return "What cuisine are you in the mood for?"
        if state.location_mode == "manual" and not state.manual_location:
            return "What location are you thinking of? You can type a place name or coordinates."
        if not state.location_mode and not state.user_location:
            return "Can I use your current location, or do you want to enter the location manually?"
        if not state.travel_minutes:
            return "How long are you willing to travel, and is that by walk, public transport, or car?"
        if state.max_price_level is None:
            return "What price range are you comfortable with: $, $$, $$$, or $$$$?"
        return None


class RetrievalAgent:
    name = "Retrieval Agent"

    def ready_for_search(self, state: ConversationState) -> bool:
        return bool(
            state.cuisine
            and (state.user_location or state.location_mode)
            and (state.location_mode != "manual" or state.manual_location)
            and state.travel_minutes
            and state.max_price_level is not None
        )

    def run(self, state: ConversationState, limit: int) -> tuple[List[dict], List[AgentTrace]]:
        if not state.user_location:
            raise ValueError("Location is missing.")

        radius = float(state.search_radius_meters or 7000)
        location_bias = {
            "circle": {
                "center": {
                    "latitude": state.user_location["lat"],
                    "longitude": state.user_location["lng"],
                },
                "radius": radius,
            }
        }

        traces = [
            AgentTrace(
                self.name,
                "search",
                f"Searching Places with query '{build_search_query(state)}' inside a {int(radius)} meter radius.",
            )
        ]

        places = places_text_search(
            text_query=build_search_query(state),
            field_mask=(
                "places.id,places.displayName,places.formattedAddress,places.location,"
                "places.rating,places.userRatingCount,places.priceLevel,places.primaryTypeDisplayName"
            ),
            max_result_count=max(limit, 12),
            location_bias=location_bias,
        )

        results = []
        for place in places:
            details = place_details(
                place["id"],
                (
                    "id,displayName,formattedAddress,location,rating,userRatingCount,priceLevel,"
                    "primaryTypeDisplayName,currentOpeningHours,regularOpeningHours,reviewSummary,editorialSummary,"
                    "googleMapsUri,websiteUri"
                ),
            )
            location = details.get("location", {})
            if location.get("latitude") is None or location.get("longitude") is None:
                continue

            distance_miles = round(
                haversine_miles(
                    state.user_location["lat"],
                    state.user_location["lng"],
                    location["latitude"],
                    location["longitude"],
                ),
                2,
            )
            travel_estimates = build_travel_estimates(distance_miles, state.travel_mode)
            estimated_travel_minutes = estimate_travel_minutes(distance_miles, state.travel_mode)
            open_now = details.get("currentOpeningHours", {}).get("openNow")
            open_at_requested_time = is_place_open_at_requested_time(details, state)

            if state.when == "later" and open_at_requested_time is False:
                continue

            raw_price_level = details.get("priceLevel")
            price_level = raw_price_level if isinstance(raw_price_level, int) else None
            menu_verification = verify_dish_availability(details.get("websiteUri"), state.dish)
            if state.max_price_level is not None and price_level is not None and price_level > state.max_price_level:
                continue

            score = float(details.get("rating", 0) or 0) * 2.0
            score += min(details.get("userRatingCount", 0), 3000) / 1000
            score += max(0.0, 5.0 - min(distance_miles, 5.0))
            if estimated_travel_minutes is not None and state.travel_minutes:
                preferred_center = (
                    (state.min_travel_minutes + state.travel_minutes) / 2
                    if state.min_travel_minutes is not None
                    else state.travel_minutes
                )
                score += max(0.0, 3.0 - abs(preferred_center - estimated_travel_minutes) / 5.0)
            if state.max_price_level is not None and price_level is not None:
                score += max(0.0, 1.5 - abs(state.max_price_level - price_level) * 0.75)
            if state.dish:
                if menu_verification.get("verified"):
                    score += 3.0
                elif menu_verification.get("status") == "not_verified":
                    score -= 2.0
            if state.when == "later":
                if open_at_requested_time is True:
                    score += 2.0
                elif open_at_requested_time is None:
                    score -= 0.5

            result = {
                "name": details.get("displayName", {}).get("text"),
                "address": details.get("formattedAddress"),
                "rating": details.get("rating"),
                "user_rating_count": details.get("userRatingCount"),
                "price_level": price_level,
                "distance_miles": distance_miles,
                "estimated_travel_minutes": estimated_travel_minutes,
                "travel_estimates": travel_estimates,
                "primary_type": details.get("primaryTypeDisplayName", {}).get("text"),
                "open_now": open_now,
                "open_at_requested_time": open_at_requested_time,
                "summary": details.get("reviewSummary", {}).get("text")
                or details.get("editorialSummary", {}).get("text"),
                "website_url": details.get("websiteUri"),
                "menu_verification": menu_verification,
                "google_maps_url": details.get("googleMapsUri"),
                "directions_url": build_directions_url(
                    state.user_location,
                    {"lat": location["latitude"], "lng": location["longitude"]},
                    state.travel_mode,
                ),
                "lat": location.get("latitude"),
                "lng": location.get("longitude"),
                "score": round(score, 2),
            }
            result["unmet_criteria"] = unmet_criteria_for_place(result, state)
            result["matched_criteria"] = self._matched_criteria(result, state)
            result["verification_status"] = verification_status_for_place(result, state)
            result["confidence_score"] = confidence_score_for_place(result, state)
            result["confidence_label"] = confidence_label(result["confidence_score"])
            results.append(result)

        ranked_results = sorted(results, key=lambda item: item["score"], reverse=True)
        state.last_results = self._diversify_results(ranked_results, limit)
        state.last_limit = limit
        traces.append(
            AgentTrace(
                self.name,
                "rank",
                f"Ranked {len(state.last_results)} place(s) and kept the top {limit}.",
            )
        )
        return state.last_results, traces

    def _diversify_results(self, ranked_results: List[dict], limit: int) -> List[dict]:
        diversified: List[dict] = []
        seen_names = set()
        leftovers: List[dict] = []

        for result in ranked_results:
            canonical_name = canonicalize_place_name(result.get("name"))
            if canonical_name and canonical_name not in seen_names:
                seen_names.add(canonical_name)
                diversified.append(result)
            else:
                leftovers.append(result)
            if len(diversified) == limit:
                return diversified

        for result in leftovers:
            diversified.append(result)
            if len(diversified) == limit:
                break

        return diversified

    def _matched_criteria(self, place: dict, state: ConversationState) -> List[str]:
        matched = []
        if state.cuisine:
            matched.append(f"fits the {state.cuisine} intent")
        if state.dish and place.get("menu_verification", {}).get("verified"):
            matched.append(place["menu_verification"]["label"])
        if state.when == "now" and place.get("open_now") is True:
            matched.append("open right now")
        if state.when == "later" and place.get("open_at_requested_time") is True:
            target_dt = requested_datetime_for_state(state)
            if target_dt is not None:
                matched.append(f"likely open around {target_dt.strftime('%-I:%M %p').lower()} on {target_dt.strftime('%A')}")
        if state.max_price_level is not None and place.get("price_level") is not None and int(place["price_level"]) <= int(state.max_price_level):
            matched.append(f"within your budget range of {format_price_level(state.max_price_level)}")
        if state.min_rating is not None and place.get("rating") is not None and float(place["rating"]) >= float(state.min_rating):
            matched.append(f"meets your minimum rating of {state.min_rating}")
        if state.travel_minutes and place.get("estimated_travel_minutes") and place["estimated_travel_minutes"] <= state.travel_minutes:
            matched.append("within your travel-time preference")
        elif state.travel_minutes and place.get("estimated_travel_minutes"):
            extra_minutes = place["estimated_travel_minutes"] - state.travel_minutes
            if 0 < extra_minutes <= TRAVEL_TIME_LEEWAY_MINUTES:
                matched.append(f"close to your travel preference, only about {extra_minutes} minutes farther")
        return matched


class ResponseAgent:
    name = "Response Agent"

    def build_reply(self, state: ConversationState, results: List[dict], limit: int) -> str:
        if gemini_enabled():
            try:
                llm_reply = build_final_response_with_gemini(asdict(state), results, limit)
                if llm_reply:
                    return llm_reply
            except Exception:
                pass

        lines = [f"Here are the top {limit} recommendations I found:"]
        for idx, place in enumerate(results, start=1):
            detail_bits = []
            if place["rating"] is not None:
                detail_bits.append(f"rated {place['rating']}")
            price_label = format_price_level(place.get("price_level"))
            if price_label:
                detail_bits.append(f"price level {price_label}")
            if place["distance_miles"] is not None:
                detail_bits.append(f"{place['distance_miles']} miles away")
            travel_estimate_text = format_travel_estimates(place.get("travel_estimates") or {})
            if travel_estimate_text:
                detail_bits.append(travel_estimate_text)
            elif place["estimated_travel_minutes"] is not None:
                detail_bits.append(f"about {place['estimated_travel_minutes']} minutes by {travel_mode_label(state.travel_mode)}")
            if place["open_now"] is True:
                detail_bits.append("open now")
            line = f"{idx}. {place['name']} - {place['address']}"
            if detail_bits:
                line += f" ({', '.join(detail_bits)})"
            line += f"\n   Confidence: {place.get('confidence_label', 'medium confidence')} ({place.get('confidence_score', 0)}/100)."
            status = place.get("verification_status")
            if status == "verified":
                line += "\n   Verification: verified."
            elif status == "not_verified":
                line += "\n   Verification: not verified."
            else:
                line += "\n   Verification: likely match, but not fully verified."
            line += f"\n   Fit: {fit_label_for_place(place)}"
            verification = place.get("menu_verification") or {}
            if state.dish and verification.get("label"):
                line += f"\n   Menu check: {verification['label']}."
                if verification.get("evidence"):
                    line += f" Evidence: {verification['evidence']}"
            if place["matched_criteria"]:
                line += f"\n   Matches: {', '.join(place['matched_criteria'])}."
            if place["unmet_criteria"]:
                line += f"\n   Does not fully follow: {', '.join(place['unmet_criteria'])}."
            if place["summary"]:
                line += f"\n   Why it still stands out: {place['summary']}"
            lines.append(line)
        lines.append("Ask for top ten recommendations if you want a longer list.")
        return "\n".join(lines)


class CoordinatorAgent:
    def __init__(self) -> None:
        self.intent_agent = IntentAgent()
        self.clarification_agent = ClarificationAgent()
        self.retrieval_agent = RetrievalAgent()
        self.response_agent = ResponseAgent()

    def handle_turn(
        self,
        state: ConversationState,
        message: str,
        browser_location: Optional[dict],
    ) -> Dict[str, Any]:
        traces: List[AgentTrace] = []

        top_k_request = parse_top_k_request(message)
        if top_k_request and state.last_results:
            results, retrieval_traces = self.retrieval_agent.run(state, top_k_request)
            traces.extend(retrieval_traces)
            return self._payload(state, self.response_agent.build_reply(state, results, top_k_request), results, traces)

        traces.extend(self.intent_agent.run(state, message, browser_location))
        question = self.clarification_agent.next_question(state)

        if not self.retrieval_agent.ready_for_search(state):
            reply = question or "Tell me a little more so I can narrow it down."
            traces.append(AgentTrace("Clarification Agent", "question", reply))
            return self._payload(state, reply, [], traces)

        results, retrieval_traces = self.retrieval_agent.run(state, 5)
        traces.extend(retrieval_traces)
        reply = self.response_agent.build_reply(state, results, 5)
        traces.append(AgentTrace("Response Agent", "compose", "Built the final ranked recommendation message."))
        return self._payload(state, reply, results, traces)

    def _payload(
        self,
        state: ConversationState,
        reply: str,
        results: List[dict],
        traces: List[AgentTrace],
    ) -> Dict[str, Any]:
        return {
            "reply": reply,
            "results": results,
            "user_location": state.user_location,
            "needs_map": bool(results and state.user_location),
            "state": asdict(state),
            "agent_trace": [asdict(trace) for trace in traces],
            "project_summary": {
                "title": "Agentic Dining Assistant",
                "architecture": [
                    "Intent Agent: extracts slots like cuisine, time, location, and travel preference.",
                    "Clarification Agent: asks only for missing information before search.",
                    "Retrieval Agent: grounds results in live Google Places data and ranks candidates.",
                    "Response Agent: explains why each recommendation matches or misses criteria.",
                ],
            },
        }


coordinator = CoordinatorAgent()


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/session")
def create_session():
    session_id = uuid.uuid4().hex
    get_state(session_id)
    return jsonify({"session_id": session_id})


@app.post("/api/chat")
def chat():
    payload = request.get_json(force=True)
    session_id = payload.get("session_id")
    if not session_id:
        return jsonify({"error": "Missing session_id"}), 400

    message = (payload.get("message") or "").strip()
    if not message:
        return jsonify({"error": "Missing message"}), 400

    browser_location = payload.get("browser_location")
    state = get_state(session_id)

    try:
        return jsonify(coordinator.handle_turn(state, message, browser_location))
    except Exception as exc:
        return jsonify(
            {
                "reply": f"I hit an error while processing that turn: {exc}",
                "results": [],
                "user_location": state.user_location,
                "needs_map": False,
                "state": asdict(state),
                "agent_trace": [
                    asdict(AgentTrace("Coordinator", "error", str(exc))),
                ],
            }
        ), 500


if __name__ == "__main__":
    app.run(debug=True, port=int(os.getenv("PORT", "5001")))
