import json
import math
import os
import re
import secrets
import sqlite3
import uuid
import smtplib
import ssl
from functools import wraps
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from email.message import EmailMessage
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from flask import Flask, Response, jsonify, redirect, render_template, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash

try:
    import psycopg
    from psycopg.rows import dict_row
except ImportError:
    psycopg = None
    dict_row = None

from places_restaurant_chatbot import (
    analyze_app_turn_with_gemini,
    build_final_response_with_gemini,
    fetch_place_photo_content,
    gemini_enabled,
    geocode_location,
    haversine_miles,
    normalize_text,
    place_details,
    places_text_search,
    verify_dish_availability,
)


app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "instadine-dev-secret-key")
app.permanent_session_lifetime = timedelta(days=30)
ENABLE_GEMINI_FINAL_RESPONSE = os.getenv("ENABLE_GEMINI_FINAL_RESPONSE", "").lower() in {"1", "true", "yes", "on"}

TRAVEL_TIME_LEEWAY_MINUTES = 20
APP_TIMEZONE = ZoneInfo("America/New_York")
DATA_DIR = Path(__file__).resolve().parent / "data"
SESSION_STORE_PATH = DATA_DIR / "chat_sessions.json"
USER_STORE_PATH = DATA_DIR / "users.json"
DB_PATH = DATA_DIR / "instadine.db"
DB_BACKEND = os.getenv("DB_BACKEND", "sqlite").strip().lower()
DB_NAME = os.getenv("DB_NAME", "").strip()
DB_USER = os.getenv("DB_USER", "").strip()
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_HOST = os.getenv("DB_HOST", "").strip()
DB_PORT = int(os.getenv("DB_PORT", "5432"))
INSTANCE_CONNECTION_NAME = os.getenv("INSTANCE_CONNECTION_NAME", "").strip()
DEFAULT_ASSISTANT_MESSAGE = (
    "Tell me what you want to eat, where you want to eat, and how far you're willing to travel. "
    "If you want, you can also include timing, cuisine, or a minimum rating."
)
BASELINE_PLACE_RATING = 4.2
REVIEW_CONFIDENCE_PRIOR = 200


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
    "ice cream": "dessert",
    "gelato": "dessert",
    "sorbet": "dessert",
}

VAGUE_FOOD_INTENTS = {
    "breakfast",
    "brunch",
    "lunch",
    "dinner",
    "snack",
    "dessert",
    "sweets",
    "food",
    "eat",
}

DISH_QUERY_HINTS = {
    "ice cream": "ice cream shop",
    "gelato": "gelato shop",
    "sorbet": "dessert shop",
    "froyo": "frozen yogurt shop",
    "frozen yogurt": "frozen yogurt shop",
    "dessert": "dessert shop",
    "cake": "bakery",
    "croissant": "bakery",
    "pastry": "bakery",
    "coffee": "coffee shop",
}

CUISINE_ONLY_TERMS = {
    "indian",
    "south indian",
    "north indian",
    "italian",
    "chinese",
    "indo chinese",
    "indo-chinese",
    "thai",
    "korean",
    "japanese",
    "mexican",
    "american",
    "coffee",
    "japanese food",
    "indian food",
    "italian food",
    "chinese food",
    "mexican food",
}


@dataclass
class ConversationState:
    title: str = "New chat"
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
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
    min_rating: Optional[float] = None
    requested_day_offset: Optional[int] = None
    requested_hour: Optional[int] = None
    requested_minute: Optional[int] = None
    llm_next_question: Optional[str] = None
    messages: List[dict] = field(default_factory=list)
    last_results: List[dict] = field(default_factory=list)
    last_limit: int = 5
    conversation_turns: int = 0


@dataclass
class AgentTrace:
    agent: str
    action: str
    details: str


DB_LOCK = Lock()


def using_postgres() -> bool:
    return DB_BACKEND == "postgres"


def db_sql(query: str) -> str:
    if using_postgres():
        return query.replace("?", "%s")
    return query


def _postgres_connect_kwargs() -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "dbname": DB_NAME,
        "user": DB_USER,
        "password": DB_PASSWORD,
        "port": DB_PORT,
    }
    if INSTANCE_CONNECTION_NAME:
        kwargs["host"] = f"/cloudsql/{INSTANCE_CONNECTION_NAME}"
    elif DB_HOST:
        kwargs["host"] = DB_HOST
    if dict_row is not None:
        kwargs["row_factory"] = dict_row
    return kwargs


def run_schema_statements(connection: Any, statements: List[str]) -> None:
    for statement in statements:
        cleaned = statement.strip()
        if cleaned:
            connection.execute(cleaned)


def now_iso() -> str:
    return datetime.now(APP_TIMEZONE).isoformat()


def build_session_title(message: str) -> str:
    cleaned = " ".join((message or "").strip().split())
    if not cleaned:
        return "New chat"
    return cleaned[:60] + ("..." if len(cleaned) > 60 else "")


def normalize_email(value: str) -> str:
    return " ".join((value or "").strip().lower().split())


def mask_email(value: str) -> str:
    email = normalize_email(value)
    if not email or "@" not in email:
        return ""
    local_part, domain = email.split("@", 1)
    if not local_part:
        masked_local = "****"
    elif len(local_part) == 1:
        masked_local = f"{local_part}***"
    elif len(local_part) == 2:
        masked_local = f"{local_part[0]}***"
    else:
        masked_local = f"{local_part[0]}{'*' * max(3, len(local_part) - 2)}{local_part[-1]}"
    return f"{masked_local}@{domain}"


def state_payload_from_state(state: ConversationState) -> Dict[str, Any]:
    payload = asdict(state)
    payload.pop("messages", None)
    payload.pop("title", None)
    payload.pop("created_at", None)
    payload.pop("updated_at", None)
    return payload


def state_from_payload(data: Dict[str, Any], title: str, created_at: Optional[str], updated_at: Optional[str], messages: List[dict]) -> ConversationState:
    state = ConversationState()
    for field_name in state.__dataclass_fields__:
        if field_name in {"messages", "title", "created_at", "updated_at"}:
            continue
        if field_name in data:
            setattr(state, field_name, data[field_name])
    state.title = title or "New chat"
    state.created_at = created_at or now_iso()
    state.updated_at = updated_at or state.created_at
    state.messages = messages or [{"role": "assistant", "text": DEFAULT_ASSISTANT_MESSAGE}]
    return state


def append_message(state: ConversationState, role: str, text: str) -> None:
    state.messages.append({"role": role, "text": text})
    state.updated_at = now_iso()


def db_connection() -> Any:
    if using_postgres():
        if psycopg is None:
            raise RuntimeError("Postgres support requires psycopg to be installed.")
        missing = [
            name
            for name, value in {
                "DB_NAME": DB_NAME,
                "DB_USER": DB_USER,
                "DB_PASSWORD": DB_PASSWORD,
            }.items()
            if not value
        ]
        if missing:
            raise RuntimeError(f"Missing Postgres configuration: {', '.join(missing)}")
        return psycopg.connect(**_postgres_connect_kwargs())

    DATA_DIR.mkdir(exist_ok=True)
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def init_db() -> None:
    sqlite_schema = [
        """
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS chat_sessions (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            title TEXT NOT NULL,
            state_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            text TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS password_reset_tokens (
            token TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            used_at TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
        """,
    ]
    postgres_schema = [
        """
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS chat_sessions (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            title TEXT NOT NULL,
            state_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS chat_messages (
            id BIGSERIAL PRIMARY KEY,
            session_id TEXT NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
            role TEXT NOT NULL,
            text TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS password_reset_tokens (
            token TEXT PRIMARY KEY,
            user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            expires_at TEXT NOT NULL,
            used_at TEXT,
            created_at TEXT NOT NULL
        )
        """,
    ]

    with DB_LOCK:
        connection = db_connection()
        try:
            run_schema_statements(connection, postgres_schema if using_postgres() else sqlite_schema)
            connection.commit()
        finally:
            connection.close()


def fetch_user_by_id(user_id: str) -> Optional[Dict[str, Any]]:
    connection = db_connection()
    try:
        row = connection.execute(
            db_sql("SELECT id, name, email, password_hash, created_at FROM users WHERE id = ?"),
            (user_id,),
        ).fetchone()
    finally:
        connection.close()
    if row is None:
        return None
    return dict(row) if not isinstance(row, dict) else row


def fetch_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    normalized = normalize_email(email)
    connection = db_connection()
    try:
        row = connection.execute(
            db_sql("SELECT id, name, email, password_hash, created_at FROM users WHERE email = ?"),
            (normalized,),
        ).fetchone()
    finally:
        connection.close()
    if row is None:
        return None
    return dict(row) if not isinstance(row, dict) else row


def create_user_record(name: str, email: str, password_hash: str) -> str:
    user_id = uuid.uuid4().hex
    created_at = now_iso()
    with DB_LOCK:
        connection = db_connection()
        try:
            connection.execute(
                db_sql("INSERT INTO users (id, name, email, password_hash, created_at) VALUES (?, ?, ?, ?, ?)"),
                (user_id, name, normalize_email(email), password_hash, created_at),
            )
            connection.commit()
        finally:
            connection.close()
    return user_id


def password_reset_delivery_available() -> bool:
    return bool(
        os.getenv("SMTP_HOST")
        and os.getenv("SMTP_USERNAME")
        and os.getenv("SMTP_PASSWORD")
        and os.getenv("SMTP_FROM_EMAIL")
    )


def create_password_reset_token(user_id: str) -> str:
    token = secrets.token_urlsafe(32)
    created_at = now_iso()
    expires_at = (datetime.now(APP_TIMEZONE) + timedelta(hours=1)).isoformat()
    with DB_LOCK:
        connection = db_connection()
        try:
            connection.execute(db_sql("DELETE FROM password_reset_tokens WHERE user_id = ?"), (user_id,))
            connection.execute(
                db_sql("INSERT INTO password_reset_tokens (token, user_id, expires_at, used_at, created_at) VALUES (?, ?, ?, NULL, ?)"),
                (token, user_id, expires_at, created_at),
            )
            connection.commit()
        finally:
            connection.close()
    return token


def fetch_password_reset_token(token: str) -> Optional[Dict[str, Any]]:
    connection = db_connection()
    try:
        row = connection.execute(
            db_sql(
                """
            SELECT prt.token, prt.user_id, prt.expires_at, prt.used_at, prt.created_at,
                   u.email, u.name
            FROM password_reset_tokens prt
            JOIN users u ON u.id = prt.user_id
            WHERE prt.token = ?
            """
            ),
            (token,),
        ).fetchone()
    finally:
        connection.close()
    return (dict(row) if not isinstance(row, dict) else row) if row else None


def password_reset_token_is_valid(token_row: Optional[Dict[str, Any]]) -> bool:
    if not token_row or token_row.get("used_at"):
        return False
    try:
        expires_at = datetime.fromisoformat(token_row["expires_at"])
    except Exception:
        return False
    return expires_at >= datetime.now(APP_TIMEZONE)


def mark_password_reset_token_used(token: str) -> None:
    with DB_LOCK:
        connection = db_connection()
        try:
            connection.execute(
                db_sql("UPDATE password_reset_tokens SET used_at = ? WHERE token = ?"),
                (now_iso(), token),
            )
            connection.commit()
        finally:
            connection.close()


def update_user_password(user_id: str, password_hash: str) -> None:
    with DB_LOCK:
        connection = db_connection()
        try:
            connection.execute(
                db_sql("UPDATE users SET password_hash = ? WHERE id = ?"),
                (password_hash, user_id),
            )
            connection.commit()
        finally:
            connection.close()


def send_password_reset_email(recipient_email: str, recipient_name: str, reset_link: str) -> None:
    smtp_host = os.getenv("SMTP_HOST")
    smtp_username = os.getenv("SMTP_USERNAME")
    smtp_password = os.getenv("SMTP_PASSWORD")
    smtp_from_email = os.getenv("SMTP_FROM_EMAIL")
    if not all([smtp_host, smtp_username, smtp_password, smtp_from_email]):
        raise RuntimeError("SMTP is not configured.")

    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    use_tls = os.getenv("SMTP_USE_TLS", "true").lower() in {"1", "true", "yes", "on"}

    message = EmailMessage()
    message["Subject"] = "Reset your InstaDine password"
    message["From"] = smtp_from_email
    message["To"] = recipient_email
    message.set_content(
        f"""Hi {recipient_name or 'there'},

We received a request to reset your InstaDine password.

Use this link to choose a new password:
{reset_link}

This link expires in 1 hour. If you did not request this, you can ignore this email.
"""
    )

    if use_tls:
        context = ssl.create_default_context()
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls(context=context)
            server.login(smtp_username, smtp_password)
            server.send_message(message)
    else:
        with smtplib.SMTP_SSL(smtp_host, smtp_port) as server:
            server.login(smtp_username, smtp_password)
            server.send_message(message)


def session_summary(session_id: str, state: ConversationState) -> Dict[str, Any]:
    return {
        "session_id": session_id,
        "title": state.title,
        "updated_at": state.updated_at,
        "created_at": state.created_at,
    }


def initialize_session_state(state: ConversationState) -> ConversationState:
    if not state.created_at:
        state.created_at = now_iso()
    if not state.updated_at:
        state.updated_at = state.created_at
    if not state.title:
        state.title = "New chat"
    if not state.messages:
        state.messages = [{"role": "assistant", "text": DEFAULT_ASSISTANT_MESSAGE}]
    return state


def create_session_record(user_id: str) -> tuple[str, ConversationState]:
    session_id = uuid.uuid4().hex
    state = initialize_session_state(ConversationState())
    state.created_at = now_iso()
    state.updated_at = state.created_at
    state.title = "New chat"
    state.messages = [{"role": "assistant", "text": DEFAULT_ASSISTANT_MESSAGE}]
    with DB_LOCK:
        connection = db_connection()
        try:
            connection.execute(
                db_sql("INSERT INTO chat_sessions (id, user_id, title, state_json, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)"),
                (
                    session_id,
                    user_id,
                    state.title,
                    json.dumps(state_payload_from_state(state), ensure_ascii=False),
                    state.created_at,
                    state.updated_at,
                ),
            )
            for message in state.messages:
                connection.execute(
                    db_sql("INSERT INTO chat_messages (session_id, role, text, created_at) VALUES (?, ?, ?, ?)"),
                    (session_id, message["role"], message["text"], state.created_at),
                )
            connection.commit()
        finally:
            connection.close()
    return session_id, state


def list_sessions_for_user(user_id: str) -> List[Dict[str, Any]]:
    connection = db_connection()
    try:
        rows = connection.execute(
            db_sql("SELECT id, title, created_at, updated_at FROM chat_sessions WHERE user_id = ? ORDER BY updated_at DESC"),
            (user_id,),
        ).fetchall()
    finally:
        connection.close()
    return [
        {
            "session_id": row["id"] if isinstance(row, dict) else row["id"],
            "title": row["title"] if isinstance(row, dict) else row["title"],
            "created_at": row["created_at"] if isinstance(row, dict) else row["created_at"],
            "updated_at": row["updated_at"] if isinstance(row, dict) else row["updated_at"],
        }
        for row in rows
    ]


def load_session_state(user_id: str, session_id: str) -> Optional[ConversationState]:
    connection = db_connection()
    try:
        session_row = connection.execute(
            db_sql("SELECT id, title, state_json, created_at, updated_at FROM chat_sessions WHERE id = ? AND user_id = ?"),
            (session_id, user_id),
        ).fetchone()
        if session_row is None:
            return None
        message_rows = connection.execute(
            db_sql("SELECT role, text FROM chat_messages WHERE session_id = ? ORDER BY id ASC"),
            (session_id,),
        ).fetchall()
    finally:
        connection.close()

    try:
        payload = json.loads(session_row["state_json"] or "{}")
    except Exception:
        payload = {}
    messages = [{"role": row["role"], "text": row["text"]} for row in message_rows]
    return initialize_session_state(
        state_from_payload(payload, session_row["title"], session_row["created_at"], session_row["updated_at"], messages)
    )


def save_session_state(user_id: str, session_id: str, state: ConversationState) -> None:
    state = initialize_session_state(state)
    with DB_LOCK:
        connection = db_connection()
        try:
            owned = connection.execute(
                db_sql("SELECT 1 FROM chat_sessions WHERE id = ? AND user_id = ?"),
                (session_id, user_id),
            ).fetchone()
            if owned is None:
                raise ValueError("Session not found")
            connection.execute(
                db_sql("UPDATE chat_sessions SET title = ?, state_json = ?, updated_at = ? WHERE id = ? AND user_id = ?"),
                (
                    state.title,
                    json.dumps(state_payload_from_state(state), ensure_ascii=False),
                    state.updated_at or now_iso(),
                    session_id,
                    user_id,
                ),
            )
            connection.execute(db_sql("DELETE FROM chat_messages WHERE session_id = ?"), (session_id,))
            for message in state.messages:
                connection.execute(
                    db_sql("INSERT INTO chat_messages (session_id, role, text, created_at) VALUES (?, ?, ?, ?)"),
                    (session_id, message["role"], message["text"], now_iso()),
                )
            connection.commit()
        finally:
            connection.close()


def delete_session_record(user_id: str, session_id: str) -> bool:
    with DB_LOCK:
        connection = db_connection()
        try:
            cursor = connection.execute(
                db_sql("DELETE FROM chat_sessions WHERE id = ? AND user_id = ?"),
                (session_id, user_id),
            )
            connection.commit()
            return cursor.rowcount > 0
        finally:
            connection.close()


def current_user_id() -> Optional[str]:
    user_id = session.get("user_id")
    if isinstance(user_id, str) and fetch_user_by_id(user_id):
        return user_id
    return None


def current_user() -> Optional[Dict[str, Any]]:
    user_id = current_user_id()
    if not user_id:
        return None
    user = fetch_user_by_id(user_id)
    if not user:
        return None
    return {"user_id": user["id"], **user}


def find_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    user = fetch_user_by_email(email)
    if not user:
        return None
    return {"user_id": user["id"], **user}


def login_user(user_id: str, remember_me: bool) -> None:
    session["user_id"] = user_id
    session.permanent = bool(remember_me)


def logout_user() -> None:
    session.pop("user_id", None)


def login_required(view_func):
    @wraps(view_func)
    def wrapped(*args, **kwargs):
        if not current_user_id():
            if request.path.startswith("/api/"):
                return jsonify({"error": "Please sign in to continue."}), 401
            return redirect(url_for("index"))
        return view_func(*args, **kwargs)

    return wrapped

def get_state(user_id: str, session_id: str) -> ConversationState:
    state = load_session_state(user_id, session_id)
    if state is None:
        raise ValueError("Session not found")
    return state


def is_place_follow_up_request(message: str) -> bool:
    cleaned = normalize_text(message)
    phrases = [
        "tell me more about",
        "more about",
        "more info on",
        "more info about",
        "more information on",
        "more information about",
        "what about",
        "details on",
        "details about",
        "tell me about",
        "has ",
        "does ",
        "is ",
        "right",
    ]
    return any(phrase in cleaned for phrase in phrases) or "?" in message


def find_follow_up_place(message: str, results: List[dict]) -> Optional[dict]:
    if not results or not is_place_follow_up_request(message):
        return None
    cleaned_message = normalize_text(message)
    for place in results:
        cleaned_name = normalize_text(place.get("name") or "")
        if cleaned_name and cleaned_name in cleaned_message:
            return place
        if cleaned_name.startswith("the ") and cleaned_name[4:] in cleaned_message:
            return place
        compact_name = re.sub(r"[^a-z0-9]+", " ", cleaned_name).strip()
        if compact_name and compact_name in cleaned_message:
            return place

    message_tokens = {
        token
        for token in re.split(r"\s+", cleaned_message)
        if token and token not in {"tell", "me", "about", "more", "info", "details", "restaurant", "place", "the"}
    }
    if not message_tokens:
        return None

    best_place = None
    best_score = 0
    for place in results:
        cleaned_name = normalize_text(place.get("name") or "")
        place_tokens = {
            token
            for token in re.split(r"\s+", re.sub(r"[^a-z0-9]+", " ", cleaned_name))
            if token and token not in {"restaurant", "cafe", "bar", "grill", "kitchen", "co", "cof", "the"}
        }
        overlap = len(message_tokens & place_tokens)
        if overlap > best_score:
            best_score = overlap
            best_place = place
    if best_score >= 2:
        return best_place
    return None


def exact_reason_summary(place: dict) -> str:
    unmet = place.get("unmet_criteria") or []
    if unmet:
        return unmet[0]
    matched = place.get("matched_criteria") or []
    if matched:
        return matched[0]
    return "it looks like a possible match, but the evidence is limited"


def summary_text(value: Any) -> Optional[str]:
    if isinstance(value, dict):
        text = value.get("text")
        if isinstance(text, str) and text.strip():
            return text.strip()
        return None
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def confidence_weighted_rating(rating: Any, review_count: Any) -> float:
    try:
        rating_value = float(rating or 0.0)
    except (TypeError, ValueError):
        rating_value = 0.0
    try:
        count_value = max(int(review_count or 0), 0)
    except (TypeError, ValueError):
        count_value = 0
    if rating_value <= 0:
        return 0.0
    confidence = count_value / (count_value + REVIEW_CONFIDENCE_PRIOR)
    return (confidence * rating_value) + ((1.0 - confidence) * BASELINE_PLACE_RATING)


def review_volume_bonus(review_count: Any) -> float:
    try:
        count_value = max(int(review_count or 0), 0)
    except (TypeError, ValueError):
        return 0.0
    capped = min(count_value, 5000)
    return min(math.log1p(capped) / 4.0, 2.0)


init_db()


def parse_top_k_request(message: str) -> Optional[int]:
    cleaned = normalize_text(message)
    for top_k, patterns in TOP_K_PATTERNS.items():
        if any(re.search(pattern, cleaned) for pattern in patterns):
            return top_k
    return None


def parse_min_rating(message: str) -> Optional[float]:
    cleaned = normalize_text(message)
    match = re.search(r"(?:at least|min(?:imum)?|rating(?:\s+of)?)\s*(\d(?:\.\d)?)", cleaned)
    if match:
        return float(match.group(1))
    if re.fullmatch(r"\d(?:\.\d)?", cleaned):
        value = float(cleaned)
        if 1.0 <= value <= 5.0:
            return value
    return None


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


def looks_like_time_phrase(text: Optional[str]) -> bool:
    cleaned = normalize_text(text or "")
    if not cleaned:
        return False
    return bool(
        re.fullmatch(
            r"(?:(?:today|tomorrow|tonight|this afternoon|this evening|afternoon|evening)\s+)?"
            r"(?:at\s+)?\d{1,2}(?::\d{2})?\s*(?:am|pm)"
            r"(?:\s+in\s+the\s+(?:afternoon|evening))?",
            cleaned,
        )
        or re.fullmatch(r"(?:the\s+)?(?:afternoon|evening|tonight)", cleaned)
    )


def infer_manual_location_text(message: str) -> Optional[str]:
    raw = (message or "").strip()
    cleaned = normalize_text(raw)

    coordinate_location = parse_coordinate_pair(raw)
    if coordinate_location:
        return raw

    if re.fullmatch(
        r"(?:today|tomorrow|tonight|this afternoon|this evening|afternoon|evening)(?:\s+at\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)?)?",
        cleaned,
    ):
        return None

    patterns = [
        r"\bnear\s+(.+?)(?:\s+with\b|\s+and\b|$)",
        r"\bat\s+(.+?)(?:\s+with\b|\s+and\b|$)",
        r"\bin\s+(.+?)(?:\s+with\b|\s+and\b|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, cleaned)
        if match:
            location_text = match.group(1).strip(" ,.")
            if location_text and not re.fullmatch(r"\d{1,2}(?::\d{2})?\s*(?:am|pm)?", location_text) and not looks_like_time_phrase(location_text):
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


def is_cuisine_only_phrase(text: Optional[str]) -> bool:
    cleaned = normalize_text(text or "")
    return cleaned in CUISINE_ONLY_TERMS


def question_requests_cuisine(question: Optional[str]) -> bool:
    cleaned = normalize_text(question or "")
    if not cleaned:
        return False
    return "cuisine" in cleaned or "what kind of food" in cleaned


def search_category_for_state(state: ConversationState) -> str:
    combined = " ".join(part for part in [state.dish, state.cuisine] if part).strip()
    cleaned = normalize_text(combined)
    for key, hint in sorted(DISH_QUERY_HINTS.items(), key=lambda item: len(item[0]), reverse=True):
        if key in cleaned:
            return hint
    if state.cuisine == "coffee":
        return "coffee shop"
    if state.cuisine == "dessert":
        return "dessert shop"
    return "restaurant"


def infer_dish(message: str) -> Optional[str]:
    cleaned = normalize_text(message)
    if not cleaned:
        return None

    if re.fullmatch(r"\d(?:\.\d)?", cleaned):
        return None

    non_food_markers = [
        "rating",
        "stars",
        "star",
        "walk",
        "walking",
        "transit",
        "public transport",
        "public transit",
        "subway",
        "train",
        "bus",
        "car",
        "drive",
        "driving",
        "price",
        "budget",
        "under $",
        "under $$",
        "under $$$",
        "under $$$$",
        "show me",
        "top 10",
        "top ten",
        "open at",
        "tomorrow",
        "afternoon",
        "evening",
        "more about",
        "tell me more",
        "details about",
        "is it",
        "is there",
        "right now",
        "now",
        "later",
        "open now",
        "get there",
        "there by",
    ]
    if any(marker in cleaned for marker in non_food_markers):
        return None

    candidates = [
        r"i want to (?:have|eat|drink|try)\s+(.+?)(?:\s+\bnear\b|\s+\bin\b|\s+\bat\b|\s+\bwith\b|\s+\band\b|$)",
        r"i want to eat (.+?)(?:\s+\bnear\b|\s+\bin\b|\s+\bat\b|\s+\bwith\b|\s+\band\b|$)",
        r"i want (.+?)(?:\s+\bnear\b|\s+\bin\b|\s+\bat\b|\s+\bwith\b|\s+\band\b|$)",
        r"i crave (.+)",
        r"eat (.+?)(?:\s+\bnear\b|\s+\bin\b|\s+\bat\b|\s+\bwith\b|\s+\band\b|$)",
    ]
    for pattern in candidates:
        match = re.search(pattern, cleaned)
        if match:
            value = match.group(1).strip(" .")
            value = re.sub(r"^(?:a|an|the)\s+", "", value)
            if value and value not in {"under", "over", "price", "budget", "there"} and not is_cuisine_only_phrase(value):
                return value
    if len(cleaned.split()) <= 4 and cleaned not in {"under", "over", "there"} and not is_cuisine_only_phrase(cleaned):
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
    walk_requested = any(word in cleaned for word in ["walk", "walking", "foot", "by foot", "on foot"])
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
    hour_match = None
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
        numeric_value = int(cleaned)
        if numeric_value > 5:
            travel_minutes = numeric_value

    # Broader time windows are usually interpreted as driving unless the user
    # explicitly mentions a different mode.
    if travel_mode is None and travel_minutes is not None and (hour_match is not None or travel_minutes >= 30):
        travel_mode = "car"

    search_radius_meters = None
    if travel_minutes is not None:
        capped_travel_minutes = travel_minutes + TRAVEL_TIME_LEEWAY_MINUTES
        radius_candidates = []
        for mode in allowed_travel_modes(travel_mode):
            if mode == "walk":
                radius_candidates.append(max(500, min(capped_travel_minutes * 80, 5000)))
            elif mode == "car":
                radius_candidates.append(max(1500, min(capped_travel_minutes * 700, 30000)))
            elif mode == "transit":
                radius_candidates.append(max(1500, min(capped_travel_minutes * 400, 20000)))
        if radius_candidates:
            search_radius_meters = max(radius_candidates)
        else:
            search_radius_meters = max(1000, min(capped_travel_minutes * 250, 15000))

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
    category = search_category_for_state(state)
    cuisine_text = normalize_text(state.cuisine or "")
    if state.dish and category != "restaurant":
        cuisine_should_be_added = False
    else:
        cuisine_should_be_added = True
    if (
        cuisine_should_be_added
        and state.cuisine
        and state.cuisine not in " ".join(parts)
        and cuisine_text not in normalize_text(category)
    ):
        parts.append(state.cuisine)
    if category not in " ".join(parts):
        parts.append(category)
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


def format_day_time(hour: int, minute: int) -> str:
    dt = datetime(2000, 1, 1, hour, minute)
    return dt.strftime("%-I:%M %p").lower()


def opening_hours_summary(details: dict) -> Optional[str]:
    current_hours = details.get("currentOpeningHours") or {}
    weekday_descriptions = current_hours.get("weekdayDescriptions") or []
    if weekday_descriptions:
        today_name = datetime.now(APP_TIMEZONE).strftime("%A")
        for line in weekday_descriptions:
            if isinstance(line, str) and line.startswith(f"{today_name}:"):
                return line

    periods = details.get("regularOpeningHours", {}).get("periods") or []
    if not periods:
        return None

    google_day = (datetime.now(APP_TIMEZONE).weekday() + 1) % 7
    windows = []
    for period in periods:
        open_info = period.get("open")
        close_info = period.get("close")
        if not open_info or not close_info:
            continue
        if open_info.get("day") != google_day or close_info.get("day") != google_day:
            continue
        windows.append(
            f"{format_day_time(int(open_info.get('hour', 0)), int(open_info.get('minute', 0)))} to "
            f"{format_day_time(int(close_info.get('hour', 0)), int(close_info.get('minute', 0)))}"
        )

    if windows:
        return "Today: " + "; ".join(windows)
    return None


def unmet_criteria_for_place(place: dict, state: ConversationState) -> List[str]:
    unmet = []
    if state.dish and place.get("menu_verification", {}).get("status") == "not_verified":
        unmet.append(f"could not verify {state.dish} from the restaurant website menu")
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
        bare_numeric_reply = bool(re.fullmatch(r"\d{1,3}(?:\.\d)?", cleaned))
        previous_dish = state.dish
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

        if state.manual_location and state.dish:
            normalized_location = normalize_text(state.manual_location)
            normalized_dish = normalize_text(state.dish)
            if normalized_location and (
                normalized_dish == normalized_location
                or normalized_location in normalized_dish
                or normalized_dish in normalized_location
            ):
                state.dish = None
                traces.append(AgentTrace(self.name, "cleanup", "Ignored location text that was mistakenly captured as a dish."))

        inferred_cuisine = infer_cuisine_from_dish(state.dish)
        if inferred_cuisine:
            state.cuisine = inferred_cuisine
            traces.append(AgentTrace(self.name, "inference", f"Inferred cuisine from dish: {state.cuisine}."))

        inferred_travel = infer_travel_preferences(message)
        if inferred_travel["travel_mode"]:
            state.travel_mode = inferred_travel["travel_mode"]
        if inferred_travel["min_travel_minutes"]:
            state.min_travel_minutes = inferred_travel["min_travel_minutes"]
        if inferred_travel["travel_minutes"] and not (bare_numeric_reply and state.travel_minutes is not None):
            state.travel_minutes = inferred_travel["travel_minutes"]
        if inferred_travel["search_radius_meters"] and not (bare_numeric_reply and state.travel_minutes is not None):
            state.search_radius_meters = inferred_travel["search_radius_meters"]
        if (
            inferred_travel["travel_mode"]
            or (inferred_travel["travel_minutes"] and not (bare_numeric_reply and state.travel_minutes is not None))
        ):
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

        if not state.when:
            state.when = "now"
            state.requested_day_offset = None
            state.requested_hour = None
            state.requested_minute = None
            traces.append(AgentTrace(self.name, "default", "Assumed the user wants to eat right now because no future timing was specified."))

        if (
            not state.user_location
            and not state.manual_location
            and (not inferred_dish or not inferred_cuisine)
            and inferred_rating is None
            and not inferred_travel["travel_mode"]
            and not inferred_travel["travel_minutes"]
            and not inferred_when
            and len(cleaned.split()) <= 6
        ):
            state.location_mode = "manual"
            state.manual_location = message.strip()
            if inferred_dish and not inferred_cuisine:
                state.dish = previous_dish
            traces.append(AgentTrace(self.name, "location_capture", "Treated the latest short reply as a manual location."))

        if state.location_mode == "manual" and state.manual_location and not state.user_location:
            state.user_location = geocode_location(state.manual_location)
            traces.append(AgentTrace(self.name, "geocode", "Resolved the manual location with Places text search."))

        state.conversation_turns += 1
        return traces


class ClarificationAgent:
    name = "Clarification Agent"

    def next_question(self, state: ConversationState) -> Optional[str]:
        if state.llm_next_question:
            if not question_requests_cuisine(state.llm_next_question):
                return state.llm_next_question
        if not state.dish and not state.cuisine:
            return "What would you like to eat?"
        if state.location_mode == "manual" and not state.manual_location:
            return "What location are you thinking of? You can type a place name or coordinates."
        if not state.location_mode and not state.user_location:
            return "Can I use your current location, or do you want to enter the location manually?"
        if not state.travel_minutes:
            if state.travel_mode:
                return "How long are you willing to travel?"
            return "How long are you willing to travel, and is that by walk, public transport, or car?"
        return None


class RetrievalAgent:
    name = "Retrieval Agent"

    def ready_for_search(self, state: ConversationState) -> bool:
        return bool(
            (state.cuisine or state.dish)
            and (state.user_location or state.location_mode)
            and (state.location_mode != "manual" or state.manual_location)
            and state.when
            and state.travel_minutes
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
                "places.rating,places.userRatingCount,places.primaryTypeDisplayName"
            ),
            max_result_count=max(limit * 4, 30),
            location_bias=location_bias,
        )

        results = []
        for place in places:
            details = place_details(
                place["id"],
                (
                    "id,displayName,formattedAddress,location,rating,userRatingCount,"
                    "primaryTypeDisplayName,currentOpeningHours,regularOpeningHours,reviewSummary,editorialSummary,"
                    "googleMapsUri,websiteUri,photos"
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
            if (
                state.travel_minutes is not None
                and estimated_travel_minutes is not None
                and estimated_travel_minutes > state.travel_minutes + TRAVEL_TIME_LEEWAY_MINUTES
            ):
                continue

            quality_rating = confidence_weighted_rating(
                details.get("rating"),
                details.get("userRatingCount"),
            )
            score = quality_rating * 3.0
            score += review_volume_bonus(details.get("userRatingCount"))
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
                "quality_rating": round(quality_rating, 3),
                "distance_miles": distance_miles,
                "estimated_travel_minutes": estimated_travel_minutes,
                "travel_mode_label": travel_mode_label(state.travel_mode),
                "travel_estimates": travel_estimates,
                "primary_type": details.get("primaryTypeDisplayName", {}).get("text"),
                "open_now": open_now,
                "open_at_requested_time": open_at_requested_time,
                "opening_hours_summary": opening_hours_summary(details),
                "summary": summary_text(details.get("reviewSummary"))
                or summary_text(details.get("editorialSummary")),
                "website_url": details.get("websiteUri"),
                "menu_verification": {
                    "status": "not_attempted",
                    "label": "verification not attempted yet",
                    "verified": False,
                    "source_url": details.get("websiteUri"),
                    "evidence": None,
                },
                "place_photos": details.get("photos") or [],
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
            results.append(result)

        ranked_results = sorted(results, key=lambda item: self._sort_key(item, state), reverse=True)
        verification_budget = min(len(ranked_results), max(limit, 3))
        if state.dish:
            for place in ranked_results[:verification_budget]:
                place["menu_verification"] = verify_dish_availability(
                    place.get("website_url"),
                    state.dish,
                    place.get("place_photos"),
                )
                if place["menu_verification"].get("verified"):
                    place["score"] = round(float(place.get("score") or 0.0) + 3.0, 2)
                elif place["menu_verification"].get("status") == "not_verified":
                    place["score"] = round(float(place.get("score") or 0.0) - 2.0, 2)

        for result in ranked_results:
            result["unmet_criteria"] = unmet_criteria_for_place(result, state)
            result["matched_criteria"] = self._matched_criteria(result, state)
            result["verification_status"] = verification_status_for_place(result, state)
            result["confidence_score"] = confidence_score_for_place(result, state)
            result["confidence_label"] = confidence_label(result["confidence_score"])
            result["photo_refs"] = [
                photo.get("name")
                for photo in (result.get("place_photos") or [])[:4]
                if isinstance(photo, dict) and photo.get("name")
            ]
            result.pop("place_photos", None)

        ranked_results = sorted(ranked_results, key=lambda item: self._sort_key(item, state), reverse=True)
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

    def _sort_key(self, place: dict, state: ConversationState) -> tuple:
        rating = float(place.get("rating") or 0.0)
        review_count = int(place.get("user_rating_count") or 0)
        quality_rating = float(place.get("quality_rating") or 0.0)
        score = float(place.get("score") or 0.0)
        return (score, quality_rating, min(review_count, 3000), rating)

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
        if ENABLE_GEMINI_FINAL_RESPONSE and gemini_enabled():
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
            line += f"\n   Fit: {fit_label_for_place(place)}"
            if place["matched_criteria"]:
                line += f"\n   Matches: {', '.join(place['matched_criteria'])}."
            if place["unmet_criteria"]:
                line += f"\n   Does not fully follow: {', '.join(place['unmet_criteria'])}."
            place_summary = summary_text(place.get("summary"))
            if place_summary:
                line += f"\n   Why it still stands out: {place_summary}"
            lines.append(line)
        lines.append("Ask for top ten recommendations if you want a longer list.")
        return "\n".join(lines)

    def build_place_follow_up_reply(self, state: ConversationState, place: dict) -> str:
        verification = place.get("menu_verification") or {}
        asking_about_dish = bool(state.dish and verification)
        lines = []

        if asking_about_dish:
            if verification.get("verified"):
                lines.append(f"{place['name']} looks like a good bet for {state.dish}.")
            elif verification.get("status") == "not_verified":
                lines.append(f"I could not verify that {place['name']} has {state.dish}.")
            else:
                lines.append(f"I can’t confidently confirm that {place['name']} has {state.dish}.")
        else:
            lines.append(f"Here’s more about {place['name']}:")

        overview_bits = []
        if place.get("address"):
            overview_bits.append(place["address"])
        if place.get("rating") is not None:
            overview_bits.append(f"rating {place['rating']}")
        if place.get("distance_miles") is not None:
            overview_bits.append(f"{place['distance_miles']} miles away")
        if overview_bits:
            lines.append("It is " + ", ".join(overview_bits) + ".")

        travel_estimate_text = format_travel_estimates(place.get("travel_estimates") or {})
        if travel_estimate_text:
            lines.append(f"Estimated travel: {travel_estimate_text}.")
        elif place.get("estimated_travel_minutes") is not None:
            lines.append(
                f"Estimated travel: about {place['estimated_travel_minutes']} minutes by {travel_mode_label(state.travel_mode)}."
            )

        if place.get("open_now") is True:
            lines.append("It appears to be open right now.")
        elif place.get("open_now") is False:
            lines.append("It does not appear to be open right now.")

        if place.get("opening_hours_summary"):
            lines.append(f"Hours: {place['opening_hours_summary']}")

        if state.when == "later" and place.get("open_at_requested_time") is True:
            target_dt = requested_datetime_for_state(state)
            if target_dt is not None:
                lines.append(f"It also looks likely to be open around {target_dt.strftime('%-I:%M %p').lower()} on {target_dt.strftime('%A')}.")

        place_summary = summary_text(place.get("summary"))
        if place_summary:
            lines.append(f"Why it stands out: {place_summary}")

        if place.get("matched_criteria"):
            lines.append(f"What works well: {', '.join(place['matched_criteria'])}.")

        if place.get("unmet_criteria"):
            lines.append(f"The main catch: {exact_reason_summary(place)}.")

        return "\n".join(lines)


class CoordinatorAgent:
    def __init__(self) -> None:
        self.intent_agent = IntentAgent()
        self.clarification_agent = ClarificationAgent()
        self.retrieval_agent = RetrievalAgent()
        self.response_agent = ResponseAgent()

    def handle_turn(
        self,
        session_id: str,
        state: ConversationState,
        message: str,
        browser_location: Optional[dict],
    ) -> Dict[str, Any]:
        traces: List[AgentTrace] = []
        append_message(state, "user", message)
        if state.title == "New chat" and message.strip():
            state.title = build_session_title(message)

        top_k_request = parse_top_k_request(message)
        if top_k_request and state.last_results:
            results, retrieval_traces = self.retrieval_agent.run(state, top_k_request)
            traces.extend(retrieval_traces)
            reply = self.response_agent.build_reply(state, results, top_k_request)
            append_message(state, "assistant", reply)
            return self._payload(session_id, state, reply, results, traces)

        follow_up_place = find_follow_up_place(message, state.last_results)
        if follow_up_place is not None:
            reply = self.response_agent.build_place_follow_up_reply(state, follow_up_place)
            traces.append(
                AgentTrace(
                    "Response Agent",
                    "place_follow_up",
                    f"Answered a follow-up question about {follow_up_place['name']} from the existing results.",
                )
            )
            append_message(state, "assistant", reply)
            return self._payload(session_id, state, reply, state.last_results, traces)
        if state.last_results and is_place_follow_up_request(message):
            reply = "I can help with that place, but I couldn't tell which recommendation you meant. Mention the restaurant name again, or say something like 'tell me about the second one.'"
            traces.append(
                AgentTrace(
                    "Clarification Agent",
                    "follow_up_clarify",
                    "Asked the user to clarify which existing recommendation they meant.",
                )
            )
            append_message(state, "assistant", reply)
            return self._payload(session_id, state, reply, state.last_results, traces)

        traces.extend(self.intent_agent.run(state, message, browser_location))
        question = self.clarification_agent.next_question(state)

        if not self.retrieval_agent.ready_for_search(state):
            reply = question or "Tell me a little more so I can narrow it down."
            traces.append(AgentTrace("Clarification Agent", "question", reply))
            append_message(state, "assistant", reply)
            return self._payload(session_id, state, reply, [], traces)

        results, retrieval_traces = self.retrieval_agent.run(state, 5)
        traces.extend(retrieval_traces)
        reply = self.response_agent.build_reply(state, results, 5)
        traces.append(AgentTrace("Response Agent", "compose", "Built the final ranked recommendation message."))
        append_message(state, "assistant", reply)
        return self._payload(session_id, state, reply, results, traces)

    def _payload(
        self,
        session_id: str,
        state: ConversationState,
        reply: str,
        results: List[dict],
        traces: List[AgentTrace],
    ) -> Dict[str, Any]:
        return {
            "reply": reply,
            "results": results,
            "messages": state.messages,
            "user_location": state.user_location,
            "needs_map": bool(results and state.user_location),
            "state": asdict(state),
            "session": session_summary(session_id, state),
            "agent_trace": [asdict(trace) for trace in traces],
            "project_summary": project_summary_payload(),
        }


coordinator = CoordinatorAgent()


def project_summary_payload() -> Dict[str, Any]:
    return {
        "title": "Agentic Dining Assistant",
        "architecture": [
            "Intent Agent: extracts slots like item, time, location, rating, and travel preference.",
            "Clarification Agent: asks only for missing information before search.",
            "Retrieval Agent: grounds results in live Google Places data and ranks candidates.",
            "Response Agent: explains why each recommendation matches or misses criteria.",
        ],
    }


@app.get("/")
def index():
    user = current_user()
    if not user:
        return render_template("auth.html", mode="signin")
    return render_template("index.html", current_user=user)


@app.get("/login")
def auth_entry():
    if current_user():
        return redirect(url_for("index"))
    return render_template("auth.html", mode="signin")


@app.get("/signup")
def signup_entry():
    if current_user():
        return redirect(url_for("index"))
    return render_template("auth.html", mode="signup")


@app.get("/forgot-password")
def forgot_password_entry():
    if current_user():
        return redirect(url_for("index"))
    return render_template("auth.html", mode="forgot")


@app.post("/forgot-password")
def forgot_password():
    if current_user():
        return redirect(url_for("index"))
    email = normalize_email(request.form.get("email", ""))
    generic_message = "If that email exists in InstaDine, a password reset link has been prepared."
    user = find_user_by_email(email) if email else None

    context: Dict[str, Any] = {
        "mode": "forgot",
        "forgot_success": generic_message,
        "forgot_email": email,
    }

    if user:
        token = create_password_reset_token(user["user_id"])
        reset_link = url_for("reset_password_entry", token=token, _external=True)
        masked_user_email = mask_email(user["email"])
        try:
            send_password_reset_email(user["email"], user.get("name") or "", reset_link)
            context["forgot_success"] = (
                "If that email exists in InstaDine, a password reset link has been sent"
                + (f" to {masked_user_email}." if masked_user_email else ".")
            )
        except Exception:
            if request.host.startswith("127.0.0.1") or request.host.startswith("localhost"):
                context["forgot_success"] = (
                    "Email sending is not configured yet."
                    + (f" For {masked_user_email}," if masked_user_email else "")
                    + " use the reset link below for local testing."
                )
                context["forgot_reset_link"] = reset_link
    return render_template("auth.html", **context)


@app.get("/reset-password/<token>")
def reset_password_entry(token: str):
    if current_user():
        return redirect(url_for("index"))
    token_row = fetch_password_reset_token(token)
    if not password_reset_token_is_valid(token_row):
        return render_template(
            "auth.html",
            mode="forgot",
            forgot_error="That reset link is invalid or has expired. Request a new one below.",
        ), 400
    return render_template("auth.html", mode="reset", reset_token=token)


@app.post("/login")
def login():
    email = normalize_email(request.form.get("email", ""))
    password = request.form.get("password", "")
    remember_me = request.form.get("remember_me") == "on"
    user = find_user_by_email(email)
    if not user or not check_password_hash(user.get("password_hash", ""), password):
        return render_template(
            "auth.html",
            mode="signin",
            login_error="Invalid email or password.",
            login_email=email,
        ), 401
    login_user(user["user_id"], remember_me)
    return redirect(url_for("index"))


@app.post("/signup")
def signup():
    name = " ".join((request.form.get("name", "")).strip().split())
    email = normalize_email(request.form.get("email", ""))
    password = request.form.get("password", "")
    remember_me = request.form.get("remember_me") == "on"

    if not name:
        return render_template("auth.html", mode="signup", signup_error="Please enter your name.", signup_email=email), 400
    if not email or "@" not in email:
        return render_template("auth.html", mode="signup", signup_error="Please enter a valid email.", signup_name=name), 400
    if len(password) < 8:
        return render_template(
            "auth.html",
            mode="signup",
            signup_error="Password must be at least 8 characters.",
            signup_name=name,
            signup_email=email,
        ), 400
    if find_user_by_email(email):
        return render_template(
            "auth.html",
            mode="signup",
            signup_error="An account with that email already exists.",
            signup_name=name,
            signup_email=email,
        ), 400

    user_id = create_user_record(name, email, generate_password_hash(password))
    login_user(user_id, remember_me)
    return redirect(url_for("index"))


@app.post("/reset-password/<token>")
def reset_password(token: str):
    if current_user():
        return redirect(url_for("index"))
    token_row = fetch_password_reset_token(token)
    if not password_reset_token_is_valid(token_row):
        return render_template(
            "auth.html",
            mode="forgot",
            forgot_error="That reset link is invalid or has expired. Request a new one below.",
        ), 400

    password = request.form.get("password", "")
    confirm_password = request.form.get("confirm_password", "")
    if len(password) < 8:
        return render_template(
            "auth.html",
            mode="reset",
            reset_token=token,
            reset_error="Password must be at least 8 characters.",
        ), 400
    if password != confirm_password:
        return render_template(
            "auth.html",
            mode="reset",
            reset_token=token,
            reset_error="Passwords do not match.",
        ), 400

    update_user_password(token_row["user_id"], generate_password_hash(password))
    mark_password_reset_token_used(token)
    masked_user_email = mask_email(token_row["email"])
    return render_template(
        "auth.html",
        mode="signin",
        login_success=(
            "Your password has been reset"
            + (f" for {masked_user_email}." if masked_user_email else ".")
            + " Sign in with the new password."
        ),
    )


@app.post("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("index"))


@app.post("/api/session")
@login_required
def create_session():
    user_id = current_user_id()
    if not user_id:
        return jsonify({"error": "Please sign in to continue."}), 401
    session_id, state = create_session_record(user_id)
    return jsonify({"session_id": session_id, "session": session_summary(session_id, state), "messages": state.messages})


@app.get("/api/sessions")
@login_required
def list_sessions():
    user_id = current_user_id()
    if not user_id:
        return jsonify({"error": "Please sign in to continue."}), 401
    sessions = list_sessions_for_user(user_id)
    return jsonify({"sessions": sessions})


@app.get("/api/session/<session_id>")
@login_required
def get_session(session_id: str):
    user_id = current_user_id()
    if not user_id:
        return jsonify({"error": "Please sign in to continue."}), 401
    state = load_session_state(user_id, session_id)
    if state is None:
        return jsonify({"error": "Session not found"}), 404
    return jsonify(
        {
            "session_id": session_id,
            "session": session_summary(session_id, state),
            "messages": state.messages,
            "results": state.last_results,
            "user_location": state.user_location,
            "agent_trace": [],
            "project_summary": project_summary_payload(),
        }
    )


@app.get("/api/place-photo")
@login_required
def get_place_photo():
    photo_name = (request.args.get("name") or "").strip()
    if not photo_name:
        return jsonify({"error": "Missing photo name"}), 400

    photo_payload = fetch_place_photo_content(photo_name, max_width_px=900)
    if not photo_payload:
        return jsonify({"error": "Photo not found"}), 404

    content, content_type = photo_payload
    return Response(
        content,
        mimetype=content_type or "image/jpeg",
        headers={"Cache-Control": "public, max-age=86400"},
    )


@app.delete("/api/session/<session_id>")
@login_required
def delete_session(session_id: str):
    user_id = current_user_id()
    if not user_id:
        return jsonify({"error": "Please sign in to continue."}), 401
    deleted = delete_session_record(user_id, session_id)
    if not deleted:
        return jsonify({"error": "Session not found"}), 404
    return jsonify({"deleted_session_id": session_id})


@app.post("/api/chat")
@login_required
def chat():
    payload = request.get_json(force=True)
    session_id = payload.get("session_id")
    if not session_id:
        return jsonify({"error": "Missing session_id"}), 400

    message = (payload.get("message") or "").strip()
    if not message:
        return jsonify({"error": "Missing message"}), 400

    browser_location = payload.get("browser_location")
    user_id = current_user_id()
    if not user_id:
        return jsonify({"error": "Please sign in to continue."}), 401
    state = get_state(user_id, session_id)

    try:
        response_payload = coordinator.handle_turn(session_id, state, message, browser_location)
        save_session_state(user_id, session_id, state)
        return jsonify(response_payload)
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
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", "5001")),
        debug=os.getenv("FLASK_DEBUG", "").lower() in {"1", "true", "yes", "on"},
    )
