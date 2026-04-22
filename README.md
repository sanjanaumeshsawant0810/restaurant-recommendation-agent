# Agentic Dining Assistant

An agentic AI restaurant recommendation product with a runnable web interface. The system asks for missing user constraints, uses Gemini for conversational slot-filling and response generation when configured, grounds recommendations in live Google Places data, verifies requested dishes against restaurant website menus when possible, explains why each place matches or misses the user request, and displays the results on a map.

## Why this fits the course

This project is intentionally framed as an **agentic AI product**, not just a restaurant finder.

- **Intent Agent** extracts structured constraints from messy user language, with Gemini-backed understanding when available.
- **Clarification Agent** decides what missing question to ask next.
- **Retrieval Agent** calls live APIs, ranks places, and grounds the system in external data.
- **Response Agent** converts ranked results into user-facing explanations with unmet criteria, with Gemini-backed phrasing when available.

That gives you:

- multi-turn agentic behavior
- API-based action taking
- grounded live retrieval
- a runnable product interface
- visible reasoning and explainability for demo day

## Product behavior

- Chat-style frontend similar to ChatGPT
- Top 5 recommendations by default
- User can ask for top 10 recommendations
- Live map view with directions links
- Uses place names, decimal coordinates, or DMS coordinates for location input
- Considers travel-time preferences
- Tries to verify requested dishes from restaurant websites, menu PDFs, and optional OCR on image menus
- Shows matched criteria and unmet criteria for each place

## Architecture

### Backend

- `app.py`: Flask app plus the explicit agent pipeline
- `places_restaurant_chatbot.py`: shared Places helper functions

### Frontend

- `templates/index.html`: UI shell
- `static/app.js`: browser chat logic, map rendering, agent trace rendering
- `static/style.css`: visual styling

## How to run

```bash
cd "/Users/sanjanasawant/Desktop/AI and LLM"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export GOOGLE_MAPS_API_KEY="your_key_here"
export GEMINI_API_KEY="your_key_here"
python3 app.py
```

Then open:

`http://127.0.0.1:5001`

If `GEMINI_API_KEY` is missing, the app still runs with the rule-based fallback path. If `GOOGLE_MAPS_API_KEY` is missing, live place search will fail.
If you want OCR for image-based menus, install the Tesseract system binary as well. Without it, the app still verifies HTML pages and PDF menus.

## Demo ideas

Try prompts like:

- `I want pizza near Times Square and I can walk for 15 minutes.`
- `I want dosa now and I can drive 20 minutes from 40.754314, -73.977541.`
- `I want coffee for a date and show me top ten recommendations.`

## Suggested report framing

Frame the report around:

1. Problem and target users
2. Agent architecture
3. External grounding with Places API
4. Ranking and constraint handling
5. UX and explainability
6. Limitations and future work

## Good future upgrades

- Add Gemini for richer intent extraction and response generation
- Add LangGraph to formalize the multi-agent workflow
- Add saved sessions or user profiles
- Add a RAG layer with neighborhood guides, menus, or dining notes
