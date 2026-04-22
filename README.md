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
- Multiple saved chats with a switchable chat history sidebar
- Top 5 recommendations by default
- User can ask for top 10 recommendations
- Live map view with directions links
- Uses place names, decimal coordinates, or DMS coordinates for location input
- Considers travel-time preferences, including ranges and mixed modes like walk or public transport
- Captures budget preferences such as `$`, `$$`, `$$$`, and `$$$$`
- Tries to verify requested dishes from restaurant websites, menu PDFs, and optional OCR on image menus
- Shows matched criteria and unmet criteria for each place
- Labels results with verification status and confidence instead of overclaiming

## Architecture

### Backend

- `app.py`: Flask app plus the explicit agent pipeline
- `places_restaurant_chatbot.py`: shared Places helper functions

### Frontend

- `templates/index.html`: UI shell
- `static/app.js`: browser chat logic, map rendering, agent trace rendering
- `static/style.css`: visual styling

## How to run

1. Open a terminal and go to the project folder:

```bash
cd "/Users/sanjanasawant/Desktop/AI and LLM"
```

2. Create a virtual environment:

```bash
python3 -m venv .venv
```

3. Activate the virtual environment:

```bash
source .venv/bin/activate
```

4. Install dependencies:

```bash
pip install -r requirements.txt
```

5. Set your API keys:

```bash
export GOOGLE_MAPS_API_KEY="your_key_here"
export GEMINI_API_KEY="your_key_here"
```

If you do not want to use Gemini, you can leave `GEMINI_API_KEY` unset and the app will fall back to the rule-based path.

6. Start the app:

```bash
python3 app.py
```

Then open:

`http://127.0.0.1:5001`

If `GEMINI_API_KEY` is missing, the app still runs with the rule-based fallback path. If `GOOGLE_MAPS_API_KEY` is missing, live place search will fail.
If you want OCR for image-based menus, install the Tesseract system binary as well. Without it, the app still verifies HTML pages and PDF menus.
Saved chats are stored locally on disk in `data/chat_sessions.json`.

## Demo ideas

Try prompts like:

- `I want pizza near Times Square and I can walk for 15 minutes.`
- `I want dosa now and I can drive 20 minutes from 40.754314, -73.977541.`
- `I want coffee for a date and show me top ten recommendations.`
- `I want wine and fine dining near Marcus Garvey Park, and I can travel 5 to 10 minutes by walk or public transport.`
- `I want croissants tomorrow afternoon near Columbus Circle and my budget is $$ to $$$.`

## Suggested report framing

Frame the report around:

1. Problem and target users
2. Agent architecture
3. External grounding with Places API
4. Ranking and constraint handling
5. UX and explainability
6. Limitations and future work

## Good future upgrades

- Add LangGraph to formalize the multi-agent workflow
- Add real user profiles and account-linked history beyond local saved chats
- Expand the current retrieval and menu-verification pipeline into a fuller RAG layer with indexed neighborhood guides, menus, and dining notes
