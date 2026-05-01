# InstaDine

InstaDine is an agentic AI restaurant recommendation product with a runnable web interface. The system accepts natural language requests, asks for any missing constraints, grounds recommendations in live Google Places data, verifies requested dishes from menus when possible, keeps saved chats per signed-in user, and shows results on a live map.

## Live demo

- Public app: [https://restaurant-recommendation-agent-566313238923.us-east1.run.app](https://restaurant-recommendation-agent-566313238923.us-east1.run.app)
- Source code: [https://github.com/sanjanaumeshsawant0810/restaurant-recommendation-agent](https://github.com/sanjanaumeshsawant0810/restaurant-recommendation-agent)

## Why this fits the course

This project is intentionally designed as an **agentic AI product**, not just a restaurant search page.

- **Intent Agent** interprets messy natural language and updates structured conversation state.
- **Clarification Agent** decides which missing question to ask next.
- **Retrieval Agent** calls live APIs, ranks places, and applies user constraints.
- **Response Agent** turns ranked results into readable explanations and place-specific follow-ups.

That gives you:

- multi-turn agentic behavior
- API-based action taking
- grounded retrieval
- a runnable end-to-end product
- explainability through trace output and matched/missed criteria

## Product behavior

- Chat-style frontend
- Sign in, create account, logout, and remember-me support
- Separate saved chats for each user account
- Delete saved chats from the chat drawer
- Password reset flow with reset links
- Top 5 recommendations by default
- Support for top 10 recommendations on request
- Live map with markers, distance badges, and directions links
- Location input through place names or coordinates
- Timing preferences such as `now`, `tomorrow at 2 pm`, `evening`, and `afternoon`
- Travel preferences including ranges and mixed modes like walk or public transport
- Hard travel cap of the user’s preferred time plus 20 minutes
- Minimum rating support
- Dish verification from restaurant websites, menu PDFs, restaurant-site images, and Google Places photos via OCR when available
- Place-specific follow-up questions such as asking for more detail about one recommendation
- Agent trace hidden by default and available on demand

## Current architecture

### Backend

- `app.py`: Flask app, auth, chat/session storage, agent orchestration, ranking, and response logic
- `places_restaurant_chatbot.py`: Google Places helpers, Gemini integration, scraping, OCR, and utility functions

### Frontend

- `templates/index.html`: app shell
- `templates/auth.html`: sign-in, sign-up, forgot-password, and reset-password pages
- `static/app.js`: chat flow, map rendering, saved chat handling, and trace toggle behavior
- `static/style.css`: styling

## Storage

The app now uses **SQLite** rather than JSON files for active app data.

Main tables:

- `users`
- `chat_sessions`
- `chat_messages`
- `password_reset_tokens`

The SQLite database is created locally under `data/instadine.db`.

## How to run locally

1. Clone the repository:

```bash
git clone https://github.com/sanjanaumeshsawant0810/restaurant-recommendation-agent.git
```

2. Open a terminal and go to the project folder:

```bash
cd restaurant-recommendation-agent
```

3. Create a virtual environment:

```bash
python3 -m venv .venv
```

4. Activate it:

```bash
source .venv/bin/activate
```

5. Install dependencies:

```bash
pip install -r requirements.txt
```

6. Set environment variables:

```bash
export FLASK_SECRET_KEY="your_random_secret_here"
export GOOGLE_MAPS_API_KEY="your_google_maps_key"
export GEMINI_API_KEY="your_gemini_key"
```

Optional:

```bash
export FLASK_DEBUG=true
export ENABLE_GEMINI_FINAL_RESPONSE=true
```

Notes:

- If `GEMINI_API_KEY` is missing, the app falls back to rule-based interpretation.
- By default, the **final deployed response wording does not use Gemini**, because the local formatter is more reliable on constrained hosting. If you want Gemini to write the final recommendation text too, set `ENABLE_GEMINI_FINAL_RESPONSE=true`.

7. Start the app:

```bash
python3 app.py
```

Then open:

`http://127.0.0.1:5001`

## Password reset email

For local testing, if SMTP is not configured, the app shows a local reset link on the forgot-password page so you can still test the feature.

For real email delivery in deployment, configure:

```bash
SMTP_HOST
SMTP_PORT
SMTP_USERNAME
SMTP_PASSWORD
SMTP_FROM_EMAIL
SMTP_USE_TLS
```

Gmail SMTP works for a student-friendly setup if you use:

- a Gmail account
- 2-Step Verification
- a Gmail App Password

## Deploying on Render

Recommended setup:

- **Web Service**
- Build command:

```bash
pip install -r requirements.txt
```

- Start command:

```bash
gunicorn --bind 0.0.0.0:$PORT --timeout 120 app:app
```

Recommended Render environment variables:

```bash
FLASK_SECRET_KEY
GOOGLE_MAPS_API_KEY
GEMINI_API_KEY
```

Optional email env vars:

```bash
SMTP_HOST
SMTP_PORT
SMTP_USERNAME
SMTP_PASSWORD
SMTP_FROM_EMAIL
SMTP_USE_TLS
```

## Demo ideas

Try prompts like:

- `I want pizza near Times Square and I can walk for 15 minutes.`
- `I want gelato tomorrow at 2 pm near Columbus Circle and I can travel 10 minutes by subway.`
- `I want wine and fine dining near Marcus Garvey Park, and I can travel 5 to 10 minutes by walk or public transport.`
- `I want dosa now and I can drive 20 minutes from 40.754314, -73.977541.`
- `Show me the top 10 recommendations.`
- `Tell me more about the second one.`
- `Tell me about Le Rivage.`
- `Does Down Under Coffee have dalgona coffee?`

## Current limitations

- Dish verification depends on whether menu sources are accessible and fast enough
- OCR is helpful but imperfect for blurry or image-heavy menus
- Travel time is heuristic-based rather than true route-engine output
- Recommendation quality depends on Google Places data quality
- The deployed version favors stability over slow, fully LLM-written final responses

## Future work

- Add LangGraph to formalize multi-agent orchestration
- Add richer retrieval over menus, reviews, and dining guides
- Add stronger personalization and long-term user preferences
- Connect to a real routing API for better travel estimates
- Expand support for dietary restrictions and group-based planning
