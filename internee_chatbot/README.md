# Intern Desk — Internee.pk Support Chatbot

A GenAI-style chatbot that answers intern questions about **tasks and
policies**, built entirely with **free tools** (no OpenAI subscription
required).

## How this maps to the original brief

| Brief item | What this project uses |
|---|---|
| NLU | TF-IDF + cosine similarity retrieval engine (`nlu_engine.py`), with a fuzzy-match fallback for short/typo-heavy queries |
| Connect to knowledge base | `knowledge_base.json` — 20 FAQ entries across 6 categories (onboarding, tasks, deadlines, grading/certificates, policies, support) |
| Support FAQs and task guidelines | Sidebar topic browser + free-text chat, both backed by the same KB |
| Tools: Rasa / Dialogflow / OpenAI API / Flask | **Flask** backend + a lightweight custom NLU (scikit-learn) instead of Rasa/Dialogflow, which avoids their heavier setup; **OpenAI replaced with Groq's free tier** (optional, see below) |
| 24/7 AI chatbot for interns | Runs as a normal Flask web app — deploy anywhere (Render, Railway, PythonAnywhere free tiers, or your own server) to keep it up around the clock |

## Why this is free

- **No OpenAI key needed.** The chatbot works out of the box using pure
  retrieval: it matches the intern's question against the knowledge base
  with TF-IDF vectors and returns the matching official answer. This costs
  nothing and needs no API key.
- **Optional LLM polish, still free.** If you want more conversational
  phrasing, set a `GROQ_API_KEY` environment variable (Groq's free tier
  gives generous rate limits for Llama 3.1 models via an OpenAI-compatible
  API). If the key isn't set, the bot silently falls back to plain
  retrieval — nothing breaks.

## Project structure

```
internee_chatbot/
├── app.py                # Flask routes (/, /api/chat, /api/categories, ...)
├── nlu_engine.py          # TF-IDF matching + optional Groq polish step
├── knowledge_base.json    # FAQs, task guidelines, policies (edit this to expand the bot)
├── requirements.txt
├── templates/
│   └── index.html         # Chat UI
└── static/
    ├── css/style.css
    └── js/chat.js
```

## Running it locally

```bash
cd internee_chatbot
pip install -r requirements.txt
python app.py
```

Open **http://localhost:5000** — you'll see the chat UI with a topic
sidebar on the left and a chat box on the right.

Optional (free) LLM polish:

```bash
export GROQ_API_KEY="your-free-groq-key"   # from https://console.groq.com
python app.py
```

## Extending the knowledge base

Open `knowledge_base.json` and add a new object to the `faqs` array:

```json
{
  "id": "kb021",
  "category": "tasks",
  "question": "Can I switch my track mid-program?",
  "variants": ["change my domain", "switch track"],
  "answer": "..."
}
```

The `variants` list is what makes the NLU robust to different phrasings —
add a few realistic ways an intern might ask the same question. No
retraining step is needed; the vectorizer rebuilds automatically each time
the app starts.

## API reference

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/chat` | POST `{ "message": "..." }` | Main chat endpoint, returns the best-matching answer |
| `/api/categories` | GET | List of topic categories for the sidebar |
| `/api/category/<id>` | GET | FAQ list within one category |
| `/api/faq/<id>` | GET | Full answer for a specific FAQ id |
| `/healthz` | GET | Health check |

## Deploying for 24/7 uptime (free options)

- **Render** (free web service tier) or **Railway** — point them at this
  repo, they auto-detect Flask via `requirements.txt` + `app.py`.
- **PythonAnywhere** free tier also works well for small Flask apps.
- Set `GROQ_API_KEY` as an environment variable on whichever platform you
  pick if you want the optional LLM polish step.

## Notes / limitations

- This uses retrieval over a fixed knowledge base rather than a full
  open-domain LLM, so answers are always grounded in `knowledge_base.json`
  — it won't hallucinate new policies, but it also won't answer questions
  totally outside the KB (it says so honestly instead of guessing).
- For a closer NLU/intent-classifier feel similar to Rasa/Dialogflow, the
  `variants` field on each FAQ acts as labeled training phrases per intent
  — add more variants over time as you see real intern questions come in.
