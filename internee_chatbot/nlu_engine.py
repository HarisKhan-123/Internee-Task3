"""
nlu_engine.py

A lightweight, fully-free Natural Language Understanding engine for the
Internee.pk intern support chatbot.

Design goals
------------
1. Zero-cost by default: uses scikit-learn's TF-IDF + cosine similarity for
   semantic-ish matching against the knowledge base. No paid API required.
2. Optional upgrade path: if a free-tier LLM API key is present in the
   environment (GROQ_API_KEY), the engine will use it to lightly rephrase
   the matched knowledge-base answer in a more conversational tone, and to
   handle small talk / greetings. If no key is set, the bot still works
   perfectly using pure retrieval - it just answers a bit more literally.

Why Groq instead of OpenAI
---------------------------
Groq offers a generous free tier for open models (e.g. Llama 3.1) with an
OpenAI-compatible chat completions API, so it's a drop-in free alternative
to the OpenAI API mentioned in the original task brief. Swapping in any
other OpenAI-compatible free endpoint (e.g. OpenRouter's free models,
Google AI Studio's Gemini free tier) only requires changing BASE_URL/MODEL
below.
"""

import os
import json
import difflib
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    import requests
except ImportError:  # requests is in requirements.txt, this is just a guard
    requests = None


# ---------------------------------------------------------------------------
# Optional free LLM configuration (Groq's free tier, OpenAI-compatible)
# ---------------------------------------------------------------------------
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
GROQ_BASE_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.1-8b-instant"

CONFIDENT_THRESHOLD = 0.32   # above this -> treat as a strong FAQ match
WEAK_THRESHOLD = 0.15        # below this -> treat as "no good match"

GREETINGS = {"hi", "hello", "hey", "salam", "assalam o alaikum", "asalam o alaikum",
             "hii", "helloo", "yo", "good morning", "good evening", "good afternoon"}
THANKS = {"thanks", "thank you", "shukriya", "thankyou", "thnx", "thx"}


@dataclass
class MatchResult:
    answer: str
    matched_question: str
    category: str
    score: float
    faq_id: str | None


class InternNLU:
    """Retrieval-based NLU over the intern knowledge base, with an optional
    LLM polish step."""

    def __init__(self, knowledge_base_path: str):
        with open(knowledge_base_path, "r", encoding="utf-8") as f:
            self.kb = json.load(f)

        self.faqs = self.kb["faqs"]
        self.categories = {c["id"]: c for c in self.kb["categories"]}

        # Build the corpus: each FAQ contributes its canonical question plus
        # all phrasing variants, so the vectorizer learns the many ways an
        # intern might ask the same thing (a cheap stand-in for a trained
        # intent classifier like Rasa/Dialogflow would use).
        self._corpus = []
        self._corpus_faq_index = []
        for i, faq in enumerate(self.faqs):
            texts = [faq["question"]] + faq.get("variants", [])
            for t in texts:
                self._corpus.append(t.lower())
                self._corpus_faq_index.append(i)

        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            stop_words="english",
        )
        self._matrix = self.vectorizer.fit_transform(self._corpus)

    # -- Public API ---------------------------------------------------------

    def is_greeting(self, text: str) -> bool:
        t = text.lower().strip(" !.?")
        return t in GREETINGS or any(t.startswith(g) for g in GREETINGS)

    def is_thanks(self, text: str) -> bool:
        t = text.lower().strip(" !.?")
        return t in THANKS or any(g in t for g in THANKS)

    def find_best_match(self, user_message: str) -> MatchResult:
        """Return the best-matching FAQ for a user message using TF-IDF
        cosine similarity, with a difflib fallback for short/typo-heavy
        queries where TF-IDF struggles."""
        query = user_message.lower().strip()
        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self._matrix)[0]

        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])
        faq_idx = self._corpus_faq_index[best_idx]

        # Fallback: fuzzy string match against canonical questions, useful
        # for short queries ("deadline?", "certificate") that TF-IDF alone
        # scores poorly due to sparse overlap.
        if best_score < WEAK_THRESHOLD:
            questions = [f["question"] for f in self.faqs]
            close = difflib.get_close_matches(query, [q.lower() for q in questions], n=1, cutoff=0.4)
            if close:
                alt_idx = [q.lower() for q in questions].index(close[0])
                alt_score = difflib.SequenceMatcher(None, query, close[0]).ratio()
                if alt_score > best_score:
                    faq_idx = alt_idx
                    best_score = max(best_score, WEAK_THRESHOLD + 0.01)

        faq = self.faqs[faq_idx]
        return MatchResult(
            answer=faq["answer"],
            matched_question=faq["question"],
            category=faq["category"],
            score=best_score,
            faq_id=faq["id"],
        )

    def category_faqs(self, category_id: str):
        return [f for f in self.faqs if f["category"] == category_id]

    def respond(self, user_message: str) -> dict:
        """Top-level entry point used by the Flask app. Returns a dict ready
        to be JSON-serialized to the frontend."""
        text = user_message.strip()
        if not text:
            return self._package("Could you type your question? I'm listening!", None, 1.0, "chitchat")

        if self.is_greeting(text):
            msg = ("Hey there! I'm the Internee.pk intern assistant. Ask me about tasks, "
                   "deadlines, grading, certificates, or program policies.")
            return self._package(msg, None, 1.0, "chitchat")

        if self.is_thanks(text):
            return self._package("Anytime! Let me know if anything else comes up.", None, 1.0, "chitchat")

        match = self.find_best_match(text)

        if match.score >= WEAK_THRESHOLD:
            answer = match.answer
            if GROQ_API_KEY:
                polished = self._polish_with_llm(text, match.answer)
                if polished:
                    answer = polished
            return self._package(answer, match.faq_id, match.score, match.category,
                                  matched_question=match.matched_question)

        # No good match at all - be honest instead of guessing.
        fallback = (
            "I couldn't find a confident answer to that in the intern knowledge base yet. "
            "Try rephrasing, pick a topic below, or open a support ticket and a mentor will follow up."
        )
        return self._package(fallback, None, match.score, "unknown")

    # -- Internal helpers -----------------------------------------------------

    def _package(self, answer, faq_id, score, category, matched_question=None):
        return {
            "answer": answer,
            "faq_id": faq_id,
            "confidence": round(score, 3),
            "category": category,
            "matched_question": matched_question,
        }

    def _polish_with_llm(self, user_message: str, kb_answer: str) -> str | None:
        """Optional step: ask a free-tier LLM (Groq/Llama) to rephrase the
        knowledge-base answer conversationally, grounded strictly in the KB
        text so it can't hallucinate new policy details. Silently returns
        None on any failure so the bot always still works without it."""
        if requests is None:
            return None
        try:
            payload = {
                "model": GROQ_MODEL,
                "temperature": 0.4,
                "max_tokens": 200,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful intern support assistant for Internee.pk. "
                            "Rephrase the given official answer in a warm, concise way. "
                            "Do not invent facts, numbers, or policies that are not in the "
                            "provided answer. Keep it under 60 words."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Intern asked: {user_message}\nOfficial answer: {kb_answer}\n"
                                   f"Rephrase this answer conversationally.",
                    },
                ],
            }
            resp = requests.post(
                GROQ_BASE_URL,
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=8,
            )
            if resp.status_code == 200:
                data = resp.json()
                return data["choices"][0]["message"]["content"].strip()
        except Exception:
            pass
        return None
