"""
Internee.pk Intern Support Chatbot
-----------------------------------
A free-to-run GenAI-style support chatbot that answers intern questions
about tasks and policies, backed by a small NLU/retrieval engine
(see nlu_engine.py) instead of a paid OpenAI subscription.

Run:
    pip install -r requirements.txt
    python app.py
Then open http://localhost:5000
"""

import os
from flask import Flask, request, jsonify, render_template

from nlu_engine import InternNLU

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KB_PATH = os.path.join(BASE_DIR, "knowledge_base.json")

app = Flask(__name__)
nlu = InternNLU(KB_PATH)


@app.route("/")
def index():
    return render_template("index.html", categories=nlu.kb["categories"])


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()
    if not message:
        return jsonify({"error": "message is required"}), 400

    result = nlu.respond(message)
    return jsonify(result)


@app.route("/api/categories")
def categories():
    return jsonify(nlu.kb["categories"])


@app.route("/api/category/<category_id>")
def category_faqs(category_id):
    faqs = nlu.category_faqs(category_id)
    return jsonify([{"id": f["id"], "question": f["question"]} for f in faqs])


@app.route("/api/faq/<faq_id>")
def faq_answer(faq_id):
    for f in nlu.faqs:
        if f["id"] == faq_id:
            return jsonify({"question": f["question"], "answer": f["answer"], "category": f["category"]})
    return jsonify({"error": "not found"}), 404


@app.route("/healthz")
def healthz():
    return jsonify({"status": "ok", "faq_count": len(nlu.faqs)})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
