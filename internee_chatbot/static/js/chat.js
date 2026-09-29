const chatLog = document.getElementById("chatLog");
const composer = document.getElementById("composer");
const input = document.getElementById("messageInput");
const quickRow = document.getElementById("quickRow");
const categoryNav = document.getElementById("categoryNav");

function scrollToBottom() {
  chatLog.scrollTop = chatLog.scrollHeight;
}

function addMessage(text, sender, meta) {
  const wrap = document.createElement("div");
  wrap.className = `msg ${sender}`;
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.innerHTML = text;
  if (meta) {
    const metaEl = document.createElement("span");
    metaEl.className = "meta";
    metaEl.textContent = meta;
    bubble.appendChild(metaEl);
  }
  wrap.appendChild(bubble);
  chatLog.appendChild(wrap);
  scrollToBottom();
  return wrap;
}

function addTyping() {
  const wrap = document.createElement("div");
  wrap.className = "msg bot";
  wrap.id = "typingIndicator";
  wrap.innerHTML = `<div class="bubble"><div class="typing"><span></span><span></span><span></span></div></div>`;
  chatLog.appendChild(wrap);
  scrollToBottom();
}

function removeTyping() {
  const el = document.getElementById("typingIndicator");
  if (el) el.remove();
}

async function sendMessage(text) {
  if (!text.trim()) return;
  addMessage(escapeHtml(text), "user");
  input.value = "";
  addTyping();

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text }),
    });
    const data = await res.json();
    removeTyping();

    let confidenceNote = "";
    if (data.category && data.category !== "chitchat" && data.category !== "unknown") {
      confidenceNote = `Topic: ${data.category}`;
    }
    addMessage(escapeHtml(data.answer), "bot", confidenceNote);
  } catch (err) {
    removeTyping();
    addMessage("Sorry, I couldn't reach the server. Please try again.", "bot");
  }
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

composer.addEventListener("submit", (e) => {
  e.preventDefault();
  sendMessage(input.value);
});

quickRow.addEventListener("click", (e) => {
  if (e.target.classList.contains("chip")) {
    sendMessage(e.target.textContent);
  }
});

categoryNav.addEventListener("click", async (e) => {
  if (!e.target.classList.contains("rail-item")) return;

  document.querySelectorAll(".rail-item").forEach((b) => b.classList.remove("active"));
  e.target.classList.add("active");

  const categoryId = e.target.dataset.category;
  addTyping();
  try {
    const res = await fetch(`/api/category/${categoryId}`);
    const items = await res.json();
    removeTyping();

    if (!items.length) {
      addMessage("No FAQs found in this topic yet.", "bot");
      return;
    }

    const list = items
      .map((f) => `<li data-faq="${f.id}">${escapeHtml(f.question)}</li>`)
      .join("");
    const wrap = addMessage(
      `Here's what I have under <em>${e.target.textContent}</em>:<ul class="faq-list">${list}</ul>`,
      "bot"
    );

    wrap.querySelectorAll("[data-faq]").forEach((li) => {
      li.addEventListener("click", async () => {
        const faqId = li.dataset.faq;
        addTyping();
        const r = await fetch(`/api/faq/${faqId}`);
        const faq = await r.json();
        removeTyping();
        addMessage(escapeHtml(faq.answer), "bot", `Topic: ${faq.category}`);
      });
    });
  } catch (err) {
    removeTyping();
    addMessage("Couldn't load that topic right now.", "bot");
  }
});
