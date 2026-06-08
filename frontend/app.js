// ============================================================
//  HR AI Chatbot - chat client
// ============================================================
const token = localStorage.getItem("hr_token");
const user = JSON.parse(localStorage.getItem("hr_user") || "null");

if (!token || !user) {
  window.location.href = "/";
}

const messagesEl = document.getElementById("messages");
const emptyState = document.getElementById("empty-state");
const inputEl = document.getElementById("input");
const sendBtn = document.getElementById("send-btn");

// --- header / user chip ---
const roleBadge = document.getElementById("role-badge");
if (user) {
  if (user.role === "admin") {
    document.getElementById("user-label").textContent = "Administrator";
    roleBadge.textContent = "ADMIN";
    roleBadge.classList.add("admin");
  } else {
    document.getElementById("user-label").textContent = "Employee #" + user.employee_id;
    roleBadge.textContent = "EMPLOYEE";
  }
}

document.getElementById("logout-btn").addEventListener("click", () => {
  localStorage.clear();
  window.location.href = "/";
});

// session id persisted for conversation memory continuity
let sessionId = localStorage.getItem("hr_session");
if (!sessionId) {
  sessionId = "sess-" + Math.random().toString(36).slice(2) + Date.now().toString(36);
  localStorage.setItem("hr_session", sessionId);
}

// --- helpers ---
function escapeHtml(s) {
  return s
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;").replace(/'/g, "&#039;");
}

// Render a result table (array of row objects) into HTML
function renderTable(rows) {
  if (!rows || !rows.length) return "";
  const cols = Object.keys(rows[0]);
  let html = "<table><thead><tr>";
  cols.forEach((c) => (html += `<th>${escapeHtml(String(c))}</th>`));
  html += "</tr></thead><tbody>";
  rows.forEach((r) => {
    html += "<tr>";
    cols.forEach((c) => (html += `<td>${escapeHtml(String(r[c] ?? ""))}</td>`));
    html += "</tr>";
  });
  html += "</tbody></table>";
  return html;
}

function addMessage(role, text, extra = {}) {
  emptyState.style.display = "none";
  const wrap = document.createElement("div");
  wrap.className = "msg " + role;

  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = role === "user" ? "U" : "AI";

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.innerHTML = escapeHtml(text || "");

  if (extra.table) bubble.innerHTML += renderTable(extra.table);

  if (extra.sources && extra.sources.length) {
    const src = document.createElement("div");
    src.className = "sources";
    src.innerHTML = "Sources: " + extra.sources.map((s) => escapeHtml(s)).join(", ");
    bubble.appendChild(src);
  }

  if (extra.meta) {
    const meta = document.createElement("div");
    meta.className = "meta";
    const tags = [];
    if (extra.meta.intent) tags.push("intent: " + extra.meta.intent);
    if (extra.meta.agent) tags.push("agent: " + extra.meta.agent);
    if (typeof extra.meta.confidence === "number")
      tags.push("confidence: " + Math.round(extra.meta.confidence * 100) + "%");
    if (extra.meta.latency_ms) tags.push(extra.meta.latency_ms + " ms");
    meta.innerHTML = tags.map((t) => `<span class="tag">${escapeHtml(t)}</span>`).join("");
    bubble.appendChild(meta);
  }

  wrap.appendChild(avatar);
  wrap.appendChild(bubble);
  messagesEl.appendChild(wrap);
  scrollToBottom();
  return bubble;
}

function addTyping() {
  emptyState.style.display = "none";
  const wrap = document.createElement("div");
  wrap.className = "msg bot";
  wrap.id = "typing-indicator";
  wrap.innerHTML =
    '<div class="avatar">AI</div><div class="bubble"><div class="typing"><span></span><span></span><span></span></div></div>';
  messagesEl.appendChild(wrap);
  scrollToBottom();
}
function removeTyping() {
  const t = document.getElementById("typing-indicator");
  if (t) t.remove();
}
function scrollToBottom() {
  const area = document.getElementById("chat-area");
  area.scrollTop = area.scrollHeight;
}

// --- send ---
let sending = false;
async function send(text) {
  if (sending || !text.trim()) return;
  sending = true;
  sendBtn.disabled = true;
  addMessage("user", text);
  inputEl.value = "";
  inputEl.style.height = "auto";
  addTyping();

  const t0 = performance.now();
  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: "Bearer " + token,
      },
      body: JSON.stringify({ message: text, session_id: sessionId }),
    });
    if (res.status === 401) {
      localStorage.clear();
      window.location.href = "/";
      return;
    }
    const data = await res.json();
    removeTyping();
    if (!res.ok) {
      addMessage("bot", data.detail || "Something went wrong.");
      return;
    }
    const latency = data.latency_ms || Math.round(performance.now() - t0);
    addMessage("bot", data.answer, {
      table: data.table,
      sources: data.sources,
      meta: {
        intent: data.intent,
        agent: data.agent,
        confidence: data.confidence,
        latency_ms: latency,
      },
    });
  } catch (err) {
    removeTyping();
    addMessage("bot", "Network error: " + err.message);
  } finally {
    sending = false;
    sendBtn.disabled = false;
    inputEl.focus();
  }
}

sendBtn.addEventListener("click", () => send(inputEl.value));
inputEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    send(inputEl.value);
  }
});
inputEl.addEventListener("input", () => {
  inputEl.style.height = "auto";
  inputEl.style.height = Math.min(inputEl.scrollHeight, 160) + "px";
});

document.getElementById("suggestions").addEventListener("click", (e) => {
  if (e.target.tagName === "BUTTON") send(e.target.textContent);
});

inputEl.focus();
