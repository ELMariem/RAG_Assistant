// Thin wrapper around the FastAPI backend (main.py). Every function here maps to
// exactly one endpoint. Components call these, never fetch() directly -- if the
// backend's shape changes, only this file needs to change.

const TOKEN_KEY = "itgate_token";
const GROQ_KEY_STORAGE = "itgate_groq_api_key";

export function getToken() {
  return localStorage.getItem(TOKEN_KEY);
}
function setToken(token) {
  localStorage.setItem(TOKEN_KEY, token);
}
export function clearToken() {
  localStorage.removeItem(TOKEN_KEY);
}

// The user's own Groq key, kept ONLY in this browser (never synced to an account,
// never persisted server-side) -- see getGroqApiKey() usage in ask()/askStream(),
// which attach it to the request only when backend === "groq".
export function getGroqApiKey() {
  return localStorage.getItem(GROQ_KEY_STORAGE) || "";
}
export function setGroqApiKey(key) {
  if (key) {
    localStorage.setItem(GROQ_KEY_STORAGE, key);
  } else {
    localStorage.removeItem(GROQ_KEY_STORAGE);
  }
}

async function authFetch(path, options = {}) {
  const token = getToken();
  const headers = {
    ...(options.headers || {}),
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
  };
  const res = await fetch(path, { ...options, headers });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed: ${res.status}`);
  }
  return res;
}

// ---- Auth (auth.py / main.py) ----

export async function register(userId, password) {
  const res = await fetch("/auth/register", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ user_id: userId, password }),
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || "Registration failed");
  }
  return res.json();
}

export async function login(userId, password) {
  const res = await fetch("/auth/login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ user_id: userId, password }),
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || "Invalid username or password");
  }
  const data = await res.json();
  setToken(data.access_token);
  return data;
}

export async function me() {
  const res = await authFetch("/auth/me");
  return res.json();
}

// ---- Chat (rag.py via /ask, /ask_stream) ----

export async function ask(question, { backend, conversationId } = {}) {
  const res = await authFetch("/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      user_id: "web",              // ignored server-side (current_user comes from the JWT), but AskRequest requires the field
      question,
      backend: backend || null,
      conversation_id: conversationId ?? null,
      groq_api_key: backend === "groq" ? getGroqApiKey() || null : null,
    }),
  });
  return res.json(); // { answer, backend_used, conversation_id, sources: [{file, page}] }
}

// Streaming variant: reads the SSE-style "data: {...}\n\n" stream from /ask_stream
// and calls onToken(text) for each chunk as it arrives, so the UI can render tokens
// as they're generated instead of waiting for the full answer.
export async function askStream(question, { backend, conversationId } = {}, onToken) {
  const token = getToken();
  const res = await fetch("/ask_stream", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    body: JSON.stringify({
      user_id: "web",
      question,
      backend: backend || null,
      conversation_id: conversationId ?? null,
      groq_api_key: backend === "groq" ? getGroqApiKey() || null : null,
    }),
  });
  if (!res.ok || !res.body) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || "Streaming request failed");
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let finalMeta = null;

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // Events are separated by a blank line ("\n\n"), each line prefixed "data: "
    const events = buffer.split("\n\n");
    buffer = events.pop(); // last (possibly incomplete) chunk stays in the buffer
    for (const evt of events) {
      const line = evt.replace(/^data:\s*/, "");
      if (!line) continue;
      const payload = JSON.parse(line);
      if (payload.token) onToken(payload.token);
      if (payload.done) finalMeta = payload;
    }
  }
  return finalMeta; // { done, conversation_id, backend_used, sources: [{file, page}] }
}

// ---- Documents (ingest.py via /ingest, /documents) ----

export async function listDocuments() {
  const res = await authFetch("/documents");
  return res.json(); // { documents: [{ filename, indexed }] }
}

export async function uploadDocument(file) {
  const formData = new FormData();
  formData.append("file", file);
  const res = await authFetch("/ingest", { method: "POST", body: formData });
  return res.json(); // { filename, blocks_created, message }
}

export async function deleteDocument(filename) {
  const res = await authFetch(`/documents/${encodeURIComponent(filename)}`, { method: "DELETE" });
  return res.json();
}

// Diagram images are served behind auth (/documents/image/{filename}), so a plain
// <img src="..."> can't reach them -- no way to attach an Authorization header to
// an <img> request. Instead we fetch the bytes ourselves and hand back a local
// blob: URL the <img> tag CAN use. Caller is responsible for revoking it
// (URL.revokeObjectURL) once the image is no longer displayed, to avoid leaking memory.
export async function fetchDocumentImageUrl(filename) {
  const res = await authFetch(`/documents/image/${encodeURIComponent(filename)}`);
  const blob = await res.blob();
  return URL.createObjectURL(blob);
}

// ---- Conversations (memory.py via /conversations) ----

export async function listConversations() {
  const res = await authFetch("/conversations");
  return res.json(); // [{ conversation_id, started_at, preview }]
}

export async function newConversation() {
  const res = await authFetch("/new_conversation", { method: "POST" });
  return res.json(); // { conversation_id, message }
}

export async function getConversationMessages(conversationId) {
  const res = await authFetch(`/conversations/${conversationId}/messages`);
  return res.json(); // [{ role, content }]
}

export async function clearConversation(conversationId) {
  const res = await authFetch(`/conversations/${conversationId}/clear`, { method: "POST" });
  return res.json();
}