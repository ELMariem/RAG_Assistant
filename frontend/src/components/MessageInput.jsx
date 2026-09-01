import { useState } from "react";
import { Send } from "lucide-react";
import * as api from "../api/client";
import GroqKeyModal from "./GroqKeyModal";

export default function MessageInput({ onSend, disabled, backend, onBackendChange }) {
  const [value, setValue] = useState("");
  const [showGroqModal, setShowGroqModal] = useState(false);

  function handleBackendSelect(e) {
    const newBackend = e.target.value;
    // Switching to Groq without a saved key: pause the switch and ask for one
    // via the popup instead of silently letting the next message fail.
    if (newBackend === "groq" && !api.getGroqApiKey()) {
      setShowGroqModal(true);
      return;
    }
    onBackendChange(newBackend);
  }

  function handleGroqKeySave(key) {
    api.setGroqApiKey(key);
    setShowGroqModal(false);
    onBackendChange("groq");
  }

  function handleGroqModalCancel() {
    setShowGroqModal(false);
    // No state to revert: onBackendChange was never called, so the <select>
    // (controlled by the `backend` prop) snaps back to its previous value on its own.
  }

  function handleSubmit(e) {
    e.preventDefault();
    const trimmed = value.trim();
    if (!trimmed || disabled) return;
    onSend(trimmed);
    setValue("");
  }

  return (
    <>
      <form onSubmit={handleSubmit} className="border-t border-gray-200 bg-white px-6 py-4">
        <div className="flex items-end gap-3 max-w-3xl mx-auto">
          <select
            value={backend}
            onChange={handleBackendSelect}
            className="text-xs text-muted border border-gray-200 rounded-lg px-2 py-2 shrink-0"
            title="Assistant backend"
          >
            <option value="ollama">Ollama</option>
            <option value="groq">Groq</option>
          </select>
          <textarea
            value={value}
            onChange={(e) => setValue(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) handleSubmit(e);
            }}
            placeholder={disabled ? "Generating response..." : "Ask something..."}
            disabled={disabled}
            rows={1}
            className="flex-1 resize-none rounded-xl border border-gray-200 px-4 py-2.5 text-sm
                       focus:outline-none focus:ring-2 focus:ring-accent/40 disabled:bg-gray-50 disabled:text-muted"
          />
          <button
            type="submit"
            disabled={disabled || !value.trim()}
            className="w-10 h-10 shrink-0 rounded-xl bg-accent hover:bg-accent-hover disabled:bg-gray-300
                       text-white flex items-center justify-center transition-colors"
          >
            <Send size={16} />
          </button>
        </div>
      </form>

      <GroqKeyModal open={showGroqModal} onCancel={handleGroqModalCancel} onSave={handleGroqKeySave} />
    </>
  );
}