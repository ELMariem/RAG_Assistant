import { useState } from "react";
import { KeyRound, Eye, EyeOff } from "lucide-react";
import Modal from "./Modal";

export default function GroqKeyModal({ open, onCancel, onSave }) {
  const [key, setKey] = useState("");
  const [showKey, setShowKey] = useState(false);

  function handleSave() {
    const trimmed = key.trim();
    if (!trimmed) return;
    onSave(trimmed);
    setKey("");
  }

  return (
    <Modal open={open} onClose={onCancel} title="Clé API Groq requise">
      <p className="text-sm text-muted mb-4">
        Entre ta clé API Groq pour utiliser ce backend. Elle est enregistrée uniquement dans ce
        navigateur.
      </p>

      <div className="relative mb-4">
        <KeyRound
          size={14}
          className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400 pointer-events-none"
        />
        <input
          type={showKey ? "text" : "password"}
          value={key}
          onChange={(e) => setKey(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") handleSave();
          }}
          placeholder="gsk_..."
          autoFocus
          autoComplete="off"
          className="w-full text-sm text-ink border border-gray-200 rounded-lg pl-9 pr-9 py-2.5
                     focus:outline-none focus:ring-2 focus:ring-accent/40"
        />
        <button
          type="button"
          onClick={() => setShowKey((s) => !s)}
          className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-ink"
          title={showKey ? "Masquer la clé" : "Afficher la clé"}
        >
          {showKey ? <EyeOff size={14} /> : <Eye size={14} />}
        </button>
      </div>

      <div className="flex justify-end gap-2">
        <button
          onClick={onCancel}
          className="px-4 py-2 text-sm rounded-lg text-muted hover:bg-canvas transition-colors"
        >
          Annuler
        </button>
        <button
          onClick={handleSave}
          disabled={!key.trim()}
          className="px-4 py-2 text-sm rounded-lg bg-accent hover:bg-accent-hover disabled:bg-gray-300
                     text-white transition-colors"
        >
          Enregistrer
        </button>
      </div>
    </Modal>
  );
}