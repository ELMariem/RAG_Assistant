import { useState } from "react";
import { Cpu, Cloud, KeyRound, Eye, EyeOff, Check } from "lucide-react";
import Sidebar from "../components/Sidebar";
import * as api from "../api/client";

// There's no dedicated backend endpoint for "features" yet -- config.py's LLM_BACKEND
// options (ollama/groq) are the closest thing, so this surfaces that choice with an
// explanation, rather than inventing settings the backend doesn't actually support.
export default function AIFeatures() {
  const [key, setKey] = useState(() => api.getGroqApiKey());
  const [showKey, setShowKey] = useState(false);
  const [saved, setSaved] = useState(false);

  function handleSave() {
    api.setGroqApiKey(key.trim());
    setSaved(true);
    setTimeout(() => setSaved(false), 1500);
  }

  return (
    <div className="flex h-screen bg-canvas">
      <Sidebar conversations={[]} onNewChat={() => {}} onSelectChat={() => {}} />
      <div className="flex-1 flex flex-col min-w-0">
        <div className="px-6 py-4 bg-white border-b border-gray-200">
          <h1 className="text-sm font-medium text-ink">AI Features</h1>
        </div>
        <div className="flex-1 overflow-y-auto px-6 py-6">
          <div className="max-w-2xl mx-auto flex flex-col gap-3">
            <div className="bg-white rounded-xl px-5 py-4 shadow-sm flex items-start gap-4">
              <div className="w-9 h-9 rounded-lg bg-accent/10 flex items-center justify-center shrink-0">
                <Cpu size={18} className="text-accent" />
              </div>
              <div>
                <h2 className="text-sm font-medium text-ink">Ollama (local)</h2>
                <p className="text-xs text-muted mt-1">
                  Runs entirely on your machine -- private, no data leaves your network. Select it from the
                  dropdown next to the message box in Chat Assistant.
                </p>
              </div>
            </div>
            <div className="bg-white rounded-xl px-5 py-4 shadow-sm flex items-start gap-4">
              <div className="w-9 h-9 rounded-lg bg-accent/10 flex items-center justify-center shrink-0">
                <Cloud size={18} className="text-accent" />
              </div>
              <div className="flex-1 min-w-0">
                <h2 className="text-sm font-medium text-ink">Groq (cloud)</h2>
                <p className="text-xs text-muted mt-1">
                  Faster responses via Groq's cloud inference. Uses the server's key if one is configured;
                  otherwise falls back to your own key below.
                </p>

                <label className="block text-xs font-medium text-ink mt-4 mb-1.5">Ta clé API Groq</label>
                <div className="flex items-center gap-2">
                  <div className="relative flex-1">
                    <KeyRound
                      size={13}
                      className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400 pointer-events-none"
                    />
                    <input
                      type={showKey ? "text" : "password"}
                      value={key}
                      onChange={(e) => setKey(e.target.value)}
                      placeholder="gsk_..."
                      autoComplete="off"
                      className="w-full text-sm text-ink border border-gray-200 rounded-lg pl-8 pr-8 py-2
                                 focus:outline-none focus:ring-2 focus:ring-accent/40"
                    />
                    <button
                      type="button"
                      onClick={() => setShowKey((s) => !s)}
                      className="absolute right-2.5 top-1/2 -translate-y-1/2 text-gray-400 hover:text-ink"
                      title={showKey ? "Masquer la clé" : "Afficher la clé"}
                    >
                      {showKey ? <EyeOff size={13} /> : <Eye size={13} />}
                    </button>
                  </div>
                  <button
                    onClick={handleSave}
                    className="shrink-0 px-4 py-2 text-sm rounded-lg bg-accent hover:bg-accent-hover
                               text-white transition-colors flex items-center gap-1.5"
                  >
                    {saved ? <Check size={14} /> : null}
                    {saved ? "Enregistrée" : "Enregistrer"}
                  </button>
                </div>
                <p className="text-[11px] text-muted mt-1.5">Stockée uniquement dans ce navigateur.</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}