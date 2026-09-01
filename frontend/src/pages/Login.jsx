import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Layers } from "lucide-react";
import { useAuth } from "../context/AuthContext";
import * as api from "../api/client";

export default function Login() {
  const [mode, setMode] = useState("login"); // "login" | "register"
  const [userId, setUserId] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const { login } = useAuth();
  const navigate = useNavigate();

  async function handleSubmit(e) {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      if (mode === "register") {
        await api.register(userId, password);
      }
      await login(userId, password);
      navigate("/chat");
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen bg-canvas flex items-center justify-center px-4">
      <div className="w-full max-w-sm bg-white rounded-2xl shadow-sm p-8">
        <div className="flex flex-col items-center mb-6">
          <div className="w-11 h-11 rounded-xl bg-accent flex items-center justify-center mb-3">
            <Layers size={20} className="text-white" />
          </div>
          <h1 className="text-lg font-semibold text-ink"> AI Assistant</h1>
        </div>

        <form onSubmit={handleSubmit} className="flex flex-col gap-3">
          <input
            value={userId}
            onChange={(e) => setUserId(e.target.value)}
            placeholder="Username"
            className="border border-gray-200 rounded-lg px-3 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-accent/40"
            required
          />
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder="Password"
            className="border border-gray-200 rounded-lg px-3 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-accent/40"
            required
          />
          {error && <p className="text-xs text-red-500">{error}</p>}
          <button
            type="submit"
            disabled={loading}
            className="mt-2 bg-accent hover:bg-accent-hover disabled:bg-gray-300 text-white text-sm font-medium py-2.5 rounded-lg transition-colors"
          >
            {loading ? "..." : mode === "login" ? "Sign in" : "Create account"}
          </button>
        </form>

        <button
          onClick={() => { setMode(mode === "login" ? "register" : "login"); setError(""); }}
          className="w-full text-center text-xs text-muted hover:text-accent mt-4 transition-colors"
        >
          {mode === "login" ? "No account yet? Create one" : "Already have an account? Sign in"}
        </button>
      </div>
    </div>
  );
}
