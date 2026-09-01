import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The React app calls relative paths like "/ask" or "/auth/login".
// Vite's dev server intercepts those and forwards them server-side to FastAPI --
// the browser only ever talks to localhost:5173 (this dev server), so there's no
// cross-origin request at all from the browser's point of view. This means we
// never have to add CORSMiddleware to main.py.
//
// Change target below if your `uvicorn main:app` runs on a different port.
const BACKEND_URL = "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/auth": BACKEND_URL,
      "/ask": BACKEND_URL,
      "/ask_stream": BACKEND_URL,
      "/ingest": BACKEND_URL,
      "/documents": BACKEND_URL,
      "/conversations": BACKEND_URL,
      "/new_conversation": BACKEND_URL,
      "/clear_memory": BACKEND_URL,
    },
  },
});
