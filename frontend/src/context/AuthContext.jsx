import { createContext, useContext, useState, useCallback } from "react";
import * as api from "../api/client";

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [userId, setUserId] = useState(null);
  const [checking, setChecking] = useState(true);

  // On first load, if a token is already in localStorage, verify it's still valid
  // by calling /auth/me rather than trusting it blindly.
  useState(() => {
    (async () => {
      if (api.getToken()) {
        try {
          const info = await api.me();
          setUserId(info.user_id);
        } catch {
          api.clearToken();
        }
      }
      setChecking(false);
    })();
  });

  const login = useCallback(async (id, password) => {
    await api.login(id, password);
    setUserId(id);
  }, []);

  const logout = useCallback(() => {
    api.clearToken();
    setUserId(null);
  }, []);

  return (
    <AuthContext.Provider value={{ userId, isAuthenticated: !!userId, checking, login, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within AuthProvider");
  return ctx;
}
