import { Routes, Route, Navigate } from "react-router-dom";
import { AuthProvider, useAuth } from "./context/AuthContext";
import Login from "./pages/Login";
import ChatAssistant from "./pages/ChatAssistant";
import KnowledgeDocuments from "./pages/KnowledgeDocuments";
import AIFeatures from "./pages/AIFeatures";
import UserProfile from "./pages/UserProfile";

function ProtectedRoute({ children }) {
  const { isAuthenticated, checking } = useAuth();
  if (checking) return <div className="min-h-screen flex items-center justify-center text-muted text-sm">Loading...</div>;
  return isAuthenticated ? children : <Navigate to="/login" replace />;
}

function AppRoutes() {
  return (
    <Routes>
      <Route path="/login" element={<Login />} />
      <Route path="/chat" element={<ProtectedRoute><ChatAssistant /></ProtectedRoute>} />
      <Route path="/documents" element={<ProtectedRoute><KnowledgeDocuments /></ProtectedRoute>} />
      <Route path="/features" element={<ProtectedRoute><AIFeatures /></ProtectedRoute>} />
      <Route path="/profile" element={<ProtectedRoute><UserProfile /></ProtectedRoute>} />
      <Route path="*" element={<Navigate to="/chat" replace />} />
    </Routes>
  );
}

export default function App() {
  return (
    <AuthProvider>
      <AppRoutes />
    </AuthProvider>
  );
}
