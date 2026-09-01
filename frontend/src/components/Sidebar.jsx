import { NavLink, useNavigate } from "react-router-dom";
import { Plus, MessageSquare, FileText, Sparkles, User, LogOut, Trash2 } from "lucide-react";
import GalaxyIcon from "./icons/GalaxyIcon";
import { useAuth } from "../context/AuthContext";

const NAV_ITEMS = [
  { to: "/chat", label: "Chat Assistant", icon: MessageSquare },
  { to: "/documents", label: "Knowledge Documents", icon: FileText },
  { to: "/features", label: "AI Features", icon: Sparkles },
  { to: "/profile", label: "User Profile", icon: User },
];

// Sidebar takes conversations + onNewChat/onSelectChat as props rather than fetching
// them itself, so ChatAssistant (the only page that actually needs live chat state)
// stays the single source of truth -- Sidebar is purely presentational.
export default function Sidebar({ conversations = [], activeConversationId, onNewChat, onSelectChat, onDeleteChat }) {
  const { userId, logout } = useAuth();
  const navigate = useNavigate();

  return (
    <aside className="w-72 shrink-0 bg-sidebar text-white flex flex-col h-screen">
      {/* Header */}
      <div className="flex items-center gap-3 px-5 py-5 border-b border-sidebar-border">
        <div className="w-9 h-9 rounded-lg bg-accent flex items-center justify-center shrink-0">
          <GalaxyIcon size={18} />
        </div>
        <div className="leading-tight">
          <div className="font-semibold text-sm">AI Assistant</div>
        </div>
      </div>

      {/* New Chat */}
      <div className="px-4 pt-4">
        <button
          onClick={onNewChat}
          className="w-full flex items-center justify-center gap-2 bg-accent hover:bg-accent-hover
                     text-white text-sm font-medium py-2.5 rounded-lg transition-colors"
        >
          <Plus size={16} /> New Chat
        </button>
      </div>

      {/* Navigation */}
      <div className="px-4 pt-6">
        <div className="text-[11px] font-semibold text-muted tracking-wider px-2 mb-2">NAVIGATION</div>
        <nav className="flex flex-col gap-1">
          {NAV_ITEMS.map(({ to, label, icon: Icon }) => (
            <NavLink
              key={to}
              to={to}
              className={({ isActive }) =>
                `flex items-center gap-3 px-3 py-2 rounded-lg text-sm transition-colors ${
                  isActive ? "bg-accent text-white" : "text-gray-300 hover:bg-sidebar-hover"
                }`
              }
            >
              <Icon size={16} />
              {label}
            </NavLink>
          ))}
        </nav>
      </div>

      {/* Recent Chats */}
      <div className="px-4 pt-6 flex-1 overflow-y-auto">
        <div className="text-[11px] font-semibold text-muted tracking-wider px-2 mb-2">RECENT CHATS</div>
        <div className="flex flex-col gap-1">
          {conversations.length === 0 && (
            <div className="px-3 py-2 text-xs text-muted">No conversations yet.</div>
          )}
          {conversations.map((c) => (
            <div
              key={c.conversation_id}
              className={`group flex items-center rounded-lg text-sm transition-colors ${
                c.conversation_id === activeConversationId
                  ? "bg-accent text-white"
                  : "text-gray-300 hover:bg-sidebar-hover"
              }`}
            >
              <button
                onClick={() => onSelectChat(c.conversation_id)}
                className="flex-1 min-w-0 flex items-center gap-2 px-3 py-2 text-left"
                title={c.preview}
              >
                <MessageSquare size={14} className="shrink-0 opacity-70" />
                <span className="truncate">{c.preview || "New conversation"}</span>
              </button>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onDeleteChat?.(c.conversation_id);
                }}
                className="shrink-0 mr-2 p-1 rounded opacity-0 group-hover:opacity-100 text-gray-400 hover:text-red-400 hover:bg-black/10 transition-opacity"
                title="Delete conversation"
              >
                <Trash2 size={14} />
              </button>
            </div>
          ))}
        </div>
      </div>

      {/* Footer */}
      <div className="px-4 py-4 border-t border-sidebar-border flex items-center justify-between">
        <div className="flex items-center gap-2 min-w-0">
          <div className="w-7 h-7 rounded-full bg-accent flex items-center justify-center text-xs font-semibold shrink-0">
            {userId ? userId[0].toUpperCase() : "?"}
          </div>
          <span className="text-sm truncate">{userId}</span>
        </div>
        <button
          onClick={() => {
            logout();
            navigate("/login");
          }}
          className="text-gray-400 hover:text-white transition-colors"
          title="Log out"
        >
          <LogOut size={16} />
        </button>
      </div>
    </aside>
  );
}