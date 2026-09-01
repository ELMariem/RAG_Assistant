import { useEffect, useRef, useState, useCallback } from "react";
import Sidebar from "../components/Sidebar";
import MessageBubble from "../components/MessageBubble";
import MessageInput from "../components/MessageInput";
import * as api from "../api/client";

export default function ChatAssistant() {
  const [conversations, setConversations] = useState([]);
  const [conversationId, setConversationId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [backend, setBackend] = useState("ollama");
  const [isGenerating, setIsGenerating] = useState(false);
  const scrollRef = useRef(null);

  const loadConversations = useCallback(async () => {
    const list = await api.listConversations();
    setConversations(list);
    return list;
  }, []);

  // On mount: load the conversation list, and resume the most recent one --
  // mirrors app.py's own "Welcome back, resuming your most recent conversation" behavior.
  useEffect(() => {
    (async () => {
      const list = await loadConversations();
      if (list.length > 0) {
        handleSelectChat(list[0].conversation_id);
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages]);

  async function handleSelectChat(id) {
    setConversationId(id);
    const history = await api.getConversationMessages(id);
    setMessages(history);
  }

  async function handleNewChat() {
    const { conversation_id } = await api.newConversation();
    setConversationId(conversation_id);
    setMessages([]);
    loadConversations();
  }

  async function handleDeleteChat(id) {
    if (!window.confirm("Supprimer cette conversation ? Cette action est irréversible.")) return;
    await api.clearConversation(id);
    const list = await loadConversations();
    if (id === conversationId) {
      // The active conversation was just deleted -- fall back to the next most
      // recent one, or a blank slate if that was the last conversation left.
      if (list.length > 0) {
        handleSelectChat(list[0].conversation_id);
      } else {
        setConversationId(null);
        setMessages([]);
      }
    }
  }

  async function handleSend(question) {
    setMessages((prev) => [...prev, { role: "user", content: question }, { role: "assistant", content: "" }]);
    setIsGenerating(true);
    try {
      const meta = await api.askStream(question, { backend, conversationId }, (token) => {
        setMessages((prev) => {
          const next = [...prev];
          next[next.length - 1] = { role: "assistant", content: next[next.length - 1].content + token };
          return next;
        });
      });
      if (meta?.conversation_id && meta.conversation_id !== conversationId) {
        setConversationId(meta.conversation_id);
      }
      // Sources only arrive once generation is done (in the SSE "done" event),
      // so attach them to the just-finished assistant message here.
      setMessages((prev) => {
        const next = [...prev];
        next[next.length - 1] = { ...next[next.length - 1], sources: meta?.sources || [] };
        return next;
      });
      loadConversations(); // refresh sidebar preview/title
    } catch (err) {
      setMessages((prev) => {
        const next = [...prev];
        next[next.length - 1] = { role: "assistant", content: `Error: ${err.message}` };
        return next;
      });
    } finally {
      setIsGenerating(false);
    }
  }

  const currentTitle = conversations.find((c) => c.conversation_id === conversationId)?.preview || "New chat";

  return (
    <div className="flex h-screen bg-canvas">
      <Sidebar
        conversations={conversations}
        activeConversationId={conversationId}
        onNewChat={handleNewChat}
        onSelectChat={handleSelectChat}
        onDeleteChat={handleDeleteChat}
      />

      <div className="flex-1 flex flex-col min-w-0">
        {/* Top bar */}
        <div className="flex items-center justify-between px-6 py-4 bg-white border-b border-gray-200">
          <h1 className="text-sm font-medium text-ink truncate">{currentTitle}</h1>
          <button
            onClick={handleNewChat}
            className="text-xs font-medium text-accent border border-accent/30 hover:bg-accent/5 rounded-lg px-3 py-1.5 transition-colors"
          >
            + New Chat
          </button>
        </div>

        {/* Messages */}
        <div ref={scrollRef} className="flex-1 overflow-y-auto px-6 py-6">
          <div className="max-w-3xl mx-auto flex flex-col gap-4">
            {messages.length === 0 && (
              <div className="text-center text-muted text-sm mt-20">
                Ask anything about your ingested documents to get started.
              </div>
            )}
            {messages.map((m, i) => (
              <MessageBubble
                key={i}
                role={m.role}
                content={m.content}
                sources={m.sources}
                isStreaming={isGenerating && i === messages.length - 1 && m.role === "assistant"}
              />
            ))}
          </div>
        </div>

        <MessageInput onSend={handleSend} disabled={isGenerating} backend={backend} onBackendChange={setBackend} />
      </div>
    </div>
  );
}