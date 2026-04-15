"use client";

import React, { useState, useEffect, useRef } from 'react';
import { Send, Sparkles, AlertCircle, ListChecks, MessageSquare } from 'lucide-react';
import Sidebar from './components/Sidebar';
import ChatThread from './components/ChatThread';
import DisclaimerBanner from './components/DisclaimerBanner';
import OnboardingChecklist from './components/OnboardingChecklist';
import { api, Doc, ConversationTurn, StreamEvent } from '@/lib/api';

export default function Home() {
  const [docs, setDocs] = useState<Doc[]>([]);
  const [messages, setMessages] = useState<any[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // View Toggle
  const [viewMode, setViewMode] = useState<"chat" | "onboarding">("chat");

  // Persistence Settings
  const [provider, setProvider] = useState("groq_llama_70b");

  // Session ID: one UUID per browser tab, persisted in sessionStorage
  const [sessionId, setSessionId] = useState<string>("");

  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Initial fetch of docs
    api.getDocsList().then(res => setDocs(res.docs)).catch(console.error);

    // Load provider from localStorage if available
    const savedProvider = localStorage.getItem("hr_provider_choice");
    if (savedProvider) setProvider(savedProvider);

    // Generate or rehydrate session ID for this tab
    const existing = sessionStorage.getItem("hr_session_id");
    if (existing) {
      setSessionId(existing);
    } else {
      const newId = crypto.randomUUID();
      sessionStorage.setItem("hr_session_id", newId);
      setSessionId(newId);
    }
  }, []);

  useEffect(() => {
    localStorage.setItem("hr_provider_choice", provider);
  }, [provider]);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' });
    }
  }, [messages, isLoading]);

  const handleSend = async (e?: React.FormEvent) => {
    e?.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage = input.trim();
    setInput("");
    setError(null);

    // Build history before appending the new user message
    const history: ConversationTurn[] = messages.map((m) => ({
      role: m.role === "bot" ? "assistant" : "user",
      content: m.content,
    }));

    // Add user message + empty streaming bot placeholder in one update
    // so ChatThread never shows the skeleton (streaming cursor appears instead)
    setMessages(prev => [
      ...prev,
      { role: 'user', content: userMessage },
      { role: 'bot', content: '', isStreaming: true },
    ]);
    setIsLoading(true);

    try {
      for await (const event of api.queryStream(userMessage, provider, sessionId, history)) {
        if (event.type === "token") {
          // Append token to the last (streaming) bot message
          setMessages(prev => {
            const updated = [...prev];
            const last = updated[updated.length - 1];
            if (last?.role === 'bot') {
              updated[updated.length - 1] = {
                ...last,
                content: last.content + event.content,
              };
            }
            return updated;
          });
        } else if (event.type === "meta") {
          // Streaming complete — attach metadata and stop cursor
          setMessages(prev => {
            const updated = [...prev];
            const last = updated[updated.length - 1];
            if (last?.role === 'bot') {
              updated[updated.length - 1] = {
                ...last,
                isStreaming: false,
                sources: event.sources,
                confidence_score: event.confidence_score,
                confidence_label: event.confidence_label,
                query_id: event.query_id,
                status: event.status,
              };
            }
            return updated;
          });
        } else if (event.type === "error") {
          setError(event.content);
          // Remove the empty streaming placeholder
          setMessages(prev => prev.slice(0, -1));
        }
      }
    } catch (err) {
      setError("Failed to reach the backend. Ensure the Hugging Face Space is active.");
      // Remove streaming placeholder on network failure
      setMessages(prev => {
        const last = prev[prev.length - 1];
        return last?.role === 'bot' && (last as any).isStreaming ? prev.slice(0, -1) : prev;
      });
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <main className="flex h-screen bg-white font-sans text-slate-800 overflow-hidden">
      {/* 1. Sidebar */}
      <Sidebar
        docs={docs}
        setDocs={setDocs}
        provider={provider}
        setProvider={setProvider}
      />

      {/* 2. Main content */}
      <div className="flex-1 flex flex-col items-stretch relative">
        <DisclaimerBanner />

        {/* View Toggle */}
        <div className="flex justify-center pt-4 pb-2 z-10 bg-white border-b border-slate-100">
          <div className="flex bg-slate-100 p-1 rounded-xl shadow-sm border border-slate-200">
            <button
              onClick={() => setViewMode("chat")}
              className={`flex items-center gap-2 px-6 py-2 rounded-lg text-sm font-semibold transition-all ${viewMode === "chat"
                  ? 'bg-white text-indigo-600 shadow-sm border border-slate-200/50'
                  : 'text-slate-500 hover:text-slate-700 hover:bg-slate-200/50'
                }`}
            >
              <MessageSquare className="w-4 h-4" />
              Ask the Bot
            </button>
            <button
              onClick={() => setViewMode("onboarding")}
              className={`flex items-center gap-2 px-6 py-2 rounded-lg text-sm font-semibold transition-all ${viewMode === "onboarding"
                  ? 'bg-white text-indigo-600 shadow-sm border border-slate-200/50'
                  : 'text-slate-500 hover:text-slate-700 hover:bg-slate-200/50'
                }`}
            >
              <ListChecks className="w-4 h-4" />
              Onboarding Checklist
            </button>
          </div>
        </div>

        {viewMode === "chat" ? (
          <>
            {/* Chat Area */}
            <div ref={scrollRef} className="flex-1 overflow-y-auto">
              <ChatThread messages={messages} isLoading={isLoading} />
            </div>

            {/* Input area */}
            <div className="p-8 pb-10 bg-gradient-to-t from-white via-white to-transparent">
              <form
                onSubmit={handleSend}
                className="max-w-4xl mx-auto relative group"
              >
                {error && (
                  <div className="absolute -top-12 left-0 right-0 p-3 bg-rose-50 border border-rose-100 rounded-xl text-xs text-rose-600 flex items-center gap-2 animate-in slide-in-from-bottom duration-300">
                    <AlertCircle className="w-3.5 h-3.5" />
                    {error}
                  </div>
                )}

                <div className="flex items-center gap-3 bg-white border border-slate-200 rounded-2xl p-2 shadow-xl hover:shadow-2xl hover:border-indigo-200 transition-all focus-within:ring-4 focus-within:ring-indigo-100 focus-within:border-indigo-400">
                  <input
                    type="text"
                    placeholder="Ask an HR policy question..."
                    className="flex-1 px-4 py-3 bg-transparent outline-none text-sm placeholder:text-slate-400"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                  />
                  <button
                    type="submit"
                    disabled={isLoading || !input.trim()}
                    className="p-3.5 rounded-xl bg-indigo-600 text-white hover:bg-indigo-700 disabled:opacity-50 disabled:bg-slate-300 transition-all transform active:scale-95"
                  >
                    {isLoading ? (
                      <Sparkles className="w-5 h-5 animate-pulse" />
                    ) : (
                      <Send className="w-5 h-5" />
                    )}
                  </button>
                </div>

                <div className="mt-4 flex justify-between items-center px-4">
                  <p className="text-[10px] text-slate-400 font-medium tracking-wide">
                    Built with Hybrid Retrieval + Multi-LLM RAG
                  </p>
                  <div className="flex items-center gap-4">
                    <div className="h-1.5 w-1.5 rounded-full bg-emerald-500 animate-pulse"></div>
                    <span className="text-[10px] text-slate-500 font-bold uppercase tracking-widest">System Ready</span>
                  </div>
                </div>
              </form>
            </div>
          </>
        ) : (
          <OnboardingChecklist provider={provider} />
        )}
      </div>
    </main>
  );
}
