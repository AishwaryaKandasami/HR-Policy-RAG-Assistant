"use client";

import React, { useState, useEffect, useRef } from 'react';
import { Send, Sparkles, AlertCircle, ListChecks, MessageSquare } from 'lucide-react';
import Sidebar from './components/Sidebar';
import ChatThread from './components/ChatThread';
import DisclaimerBanner from './components/DisclaimerBanner';
import OnboardingChecklist from './components/OnboardingChecklist';
import { api, Doc } from '@/lib/api';

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

  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Initial fetch of docs
    api.getDocsList().then(res => setDocs(res.docs)).catch(console.error);
    
    // Load provider from localStorage if available
    const savedProvider = localStorage.getItem("hr_provider_choice");
    if (savedProvider) setProvider(savedProvider);
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
    setMessages(prev => [...prev, { role: 'user', content: userMessage }]);
    setIsLoading(true);

    try {
      const res = await api.query(userMessage, provider);
      
      if (res.status === "BLOCK" || res.status === "ESCALATE") {
        setMessages(prev => [...prev, { 
          role: 'bot', 
          content: res.answer, 
          status: res.status,
          confidence_score: res.confidence_score,
          confidence_label: res.confidence_label,
          query_id: res.query_id
        }]);
      } else if (res.success) {
        setMessages(prev => [...prev, { 
          role: 'bot', 
          content: res.answer, 
          sources: res.sources,
          confidence_score: res.confidence_score,
          confidence_label: res.confidence_label,
          query_id: res.query_id
        }]);
      } else {
        setError(res.answer || "Check backend connection and API keys.");
      }
    } catch (err) {
      setError("Failed to reach the backend. Ensure the Hugging Face Space is active.");
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
              className={`flex items-center gap-2 px-6 py-2 rounded-lg text-sm font-semibold transition-all ${
                viewMode === "chat" 
                  ? 'bg-white text-indigo-600 shadow-sm border border-slate-200/50' 
                  : 'text-slate-500 hover:text-slate-700 hover:bg-slate-200/50'
              }`}
            >
              <MessageSquare className="w-4 h-4" />
              Ask the Bot
            </button>
            <button
              onClick={() => setViewMode("onboarding")}
              className={`flex items-center gap-2 px-6 py-2 rounded-lg text-sm font-semibold transition-all ${
                viewMode === "onboarding" 
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
