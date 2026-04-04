"use client";

import React, { useState } from 'react';
import { 
  Bot, 
  User, 
  ChevronDown, 
  ChevronUp, 
  ExternalLink,
  MessageSquare,
  CheckCircle2,
  AlertTriangle,
  AlertCircle,
  ThumbsUp,
  ThumbsDown,
  Check
} from 'lucide-react';
import { QueryResponse, api } from '@/lib/api';

interface Message {
  role: 'user' | 'bot';
  content: string;
  sources?: QueryResponse['sources'];
  status?: string;
  confidence_score?: number;
  confidence_label?: string;
  query_id?: string;
}

interface ChatThreadProps {
  messages: Message[];
  isLoading: boolean;
}

export default function ChatThread({ messages, isLoading }: ChatThreadProps) {
  if (messages.length === 0) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center text-slate-400 p-10 animate-in fade-in duration-700">
        <div className="p-5 bg-slate-50 rounded-full mb-6">
          <MessageSquare className="w-12 h-12 text-slate-300" />
        </div>
        <h2 className="text-xl font-semibold text-slate-800 mb-2">How can I help you today?</h2>
        <p className="text-sm max-w-xs text-center leading-relaxed">
          Ask any question about company policies, holidays, or UK employment standards.
        </p>
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-y-auto px-6 py-8 space-y-8 no-scrollbar">
      {messages.map((msg, idx) => (
        <MessageBubble key={idx} message={msg} />
      ))}
      {isLoading && (
        <div className="flex gap-4 bot-bubble chat-bubble animate-pulse bg-slate-50/50">
          <div className="w-8 h-8 rounded-lg bg-slate-200 flex items-center justify-center flex-shrink-0">
            <Bot className="w-5 h-5 text-slate-400" />
          </div>
          <div className="flex-1 space-y-2">
            <div className="h-3 w-1/4 bg-slate-200 rounded"></div>
            <div className="h-2 w-3/4 bg-slate-200 rounded"></div>
          </div>
        </div>
      )}
    </div>
  );
}

function MessageBubble({ message }: { message: Message }) {
  const isBot = message.role === 'bot';
  const [showSources, setShowSources] = useState(false);
  const [feedback, setFeedback] = useState<'up' | 'down' | null>(null);
  const [showReasons, setShowReasons] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleFeedback = async (rating: 'up' | 'down', reason?: string) => {
    if (!message.query_id || feedback) return;
    
    setIsSubmitting(true);
    try {
      await api.sendFeedback(message.query_id, rating, reason);
      setFeedback(rating);
      setShowReasons(false);
    } catch (err) {
      console.error("Feedback failed:", err);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className={`flex gap-4 ${isBot ? 'bot-bubble pr-10' : 'user-bubble pl-10'} chat-bubble w-fit max-w-[85%]`}>
      <div className={`w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0 ${
        isBot ? 'bg-indigo-50 text-indigo-500' : 'bg-white/10 text-white'
      }`}>
        {isBot ? <Bot className="w-5 h-5" /> : <User className="w-5 h-5" />}
      </div>
      
      <div className="flex-1 min-w-0">
        <div className={`whitespace-pre-wrap leading-relaxed ${isBot ? 'text-slate-700' : 'text-white'}`}>
          {message.content}
        </div>
        
        {isBot && message.confidence_label && (
          <div className="flex flex-wrap items-center gap-3">
            <ConfidencePill 
              label={message.confidence_label} 
              score={message.confidence_score || 0} 
            />
            
            {message.query_id && !message.status && (
              <div className="mt-3 flex items-center gap-1 border-l border-slate-200 pl-3">
                <button 
                  onClick={() => handleFeedback('up')}
                  disabled={!!feedback || isSubmitting}
                  className={`p-1.5 rounded-md transition-all ${
                    feedback === 'up' 
                      ? 'bg-emerald-50 text-emerald-600' 
                      : 'text-slate-400 hover:text-emerald-500 hover:bg-emerald-50'
                  } disabled:cursor-default`}
                  title="Helpful"
                >
                  <ThumbsUp className={`w-3.5 h-3.5 ${feedback === 'up' ? 'fill-emerald-600' : ''}`} />
                </button>
                
                <div className="relative">
                  <button 
                    onClick={() => {
                      if (feedback) return;
                      setShowReasons(!showReasons);
                    }}
                    disabled={!!feedback || isSubmitting}
                    className={`p-1.5 rounded-md transition-all ${
                      feedback === 'down' 
                        ? 'bg-rose-50 text-rose-600' 
                        : 'text-slate-400 hover:text-rose-500 hover:bg-rose-50'
                    } disabled:cursor-default`}
                    title="Not helpful"
                  >
                    <ThumbsDown className={`w-3.5 h-3.5 ${feedback === 'down' ? 'fill-rose-600' : ''}`} />
                  </button>

                  {showReasons && (
                    <div className="absolute left-0 top-full mt-2 z-10 w-40 bg-white rounded-xl shadow-2xl border border-slate-100 p-1 animate-in zoom-in-95 duration-200">
                      {["Wrong answer", "Incomplete", "Not what I meant"].map((reason) => (
                        <button 
                          key={reason}
                          onClick={() => handleFeedback('down', reason)}
                          className="w-full text-left px-3 py-2 text-[11px] font-medium text-slate-600 hover:bg-slate-50 hover:text-indigo-600 rounded-lg transition-colors flex items-center justify-between group"
                        >
                          {reason}
                          <Check className="w-3 h-3 opacity-0 group-hover:opacity-100 transition-opacity" />
                        </button>
                      ))}
                    </div>
                  )}
                </div>

                {feedback && (
                  <span className="text-[10px] font-bold text-slate-400 ml-1 animate-in fade-in slide-in-from-left-1">
                    Thanks!
                  </span>
                )}
              </div>
            )}
          </div>
        )}

        {isBot && message.sources && message.sources.length > 0 && (
          <div className="mt-4 pt-4 border-t border-slate-100">
            <button 
              onClick={() => setShowSources(!showSources)}
              className="flex items-center gap-2 text-[11px] font-bold uppercase tracking-wider text-indigo-500 hover:text-indigo-600 transition-colors"
            >
              Citations ({message.sources.length})
              {showSources ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
            </button>

            {showSources && (
              <div className="mt-3 grid gap-2">
                {message.sources.map((src, sIdx) => (
                  <div key={sIdx} className="source-card">
                    <div className="flex justify-between items-start mb-1">
                      <span className="font-bold text-slate-700 truncate max-w-[70%]">{src.doc_title}</span>
                      <span className="text-[10px] bg-white px-1.5 py-0.5 rounded border border-slate-200 uppercase tabular-nums">Page {src.page_number}</span>
                    </div>
                    <p className="line-clamp-2 italic text-slate-500 mb-1">{src.section_heading}</p>
                    <div className="flex items-center gap-1 text-[10px] text-indigo-500 font-semibold group-hover:underline">
                      <ExternalLink className="w-2.5 h-2.5" />
                      View Context
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

function ConfidencePill({ label, score }: { label: string, score: number }) {
  const isHigh = label.toLowerCase() === 'high';
  const isMedium = label.toLowerCase() === 'medium';
  const isLow = label.toLowerCase() === 'low';

  const styles = {
    high: "bg-emerald-50 text-emerald-700 border-emerald-200",
    medium: "bg-amber-50 text-amber-700 border-amber-200",
    low: "bg-rose-50 text-rose-700 border-rose-200",
  };

  const currentStyle = isHigh ? styles.high : isMedium ? styles.medium : styles.low;
  
  return (
    <div className={`mt-3 inline-flex items-center gap-1.5 px-2 py-1 rounded-md border text-[10px] font-bold uppercase tracking-tight ${currentStyle} animate-in fade-in slide-in-from-left-2 duration-500`}>
      {isHigh && <CheckCircle2 className="w-3 h-3" />}
      {isMedium && <AlertTriangle className="w-3 h-3" />}
      {isLow && <AlertCircle className="w-3 h-3" />}
      
      <span>Confidence: {label} ({Math.round(score * 100)}%)</span>
      
      {isLow && (
        <span className="ml-1 pl-1.5 border-l border-rose-200 text-rose-800">
          Verify with HR.
        </span>
      )}
    </div>
  );
}
