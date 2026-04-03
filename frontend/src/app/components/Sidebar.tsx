"use client";

import React, { useState } from 'react';
import { 
  FileUp, 
  Settings, 
  Trash2, 
  Cpu, 
  ChevronRight,
  ShieldAlert,
  Loader2,
  CheckCircle2,
  Info
} from 'lucide-react';
import { Doc, api } from '@/lib/api';

interface SidebarProps {
  docs: Doc[];
  setDocs: React.Dispatch<React.SetStateAction<Doc[]>>;
  provider: string;
  setProvider: (p: string) => void;
}

export default function Sidebar({ 
  docs, setDocs, provider, setProvider 
}: SidebarProps) {
  const [isUploading, setIsUploading] = useState(false);

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files?.length) return;
    
    setIsUploading(true);
    try {
      const result = await api.ingest(Array.from(e.target.files));
      if (result.status === "ok" || result.status === "partial") {
        const { docs: updatedDocs } = await api.getDocsList();
        setDocs(updatedDocs);
      }
      if (result.errors?.length > 0) {
        alert(result.errors[0].error);
      }
    } catch (err) {
      console.error(err);
      alert("In-memory upload failed. Ensure backend is running.");
    } finally {
      setIsUploading(false);
    }
  };

  const handleDelete = async (filename: string) => {
    try {
      await api.deleteDoc(filename);
      setDocs(docs.filter(d => d.filename !== filename));
    } catch (err) {
      console.error(err);
    }
  };

  // Dynamic Nudge Logic
  const getNudge = () => {
    if (provider.startsWith("groq")) return "Configure GROQ_API_KEY in backend secrets.";
    if (provider.startsWith("gemini")) return "Configure GOOGLE_API_KEY in backend secrets.";
    if (provider.startsWith("openai")) return "Configure OPENAI_API_KEY in backend secrets.";
    return "Check backend secrets for API keys.";
  };

  return (
    <div className="w-80 h-full flex flex-col sidebar-gradient text-slate-300 p-6 shadow-2xl border-r border-slate-800">
      <div className="flex items-center gap-3 mb-10">
        <div className="p-2 bg-indigo-500/20 rounded-lg">
          <ShieldAlert className="w-6 h-6 text-indigo-400" />
        </div>
        <h1 className="text-xl font-bold tracking-tight text-white">HR Policy Bot</h1>
      </div>

      {/* 1. KNOWLEDGE INGESTION (Moved to top) */}
      <div className="flex-1 flex flex-col min-h-0">
        <div className="flex items-center justify-between mb-4">
          <label className="text-[10px] uppercase tracking-widest font-bold text-slate-500">Knowledge Base</label>
          <span className="text-[10px] bg-indigo-500/10 text-indigo-400 px-2 py-0.5 rounded-full">{docs.length} Docs</span>
        </div>
        
        <div className="flex-1 overflow-y-auto no-scrollbar space-y-2 mb-6">
          {docs.map((doc) => (
            <div key={doc.filename} className="group flex items-center justify-between p-3 bg-slate-900/30 border border-slate-800/50 rounded-xl hover:bg-slate-800/50 hover:border-slate-700 transition-all">
              <div className="flex items-center gap-3 overflow-hidden">
                <CheckCircle2 className="w-4 h-4 text-emerald-500 flex-shrink-0" />
                <span className="text-sm truncate text-slate-200">{doc.doc_title}</span>
              </div>
              <button 
                onClick={() => handleDelete(doc.filename)}
                className="opacity-0 group-hover:opacity-100 p-1 hover:text-rose-400 transition-all"
              >
                <Trash2 className="w-3.5 h-3.5" />
              </button>
            </div>
          ))}

          <label className={`
            flex flex-col items-center justify-center p-6 border-2 border-dashed border-slate-800 rounded-2xl cursor-pointer hover:border-indigo-500/50 hover:bg-slate-900/50 transition-all
            ${isUploading ? 'opacity-50 pointer-events-none' : ''}
          `}>
            {isUploading ? <Loader2 className="w-6 h-6 animate-spin mb-2" /> : <FileUp className="w-6 h-6 mb-2" />}
            <span className="text-xs font-medium">{isUploading ? 'Ingesting...' : 'Upload Policies'}</span>
            <span className="text-[10px] text-slate-600 mt-1">PDF, DOCX, TXT</span>
            <input type="file" multiple className="hidden" onChange={handleUpload} accept=".pdf,.docx,.txt" />
          </label>
        </div>
      </div>

      {/* 2. LLM SELECTION with Nudges */}
      <div className="pt-6 border-t border-slate-800 space-y-4">
        <div>
          <label className="text-[10px] uppercase tracking-widest font-bold text-slate-500 mb-2 block">LLM Provider</label>
          <div className="relative mb-2">
            <Cpu className="absolute left-3 top-3 w-4 h-4 text-slate-500" />
            <select 
              className="w-full bg-slate-900/50 border border-slate-800 rounded-xl py-2.5 pl-10 pr-4 text-sm focus:ring-2 focus:ring-indigo-500/50 outline-none appearance-none"
              value={provider}
              onChange={(e) => setProvider(e.target.value)}
            >
              <option value="groq_llama_70b">Groq Llama 3.3 70B</option>
              <option value="groq_llama_8b">Groq Llama 3.1 8B</option>
              <option value="gemini_flash">Gemini 2.0 Flash</option>
              <option value="openai_gpt4o">GPT-4o Mini</option>
            </select>
          </div>
          
          {/* Smart Nudge */}
          <div className="flex items-start gap-2 p-2.5 bg-indigo-500/5 border border-indigo-500/10 rounded-lg">
            <Info className="w-3 h-3 text-indigo-400 mt-0.5 flex-shrink-0" />
            <p className="text-[10px] text-slate-400 leading-relaxed italic">
              {getNudge()}
            </p>
          </div>
        </div>

        {/* Configuration Note */}
        <div className="mt-4 p-3 bg-slate-950/50 rounded-xl border border-slate-800/50">
          <p className="text-[10px] text-slate-500 leading-relaxed">
            <span className="font-bold text-slate-400 block mb-1">Admin Setup:</span>
            Add API keys as **Secrets** in your Hugging Face Space settings for production security.
          </p>
        </div>

        <a 
          href={api.getLogsUrl()} 
          target="_blank" 
          className="w-full flex items-center justify-center gap-2 py-3 px-4 rounded-xl bg-slate-900 border border-slate-800 hover:bg-slate-800 border-slate-700 text-xs font-semibold transition-all group"
        >
          Download Audit Log
          <ChevronRight className="w-3 h-3 text-slate-500 group-hover:translate-x-1 transition-transform" />
        </a>
      </div>
    </div>
  );
}
