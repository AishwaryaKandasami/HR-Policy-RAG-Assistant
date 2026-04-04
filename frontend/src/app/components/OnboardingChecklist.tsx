"use client";

import React, { useState, useEffect } from 'react';
import { CheckCircle2, Circle, ChevronDown, ChevronUp, ExternalLink, Sparkles, AlertCircle } from 'lucide-react';
import { api, OnboardingTask, QueryResponse } from '@/lib/api';

interface OnboardingChecklistProps {
  provider: string;
}

export default function OnboardingChecklist({ provider }: OnboardingChecklistProps) {
  const [tasks, setTasks] = useState<OnboardingTask[]>([]);
  const [activeTab, setActiveTab] = useState<string>("Before Day 1");
  const [completedTasks, setCompletedTasks] = useState<Set<string>>(new Set());
  
  // Track RAG answers: source data and loading states for each task
  const [answers, setAnswers] = useState<Record<string, QueryResponse>>({});
  const [loadingTasks, setLoadingTasks] = useState<Set<string>>(new Set());
  const [expandedTask, setExpandedTask] = useState<string | null>(null);

  useEffect(() => {
    api.getOnboardingChecklist()
      .then(setTasks)
      .catch(console.error);
  }, []);

  const phases = ["Before Day 1", "Day 1", "Week 1", "First Month"];
  const currentTasks = tasks.filter(t => t.phase === activeTab);

  const toggleTask = (taskId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    const newCompleted = new Set(completedTasks);
    if (newCompleted.has(taskId)) {
      newCompleted.delete(taskId);
    } else {
      newCompleted.add(taskId);
    }
    setCompletedTasks(newCompleted);
  };

  const handleExpand = async (task: OnboardingTask) => {
    if (expandedTask === task.id) {
      setExpandedTask(null);
      return;
    }
    setExpandedTask(task.id);

    // If we've already answered this, don't query again unless forced
    if (answers[task.id]) return;

    setLoadingTasks(prev => new Set(prev).add(task.id));
    try {
      const res = await api.query(task.related_policy, provider);
      setAnswers(prev => ({ ...prev, [task.id]: res }));
    } catch (err) {
      console.error(err);
    } finally {
      const newLoading = new Set(loadingTasks);
      newLoading.delete(task.id);
      setLoadingTasks(newLoading);
    }
  };

  return (
    <div className="flex-1 overflow-y-auto px-6 py-8 space-y-6 flex flex-col h-full bg-slate-50/50">
      
      <div className="max-w-4xl w-full mx-auto space-y-8">
        
        <div className="text-center space-y-2 mb-8">
          <h2 className="text-2xl font-bold text-slate-800">Your Onboarding Journey</h2>
          <p className="text-sm text-slate-500 max-w-xl mx-auto">
            Complete your onboarding tasks below. Click on any task to instantly ask the HR assistant 
            about the policies related to that specific step.
          </p>
        </div>

        {/* Informational Banner */}
        <div className="bg-amber-50 border border-amber-200 text-amber-800 px-4 py-3 rounded-xl flex items-start gap-3 text-sm">
          <AlertCircle className="w-5 h-5 text-amber-600 mt-0.5 flex-shrink-0" />
          <p>
            <strong>Demo environment:</strong> Checklist progress is ephemeral and resets on refresh. 
            In production, this state persists per employee.
          </p>
        </div>

        {/* Tabs */}
        <div className="flex space-x-1 bg-slate-100/50 p-1 rounded-xl shadow-inner border border-slate-200">
          {phases.map(phase => (
            <button
              key={phase}
              onClick={() => setActiveTab(phase)}
              className={`flex-1 py-2.5 text-sm font-semibold rounded-lg transition-all ${
                activeTab === phase 
                  ? 'bg-white text-indigo-600 shadow-sm border border-slate-200/50' 
                  : 'text-slate-500 hover:text-slate-700 hover:bg-slate-200/50'
              }`}
            >
              {phase}
            </button>
          ))}
        </div>

        {/* Task List */}
        <div className="space-y-4">
          {currentTasks.map(task => {
            const isCompleted = completedTasks.has(task.id);
            const isExpanded = expandedTask === task.id;
            const isLoading = loadingTasks.has(task.id);
            const answerData = answers[task.id];

            return (
              <div 
                key={task.id} 
                className={`bg-white rounded-2xl border transition-all duration-300 overflow-hidden ${
                  isExpanded ? 'border-indigo-300 shadow-md' : 'border-slate-200 hover:border-slate-300 shadow-sm'
                }`}
              >
                {/* Task Header Row */}
                <div 
                  className="px-5 py-4 flex items-center justify-between cursor-pointer group"
                  onClick={() => handleExpand(task)}
                >
                  <div className="flex items-center gap-4 flex-1">
                    <button 
                      onClick={(e) => toggleTask(task.id, e)}
                      className={`transition-colors focus:outline-none ${isCompleted ? 'text-emerald-500' : 'text-slate-300 group-hover:text-slate-400'}`}
                    >
                      {isCompleted ? <CheckCircle2 className="w-6 h-6" /> : <Circle className="w-6 h-6" />}
                    </button>
                    <div>
                      <h3 className={`font-semibold transition-colors ${isCompleted ? 'text-slate-500 line-through' : 'text-slate-800'}`}>
                        {task.title}
                      </h3>
                      <p className="text-sm text-slate-500 mt-0.5">{task.description}</p>
                    </div>
                  </div>
                  <div className="text-slate-400 flex items-center gap-2">
                    <span className="text-[10px] font-bold uppercase tracking-wider text-indigo-400 bg-indigo-50 px-2 py-1 rounded">Ask Policy</span>
                    {isExpanded ? <ChevronUp className="w-5 h-5" /> : <ChevronDown className="w-5 h-5" />}
                  </div>
                </div>

                {/* Expanded Inline RAG Panel */}
                {isExpanded && (
                  <div className="px-5 pb-5 pt-2 border-t border-slate-100 bg-indigo-50/30 animate-in slide-in-from-top-2 duration-200">
                    <div className="flex items-center gap-2 mb-3">
                      <Sparkles className="w-4 h-4 text-indigo-600" />
                      <span className="text-xs font-semibold text-indigo-900 bg-indigo-100/50 px-2 py-0.5 rounded-full border border-indigo-200">
                        Bot Query: "{task.related_policy}"
                      </span>
                    </div>

                    {isLoading ? (
                      <div className="space-y-2 animate-pulse pl-6">
                        <div className="h-3 bg-slate-200 rounded w-3/4"></div>
                        <div className="h-3 bg-slate-200 rounded w-5/6"></div>
                        <div className="h-3 bg-slate-200 rounded w-1/2"></div>
                      </div>
                    ) : answerData ? (
                      <div className="pl-6 space-y-4">
                        <p className="text-sm text-slate-700 leading-relaxed whitespace-pre-wrap">
                          {answerData.answer}
                        </p>
                        
                        {answerData.sources && answerData.sources.length > 0 && (
                          <div className="grid gap-2 mt-4">
                            {answerData.sources.map((src, idx) => (
                              <div key={idx} className="bg-white border border-slate-200 p-3 rounded-xl flex justify-between items-start">
                                <div>
                                  <span className="font-bold text-xs text-slate-700">{src.doc_title}</span>
                                  <p className="text-[10px] text-slate-500 mt-1 italic">{src.section_heading}</p>
                                </div>
                                <span className="text-[10px] bg-slate-100 px-1.5 py-0.5 rounded font-medium border border-slate-200 whitespace-nowrap">
                                  Page {src.page_number}
                                </span>
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    ) : (
                      <p className="text-sm text-rose-500 pl-6">Failed to retrieve an answer.</p>
                    )}
                  </div>
                )}
              </div>
            );
          })}
        </div>

      </div>
    </div>
  );
}
