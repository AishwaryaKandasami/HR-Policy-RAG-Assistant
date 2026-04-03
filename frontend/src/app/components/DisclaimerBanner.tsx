"use client";

import React, { useState, useEffect } from 'react';
import { AlertTriangle, X } from 'lucide-react';

export default function DisclaimerBanner() {
  const [isVisible, setIsVisible] = useState(true);

  if (!isVisible) return null;

  return (
    <div className="bg-amber-50 border-b border-amber-100 px-6 py-2.5 flex items-center justify-between sticky top-0 z-50 animate-in slide-in-from-top duration-500">
      <div className="flex items-center gap-3">
        <AlertTriangle className="w-4 h-4 text-amber-600 flex-shrink-0" />
        <p className="text-[12px] font-medium text-amber-800 leading-tight">
          <span className="font-bold">Privacy Notice:</span> This is a demonstration RAG system. Do not upload actual sensitive PII or confidential company data.
        </p>
      </div>
      <button 
        onClick={() => setIsVisible(false)}
        className="p-1 hover:bg-amber-100 rounded-lg transition-colors text-amber-500"
      >
        <X className="w-4 h-4" />
      </button>
    </div>
  );
}
