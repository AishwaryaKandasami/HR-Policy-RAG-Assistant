/**
 * api.ts — HR Policy Q&A Frontend API Client
 * ==========================================
 * Communicates with the FastAPI backend (Hugging Face Spaces).
 * Handles: Ingestion, Query, and Document Listing.
 */

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

export interface Doc {
  filename: string;
  doc_title: string;
  chunks_added: number;
}

export interface ConversationTurn {
  role: "user" | "assistant";
  content: string;
}

/** Discriminated union for SSE events from /query/stream */
export type StreamEvent =
  | { type: "token"; content: string }
  | {
      type: "meta";
      query_id: string;
      sources: QueryResponse["sources"];
      confidence_label: string;
      confidence_score: number;
      llm_used: string;
      status?: string;
      latency_ms?: number;
    }
  | { type: "error"; content: string };

export interface QueryResponse {
  answer: string;
  sources: Array<{
    doc_title: string;
    section_heading: string;
    page_number: number;
    chunk_text: string;
  }>;
  llm_used: string;
  success: boolean;
  status: "PASS" | "BLOCK" | "ESCALATE";
  confidence_score: number;
  confidence_label: string;
  latency_ms: number;
  query_id: string;
}

export const api = {
  /** Health check */
  getHealth: async () => {
    const res = await fetch(`${BACKEND_URL}/health`);
    return res.json();
  },

  /** Upload documents */
  ingest: async (files: File[]) => {
    const formData = new FormData();
    files.forEach((file) => formData.append("files", file));

    const res = await fetch(`${BACKEND_URL}/ingest`, {
      method: "POST",
      body: formData,
    });
    return res.json();
  },

  /** Query the RAG pipeline */
  query: async (
    text: string,
    provider: string,
    sessionId?: string,
    conversationHistory?: ConversationTurn[],
  ): Promise<QueryResponse> => {
    const res = await fetch(`${BACKEND_URL}/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: text,
        llm_provider: provider,
        session_id: sessionId,
        // Send the last 6 turns maximum to keep the context window bounded
        conversation_history: conversationHistory?.slice(-6) ?? [],
      }),
    });
    return res.json();
  },

  /** Stream an HR query via SSE — yields StreamEvents until [DONE] */
  queryStream: async function* (
    text: string,
    provider: string,
    sessionId?: string,
    conversationHistory?: ConversationTurn[],
  ): AsyncGenerator<StreamEvent> {
    const res = await fetch(`${BACKEND_URL}/query/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: text,
        llm_provider: provider,
        session_id: sessionId,
        conversation_history: conversationHistory?.slice(-6) ?? [],
      }),
    });

    if (!res.ok) throw new Error(`Stream request failed: ${res.status}`);
    if (!res.body) throw new Error("No response body for stream");

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      // SSE events are separated by double newline
      const parts = buffer.split("\n\n");
      buffer = parts.pop() ?? "";   // keep any incomplete trailing chunk

      for (const part of parts) {
        const line = part.trim();
        if (!line.startsWith("data: ")) continue;
        const raw = line.slice(6).trim();
        if (raw === "[DONE]") return;
        try {
          yield JSON.parse(raw) as StreamEvent;
        } catch {
          // malformed event — skip silently
        }
      }
    }
  },

  /** List ingested docs */
  getDocsList: async (): Promise<{ docs: Doc[] }> => {
    const res = await fetch(`${BACKEND_URL}/docs-list`);
    return res.json();
  },

  /** Delete a document */
  deleteDoc: async (filename: string) => {
    const res = await fetch(`${BACKEND_URL}/docs?filename=${encodeURIComponent(filename)}`, {
      method: "DELETE",
    });
    return res.json();
  },

  /** Submit feedback (thumbs up/down) */
  sendFeedback: async (queryId: string, rating: 'up' | 'down', reason?: string) => {
    const res = await fetch(`${BACKEND_URL}/feedback`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query_id: queryId,
        rating: rating,
        reason: reason
      }),
    });
    return res.json();
  },

  /** Fetch onboarding checklist */
  getOnboardingChecklist: async (): Promise<OnboardingTask[]> => {
    const res = await fetch(`${BACKEND_URL}/onboarding-checklist`);
    return res.json();
  },

  /** Logs download helper (returns URL) */
  getLogsUrl: () => `${BACKEND_URL}/logs`,
};

export interface OnboardingTask {
  id: string;
  phase: string;
  title: string;
  description: string;
  related_policy: string;
}
