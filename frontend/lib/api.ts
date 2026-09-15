/**
 * Talking to the FinX API.
 *
 * The chat endpoint streams Server-Sent Events rather than returning a single
 * response, so this parses the stream by hand. `EventSource` cannot be used
 * because the request is a POST carrying the question and the document it is
 * about.
 */

export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

export type Citation = {
  n: number;
  chunk_id: string;
  source: "user_document" | "rbi_corpus";
  document_id: string;
  version: number;
  clause_number: string | null;
  page: number | null;
  title: string | null;
  circular_id: string | null;
  source_url: string | null;
  text: string;
  score: number;
  confidence: number;
};

export type ChatMeta = {
  document_id: string;
  version: number;
  title: string;
  confidence: number;
  considered: number;
  provider: string;
  model: string;
};

export type UploadResult = {
  document_id: string;
  version: number;
  title: string;
  kind: string;
  clause_count: number;
  chunks_written: number;
  embeddings_computed: number;
  embeddings_reused: number;
  reuse_percent: number;
  warnings: string[];
  summary: string;
};

export async function uploadDocument(file: File): Promise<UploadResult> {
  const body = new FormData();
  body.append("file", file);

  const response = await fetch(`${API_BASE}/documents`, {
    method: "POST",
    body,
  });

  if (!response.ok) {
    const detail = await response
      .json()
      .then((d) => d.detail as string)
      .catch(() => response.statusText);
    throw new Error(detail || "Upload failed");
  }
  return response.json();
}

export type ChatHandlers = {
  onMeta?: (meta: ChatMeta) => void;
  onCitations?: (citations: Citation[]) => void;
  onDelta?: (text: string) => void;
  onVerified?: (payload: {
    text: string;
    stripped: string[];
    fully_grounded?: boolean;
  }) => void;
  onError?: (message: string) => void;
  onDone?: () => void;
};

/**
 * POST a question and dispatch each SSE event as it arrives.
 *
 * Events are separated by a blank line, so the buffer is split on "\n\n" and
 * any trailing partial event is kept for the next chunk.
 */
export async function askQuestion(
  question: string,
  documentId: string,
  version: number | null,
  handlers: ChatHandlers,
  signal?: AbortSignal,
): Promise<void> {
  const response = await fetch(`${API_BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question,
      document_id: documentId,
      version: version ?? undefined,
    }),
    signal,
  });

  if (!response.ok || !response.body) {
    const detail = await response
      .json()
      .then((d) => d.detail as string)
      .catch(() => response.statusText);
    handlers.onError?.(detail || "The request failed");
    return;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const blocks = buffer.split("\n\n");
    buffer = blocks.pop() ?? "";

    for (const block of blocks) {
      if (!block.trim()) continue;

      let event = "message";
      const data: string[] = [];
      for (const line of block.split("\n")) {
        if (line.startsWith("event: ")) event = line.slice(7).trim();
        else if (line.startsWith("data: ")) data.push(line.slice(6));
      }
      if (!data.length) continue;

      let payload: unknown;
      try {
        payload = JSON.parse(data.join("\n"));
      } catch {
        continue;
      }

      switch (event) {
        case "meta":
          handlers.onMeta?.(payload as ChatMeta);
          break;
        case "citations":
          handlers.onCitations?.(payload as Citation[]);
          break;
        case "delta":
          handlers.onDelta?.((payload as { text: string }).text);
          break;
        case "verified":
          handlers.onVerified?.(
            payload as { text: string; stripped: string[]; fully_grounded?: boolean },
          );
          break;
        case "error":
          handlers.onError?.((payload as { message: string }).message);
          break;
        case "done":
          handlers.onDone?.();
          break;
      }
    }
  }
}
