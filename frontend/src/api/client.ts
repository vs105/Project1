const BASE = import.meta.env.VITE_API_BASE_URL ?? "";

export interface HealthResponse {
  ok: boolean;
  error: string | null;
  brain_dir: string;
  model: string;
  chunks: number;
  prs_chunks: number;
  groq_models: string[];
  has_key: boolean;
  facts_bills: number;
}

export interface Source {
  n: number;
  score: number | null;
  bill_title: string;
  bill_no: string;
  year: string;
  ministry: string;
  status: string;
  slot: string;
  clause: string;
  part: string;
  clause_no: string;
  source_file: string;
  language: string;
  tag: string;
  text: string;
}

export interface AskResponse {
  reply: string;
  sources: Source[];
  used: string;
  refusal: string;
  filters: { year: string | null; bill_no: string | null };
  sources_text: string;
  error?: string;
}

export interface ExamplesResponse {
  examples: string[];
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, init);
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    throw new Error(body?.error || `Request failed (${res.status})`);
  }
  return res.json();
}

export function fetchHealth(): Promise<HealthResponse> {
  return request("/api/health");
}

export function fetchExamples(): Promise<ExamplesResponse> {
  return request("/api/examples");
}

export function askQuestion(
  question: string,
  history?: [string, string][]
): Promise<AskResponse> {
  return request("/api/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, history }),
  });
}
