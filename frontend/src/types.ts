import { Source } from "../api/client";

/* ── A single conversation in memory ──────────────────────────── */
export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  error?: boolean;
}

export interface Conversation {
  id: string;
  title: string;
  messages: Message[];
  createdAt: number; // epoch ms
}

/* ── Helper to generate an id ─────────────────────────────────── */
export function uid(): string {
  return Date.now().toString(36) + Math.random().toString(36).slice(2, 8);
}

/* ── Derive a short title from the first user message ─────────── */
export function deriveTitle(text: string): string {
  const cleaned = text.replace(/\s+/g, " ").trim();
  if (cleaned.length <= 48) return cleaned;
  return cleaned.slice(0, 45).trimEnd() + "...";
}

/* ── Group conversations by recency ───────────────────────────── */
export type DateGroup = "Today" | "Previous 7 days" | "Older";

export function groupByDate(
  convos: Conversation[]
): { label: DateGroup; items: Conversation[] }[] {
  const now = Date.now();
  const dayMs = 86_400_000;
  const todayStart = now - (now % dayMs);
  const weekAgo = todayStart - 7 * dayMs;

  const today: Conversation[] = [];
  const week: Conversation[] = [];
  const older: Conversation[] = [];

  for (const c of convos) {
    if (c.createdAt >= todayStart) today.push(c);
    else if (c.createdAt >= weekAgo) week.push(c);
    else older.push(c);
  }

  const groups: { label: DateGroup; items: Conversation[] }[] = [];
  if (today.length) groups.push({ label: "Today", items: today });
  if (week.length) groups.push({ label: "Previous 7 days", items: week });
  if (older.length) groups.push({ label: "Older", items: older });
  return groups;
}
