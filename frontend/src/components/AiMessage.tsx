import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { RotateCcw } from "lucide-react";
import { Source } from "../api/client";
import SourcePills from "./SourcePills";

interface AiMessageProps {
  content: string;
  sources?: Source[];
  error?: boolean;
  onRetry?: () => void;
}

export default function AiMessage({ content, sources, error, onRetry }: AiMessageProps) {
  return (
    <div className="flex justify-start gap-3 msg-enter">
      {/* AI avatar */}
      <div
        className="w-9 h-9 rounded-full bg-ink-dark flex items-center justify-center flex-shrink-0"
        aria-hidden="true"
      >
        <svg width="14" height="14" viewBox="0 0 16 16" fill="none">
          <path d="M3 3L13 13M13 3L3 13" stroke="#fff" strokeWidth="2" strokeLinecap="round" />
        </svg>
      </div>
      <div className="max-w-[75%] flex flex-col gap-0">
        <div
          className={`rounded-2xl px-4 py-3 text-sm leading-relaxed ${
            error
              ? "bg-red-900/80 text-red-100 border border-red-700/50"
              : "ink-bubble"
          }`}
        >
          {error ? (
            <div className="flex items-center gap-3">
              <span>{content}</span>
              {onRetry && (
                <button
                  onClick={onRetry}
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-white/10 hover:bg-white/20 text-[12px] text-white font-medium transition-colors duration-150 flex-shrink-0 focus-ring"
                  aria-label="Retry request"
                >
                  <RotateCcw size={12} />
                  Retry
                </button>
              )}
            </div>
          ) : (
            <div className="ai-prose">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>
            </div>
          )}
        </div>
        {sources && sources.length > 0 && <SourcePills sources={sources} />}
      </div>
    </div>
  );
}
