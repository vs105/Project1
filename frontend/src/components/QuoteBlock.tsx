import { useState } from "react";
import { Copy, Check } from "lucide-react";

interface QuoteBlockProps {
  text: string;
  label?: string;
}

export default function QuoteBlock({ text, label }: QuoteBlockProps) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* clipboard not available */
    }
  };

  return (
    <div className="bg-black/25 border border-white/8 rounded-xl p-4 my-2 relative group">
      <div className="flex items-start justify-between gap-3">
        <div className="flex-1 min-w-0">
          {label && (
            <span className="text-[11px] text-white/50 font-medium uppercase tracking-wide mb-1.5 block">
              {label}
            </span>
          )}
          <pre className="text-[13px] font-mono text-white/90 whitespace-pre-wrap break-words leading-relaxed m-0">
            {text}
          </pre>
        </div>
        <button
          onClick={handleCopy}
          className="p-1.5 rounded-lg hover:bg-white/10 transition-colors duration-150 flex-shrink-0 opacity-0 group-hover:opacity-100 focus-ring"
          aria-label="Copy text"
        >
          {copied ? (
            <Check size={14} className="text-green-400" />
          ) : (
            <Copy size={14} className="text-white/50" />
          )}
        </button>
      </div>
    </div>
  );
}
