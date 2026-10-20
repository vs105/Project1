
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
    <div
      className="border rounded-xl p-4 my-2 relative group"
      style={{
        backgroundColor: "#F2EEE5",
        borderColor: "#D8D0C4",
      }}
    >
      <div className="flex items-start justify-between gap-3">
        <div className="flex-1 min-w-0">
          {label && (
            <span
              className="text-[11px] font-medium uppercase tracking-wide mb-1.5 block"
              style={{ color: "#414141" }}
            >
              {label}
            </span>
          )}

          <pre
            className="text-[13px] font-mono whitespace-pre-wrap break-words leading-relaxed m-0"
            style={{
              color: "#242424",
              backgroundColor: "transparent",
              opacity: 1,
            }}
          >
            {text}
          </pre>
        </div>

        <button
          onClick={handleCopy}
          className="p-1.5 rounded-lg hover:bg-black/10 transition-colors duration-150 flex-shrink-0 opacity-0 group-hover:opacity-100 focus-ring"
          style={{ color: "#414141" }}
          aria-label={copied ? "Copied" : "Copy text"}
          title={copied ? "Copied" : "Copy text"}
        >
          {copied ? (
            <Check size={14} color="#15803D" />
          ) : (
            <Copy size={14} color="#414141" />
          )}
        </button>
      </div>
    </div>
  );
}
