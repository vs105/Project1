import { useState, useRef, useEffect, type KeyboardEvent } from "react";
import { SendHorizonal } from "lucide-react";

interface ComposerProps {
  onSend: (text: string) => void;
  disabled: boolean;
  prefill?: string;
}

export default function Composer({ onSend, disabled, prefill }: ComposerProps) {
  const [text, setText] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (prefill) {
      setText(prefill);
      textareaRef.current?.focus();
    }
  }, [prefill]);

  /* Auto-grow textarea */
  useEffect(() => {
    const el = textareaRef.current;
    if (el) {
      el.style.height = "auto";
      el.style.height = Math.min(el.scrollHeight, 160) + "px";
    }
  }, [text]);

  const handleSend = () => {
    const trimmed = text.trim();
    if (!trimmed || disabled) return;
    onSend(trimmed);
    setText("");
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const canSend = text.trim().length > 0 && !disabled;

  return (
    <div className="sticky bottom-0 px-2 pb-2 pt-1">
      <div className="glass-card rounded-3xl px-4 py-3 flex items-end gap-3">
        <textarea
          ref={textareaRef}
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask a question about a bill or parliamentary proceeding"
          rows={1}
          className="flex-1 bg-transparent text-sm text-ink-dark placeholder-ink-light/40 resize-none outline-none leading-relaxed max-h-[160px]"
          disabled={disabled}
          aria-label="Message input"
        />
        <button
          onClick={handleSend}
          disabled={!canSend}
          className="btn-accent rounded-xl w-10 h-10 flex items-center justify-center flex-shrink-0 focus-ring"
          aria-label="Send message"
        >
          <SendHorizonal size={18} />
        </button>
      </div>
      <p className="text-[11px] text-ink-light/40 text-center mt-2 px-4">
        Responses are generated from the available data and may contain errors. Please verify with official sources.
      </p>
    </div>
  );
}
