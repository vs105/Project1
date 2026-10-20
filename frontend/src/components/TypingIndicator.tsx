export default function TypingIndicator() {
  return (
    <div className="flex justify-start gap-3 msg-enter">
      <div
        className="w-9 h-9 rounded-full bg-ink-dark flex items-center justify-center flex-shrink-0"
        aria-hidden="true"
      >
        <svg width="14" height="14" viewBox="0 0 16 16" fill="none">
          <path d="M3 3L13 13M13 3L3 13" stroke="#fff" strokeWidth="2" strokeLinecap="round" />
        </svg>
      </div>
      <div className="ink-bubble rounded-2xl px-5 py-3.5 flex gap-1.5 items-center">
        <span className="w-2 h-2 rounded-full bg-white/40 animate-bounce [animation-delay:0ms]" />
        <span className="w-2 h-2 rounded-full bg-white/40 animate-bounce [animation-delay:150ms]" />
        <span className="w-2 h-2 rounded-full bg-white/40 animate-bounce [animation-delay:300ms]" />
      </div>
    </div>
  );
}
