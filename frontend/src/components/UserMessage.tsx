interface UserMessageProps {
  content: string;
  initials: string;
}

export default function UserMessage({ content, initials }: UserMessageProps) {
  return (
    <div className="flex justify-end gap-3 msg-enter">
      <div className="bg-white/70 backdrop-blur-md border border-white/60 rounded-2xl px-4 py-3 max-w-[60%] text-sm text-ink-dark leading-relaxed shadow-sm">
        {content}
      </div>
      <div
        className="w-9 h-9 rounded-full bg-neutral-200 flex items-center justify-center flex-shrink-0 text-xs font-medium text-ink-light"
        aria-hidden="true"
      >
        {initials}
      </div>
    </div>
  );
}
