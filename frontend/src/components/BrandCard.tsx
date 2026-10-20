interface BrandCardProps {
  onNewChat: () => void;
}

export default function BrandCard({ onNewChat }: BrandCardProps) {
  return (
    <div className="glass-card rounded-3xl p-5 flex flex-col items-center gap-4">
      {/* Logo: black circle with white X mark */}
      <div className="flex items-center gap-3 w-full">
        <div className="w-10 h-10 rounded-full bg-ink-dark flex items-center justify-center flex-shrink-0">
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden="true">
            <path d="M3 3L13 13M13 3L3 13" stroke="#fff" strokeWidth="2" strokeLinecap="round" />
          </svg>
        </div>
        <span className="text-ink-dark font-medium text-[15px] tracking-tight">
          SansadBills
        </span>
      </div>
      <button
        onClick={onNewChat}
        className="btn-accent w-full rounded-xl py-2.5 px-4 text-sm font-medium focus-ring"
        aria-label="Start a new chat"
      >
        + New chat
      </button>
    </div>
  );
}
