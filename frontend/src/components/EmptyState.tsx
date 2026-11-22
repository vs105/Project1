
interface EmptyStateProps {
  examples: string[];
  onSelectExample: (text: string) => void;
}

export default function EmptyState({
  examples,
  onSelectExample,
}: EmptyStateProps) {
  const displayed = examples
    .filter((e) => e.toLowerCase() !== "hi")
    .slice(0, 3);

  return (
    <div className="flex-1 flex items-center justify-center px-6">
      <div className="glass-card rounded-3xl p-8 max-w-lg w-full text-center">
        <div className="w-14 h-14 rounded-full bg-ink-dark flex items-center justify-center mx-auto mb-4">
          <svg
            width="20"
            height="20"
            viewBox="0 0 16 16"
            fill="none"
            aria-hidden="true"
          >
            <path
              d="M3 3L13 13M13 3L3 13"
              stroke="#fff"
              strokeWidth="2"
              strokeLinecap="round"
            />
          </svg>
        </div>

        <h2 className="text-ink-dark font-medium text-lg mb-2">
          SansadBills
        </h2>

        <p className="text-sm text-ink-light/70 mb-6">
          Ask questions about Lok Sabha bills and parliamentary proceedings
          from 2004 to 2026.
        </p>

        {displayed.length > 0 && (
          <div className="flex flex-col gap-2">
            <p className="text-xs font-medium text-ink-light/70 text-left mb-1">
              TRY ASKING
            </p>

            {displayed.map((ex) => (
              <button
                key={ex}
                onClick={() => onSelectExample(ex)}
                className="w-full text-left px-4 py-3 rounded-xl bg-white/50 border border-white/50 text-sm text-ink-dark hover:bg-white/80 hover:border-white transition-all duration-150 focus-ring"
              >
                {ex}
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
