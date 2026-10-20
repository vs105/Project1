import { Conversation, groupByDate } from "../types";

interface ChatHistoryProps {
  conversations: Conversation[];
  activeId: string;
  onSelect: (id: string) => void;
}

export default function ChatHistory({
  conversations,
  activeId,
  onSelect,
}: ChatHistoryProps) {
  const sorted = [...conversations].sort((a, b) => b.createdAt - a.createdAt);
  const groups = groupByDate(sorted);

  return (
    <div className="glass-card rounded-3xl p-4 flex flex-col h-full overflow-hidden">
      <h2 className="text-sm font-medium text-ink-dark mb-3 px-1">Chat history</h2>
      <div className="flex-1 overflow-y-auto hide-scrollbar space-y-4">
        {groups.length === 0 && (
          <p className="text-[12px] text-ink-light/50 px-1">No conversations yet.</p>
        )}
        {groups.map(({ label, items }) => (
          <div key={label}>
            <p className="text-[11px] text-ink-light/45 font-medium uppercase tracking-wide px-1 mb-1.5">
              {label}
            </p>
            <div className="space-y-0.5">
              {items.map((c) => (
                <button
                  key={c.id}
                  onClick={() => onSelect(c.id)}
                  className={`w-full text-left px-3 py-2 rounded-xl text-[13px] truncate transition-colors duration-150 focus-ring ${
                    c.id === activeId
                      ? "bg-white/70 text-ink-dark font-medium"
                      : "text-ink-light/70 hover:bg-white/40 hover:text-ink-dark"
                  }`}
                  title={c.title}
                >
                  <span className="block truncate">{c.title}</span>
                  <span className="block text-[10px] text-ink-light/40 mt-0.5">
                    {new Date(c.createdAt).toLocaleDateString(undefined, {
                      month: "short",
                      day: "numeric",
                    })}
                  </span>
                </button>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
