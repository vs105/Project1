import { MessageSquare, Info } from "lucide-react";

type NavItem = "chats" | "about";

interface NavMenuProps {
  active: NavItem;
  onSelect: (item: NavItem) => void;
}

export default function NavMenu({ active, onSelect }: NavMenuProps) {
  const items: { id: NavItem; label: string; icon: typeof MessageSquare }[] = [
    { id: "chats", label: "Chats", icon: MessageSquare },
    { id: "about", label: "About", icon: Info },
  ];

  return (
    <div className="glass-card rounded-3xl p-3 flex flex-col gap-1">
      {items.map(({ id, label, icon: Icon }) => (
        <button
          key={id}
          onClick={() => onSelect(id)}
          className={`flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium transition-colors duration-150 focus-ring w-full text-left ${
            active === id
              ? "bg-white/70 text-ink-dark"
              : "text-ink-light/70 hover:bg-white/40 hover:text-ink-dark"
          }`}
          aria-label={label}
        >
          <Icon size={18} strokeWidth={1.8} />
          {label}
        </button>
      ))}
    </div>
  );
}
