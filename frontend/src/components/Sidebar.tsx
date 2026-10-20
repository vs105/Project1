import { X } from "lucide-react";
import BrandCard from "./BrandCard";
import NavMenu from "./NavMenu";

interface SidebarProps {
  onNewChat: () => void;
  navActive: "chats" | "about";
  onNavSelect: (item: "chats" | "about") => void;
  /* Drawer mode for mobile */
  isDrawer?: boolean;
  onCloseDrawer?: () => void;
  children?: React.ReactNode; /* Chat history injected here at mobile */
}

export default function Sidebar({
  onNewChat,
  navActive,
  onNavSelect,
  isDrawer,
  onCloseDrawer,
  children,
}: SidebarProps) {
  const content = (
    <div className="flex flex-col gap-4 h-full">
      <BrandCard onNewChat={onNewChat} />
      <NavMenu active={navActive} onSelect={onNavSelect} />
      {children && <div className="flex-1 min-h-0">{children}</div>}
    </div>
  );

  if (isDrawer) {
    return (
      <>
        {/* Backdrop */}
        <div
          className="fixed inset-0 bg-black/30 backdrop-blur-sm z-40"
          onClick={onCloseDrawer}
          aria-hidden="true"
        />
        {/* Slide-over */}
        <div className="fixed inset-y-0 left-0 z-50 w-[300px] p-4 bg-cream overflow-y-auto hide-scrollbar">
          <div className="flex justify-end mb-2">
            <button
              onClick={onCloseDrawer}
              className="p-1.5 rounded-lg hover:bg-black/5 transition-colors duration-150 focus-ring"
              aria-label="Close sidebar"
            >
              <X size={20} className="text-ink-light" />
            </button>
          </div>
          {content}
        </div>
      </>
    );
  }

  return <div className="flex flex-col gap-4">{content}</div>;
}
