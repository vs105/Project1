interface ChatHeaderProps {
  title: string;
}

export default function ChatHeader({ title }: ChatHeaderProps) {
  return (
    <div className="sticky top-0 z-10 px-2 pt-1 pb-2">
      <div className="bg-neutral-300/60 backdrop-blur-xl rounded-full px-5 py-3 flex items-center">
        <h1 className="text-sm font-medium text-ink-dark truncate">
          {title || "New conversation"}
        </h1>
      </div>
    </div>
  );
}
