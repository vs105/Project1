import { useRef, useEffect } from "react";
import { Message } from "../types";
import UserMessage from "./UserMessage";
import AiMessage from "./AiMessage";
import TypingIndicator from "./TypingIndicator";

interface MessageListProps {
  messages: Message[];
  loading: boolean;
  onRetry?: (messageId: string) => void;
}

export default function MessageList({ messages, loading, onRetry }: MessageListProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  return (
    <div className="flex-1 overflow-y-auto hide-scrollbar px-4 py-4 space-y-4">
      {messages.map((m) =>
        m.role === "user" ? (
          <UserMessage key={m.id} content={m.content} initials="U" />
        ) : (
          <AiMessage
            key={m.id}
            content={m.content}
            sources={m.sources}
            error={m.error}
            onRetry={m.error && onRetry ? () => onRetry(m.id) : undefined}
          />
        )
      )}
      {loading && <TypingIndicator />}
      <div ref={bottomRef} />
    </div>
  );
}
