import { useState, useEffect, useCallback } from "react";
import { Menu } from "lucide-react";
import { fetchExamples, askQuestion } from "./api/client";
import { Conversation, Message, uid, deriveTitle } from "./types";
import Sidebar from "./components/Sidebar";
import ChatHeader from "./components/ChatHeader";
import MessageList from "./components/MessageList";
import Composer from "./components/Composer";
import EmptyState from "./components/EmptyState";
import ChatHistory from "./components/ChatHistory";
import AboutModal from "./components/AboutModal";

/* ── Responsive breakpoints (match Tailwind defaults) ─────── */
function useMediaQuery(query: string) {
  const [matches, setMatches] = useState(() =>
    typeof window !== "undefined" ? window.matchMedia(query).matches : true
  );
  useEffect(() => {
    const mq = window.matchMedia(query);
    const handler = (e: MediaQueryListEvent) => setMatches(e.matches);
    mq.addEventListener("change", handler);
    return () => mq.removeEventListener("change", handler);
  }, [query]);
  return matches;
}

export default function App() {
  /* ── State ───────────────────────────────────────────────── */
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeId, setActiveId] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const [examples, setExamples] = useState<string[]>([]);
  const [prefill, setPrefill] = useState("");
  const [aboutOpen, setAboutOpen] = useState(false);
  const [navActive, setNavActive] = useState<"chats" | "about">("chats");
  const [drawerOpen, setDrawerOpen] = useState(false);

  const isDesktop = useMediaQuery("(min-width: 1280px)");
  const isTablet = useMediaQuery("(min-width: 768px)");

  /* ── Boot: fetch examples ────────────────────────────────── */
  useEffect(() => {
    fetchExamples()
      .then((r) => setExamples(r.examples))
      .catch(() => {
        setExamples([
          "What is the Finance Bill 2024 about?",
          "Which communities are covered under the Citizenship Amendment Act?",
          "How many bills were assented in 2019?",
        ]);
      });
  }, []);

  /* ── Active conversation ─────────────────────────────────── */
  const active = conversations.find((c) => c.id === activeId);

  /* ── Build history tuples for the backend ─────────────────── */
  const buildHistory = useCallback(
    (msgs: Message[]): [string, string][] => {
      const pairs: [string, string][] = [];
      for (let i = 0; i < msgs.length - 1; i += 2) {
        if (msgs[i].role === "user" && msgs[i + 1]?.role === "assistant") {
          pairs.push([msgs[i].content, msgs[i + 1].content]);
        }
      }
      return pairs.slice(-4);
    },
    []
  );

  /* ── New chat ────────────────────────────────────────────── */
  const handleNewChat = useCallback(() => {
    const c: Conversation = {
      id: uid(),
      title: "",
      messages: [],
      createdAt: Date.now(),
    };
    setConversations((prev) => [c, ...prev]);
    setActiveId(c.id);
    setDrawerOpen(false);
  }, []);

  /* ── Select a conversation ───────────────────────────────── */
  const handleSelectConversation = useCallback((id: string) => {
    setActiveId(id);
    setDrawerOpen(false);
  }, []);

  /* ── Send a message ──────────────────────────────────────── */
  const handleSend = useCallback(
    async (text: string) => {
      let targetId = activeId;

      /* If no active conversation, create one */
      if (!targetId) {
        const c: Conversation = {
          id: uid(),
          title: deriveTitle(text),
          messages: [],
          createdAt: Date.now(),
        };
        setConversations((prev) => [c, ...prev]);
        setActiveId(c.id);
        targetId = c.id;
      }

      const userMsg: Message = { id: uid(), role: "user", content: text };

      setConversations((prev) =>
        prev.map((c) => {
          if (c.id !== targetId) return c;
          const updated = {
            ...c,
            messages: [...c.messages, userMsg],
            title: c.title || deriveTitle(text),
          };
          return updated;
        })
      );

      setLoading(true);
      setPrefill("");

      try {
        const current = conversations.find((c) => c.id === targetId);
        const history = buildHistory(current?.messages || []);

        const res = await askQuestion(text, history);

        if (res.error) {
          const errMsg: Message = {
            id: uid(),
            role: "assistant",
            content: res.error,
            error: true,
          };
          setConversations((prev) =>
            prev.map((c) =>
              c.id === targetId
                ? { ...c, messages: [...c.messages, errMsg] }
                : c
            )
          );
        } else {
          const aiMsg: Message = {
            id: uid(),
            role: "assistant",
            content: res.reply,
            sources: res.sources,
          };
          setConversations((prev) =>
            prev.map((c) =>
              c.id === targetId
                ? { ...c, messages: [...c.messages, aiMsg] }
                : c
            )
          );
        }
      } catch (err: any) {
        const errMsg: Message = {
          id: uid(),
          role: "assistant",
          content:
            err?.message || "The request could not be completed. Please try again.",
          error: true,
        };
        setConversations((prev) =>
          prev.map((c) =>
            c.id === targetId
              ? { ...c, messages: [...c.messages, errMsg] }
              : c
          )
        );
      } finally {
        setLoading(false);
      }
    },
    [activeId, conversations, buildHistory]
  );

  /* ── Retry a failed message ──────────────────────────────── */
  const handleRetry = useCallback(
    (messageId: string) => {
      if (!active) return;
      const idx = active.messages.findIndex((m) => m.id === messageId);
      if (idx < 1) return;
      const prevUser = active.messages[idx - 1];
      if (prevUser.role !== "user") return;

      /* Remove the error message */
      setConversations((prev) =>
        prev.map((c) =>
          c.id === activeId
            ? { ...c, messages: c.messages.filter((m) => m.id !== messageId) }
            : c
        )
      );

      /* Re-send */
      handleSend(prevUser.content);
    },
    [active, activeId, handleSend]
  );

  /* ── Nav selection ───────────────────────────────────────── */
  const handleNavSelect = (item: "chats" | "about") => {
    setNavActive(item);
    if (item === "about") setAboutOpen(true);
  };

  /* ── Example selection ───────────────────────────────────── */
  const handleSelectExample = (text: string) => {
    setPrefill(text);
  };

  /* ── Chat history component ──────────────────────────────── */
  const chatHistoryEl = (
    <ChatHistory
      conversations={conversations}
      activeId={activeId}
      onSelect={handleSelectConversation}
    />
  );

  /* ── Render ──────────────────────────────────────────────── */
  return (
    <div className="h-screen p-3 md:p-4 bg-page-backdrop">
      <div className="satin-bg rounded-[28px] h-full overflow-hidden">
        {/* Mobile hamburger */}
        {!isTablet && (
          <div className="sticky top-0 z-30 p-3 flex items-center">
            <button
              onClick={() => setDrawerOpen(true)}
              className="p-2 rounded-xl glass-card hover:bg-white/70 transition-colors duration-150 focus-ring"
              aria-label="Open menu"
            >
              <Menu size={20} className="text-ink-dark" />
            </button>
          </div>
        )}

        {/* Mobile drawer */}
        {!isTablet && drawerOpen && (
          <Sidebar
            onNewChat={handleNewChat}
            navActive={navActive}
            onNavSelect={handleNavSelect}
            isDrawer
            onCloseDrawer={() => setDrawerOpen(false)}
          >
            {chatHistoryEl}
          </Sidebar>
        )}

        {/* Main grid */}
        <div
          className={`h-full ${
            isTablet
              ? isDesktop
                ? "grid grid-cols-[280px_1fr_280px] gap-4 p-4"
                : "grid grid-cols-[280px_1fr] gap-4 p-4"
              : "flex flex-col p-3 pt-0"
          }`}
        >
          {/* Left column (tablet+) */}
          {isTablet && (
            <div className="flex flex-col gap-4 overflow-y-auto hide-scrollbar">
              <Sidebar
                onNewChat={handleNewChat}
                navActive={navActive}
                onNavSelect={handleNavSelect}
              />
              {/* At tablet (no right column), put chat history here */}
              {!isDesktop && (
                <div className="flex-1 min-h-0">{chatHistoryEl}</div>
              )}
            </div>
          )}

          {/* Center column */}
          <div className="flex flex-col min-h-0 overflow-hidden">
            <ChatHeader title={active?.title || ""} />
            {active && active.messages.length > 0 ? (
              <MessageList
                messages={active.messages}
                loading={loading}
                onRetry={handleRetry}
              />
            ) : (
              <EmptyState
                examples={examples}
                onSelectExample={handleSelectExample}
              />
            )}
            <Composer
              onSend={handleSend}
              disabled={loading}
              prefill={prefill}
            />
          </div>

          {/* Right column (desktop only) */}
          {isDesktop && (
            <div className="overflow-hidden">{chatHistoryEl}</div>
          )}
        </div>
      </div>

      {/* About modal */}
      <AboutModal open={aboutOpen} onClose={() => setAboutOpen(false)} />
    </div>
  );
}
