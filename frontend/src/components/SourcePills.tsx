import { useState } from "react";
import { ChevronDown, ChevronUp } from "lucide-react";
import { Source } from "../api/client";

interface SourcePillsProps {
  sources: Source[];
}

export default function SourcePills({ sources }: SourcePillsProps) {
  const [expanded, setExpanded] = useState<number | null>(null);

  if (!sources.length) return null;

  return (
    <div className="mt-2">
      <span className="text-[11px] text-white/45 font-medium uppercase tracking-wide">
        Sources
      </span>
      <div className="flex flex-wrap gap-1.5 mt-1.5">
        {sources.map((s) => (
          <div key={s.n} className="flex flex-col">
            <button
              onClick={() => setExpanded(expanded === s.n ? null : s.n)}
              className="inline-flex items-center gap-1 px-2.5 py-1 rounded-lg bg-white/10 border border-white/10 text-[12px] text-white/70 hover:bg-white/15 transition-colors duration-150 focus-ring"
              aria-label={`Source ${s.n}: ${s.bill_title}`}
            >
              <span className="text-accent-light font-medium">[{s.n}]</span>
              <span className="truncate max-w-[180px]">{s.bill_title}</span>
              {s.clause && (
                <span className="text-white/40 ml-0.5">
                  — {s.clause}
                </span>
              )}
              {expanded === s.n ? (
                <ChevronUp size={12} className="ml-0.5 flex-shrink-0" />
              ) : (
                <ChevronDown size={12} className="ml-0.5 flex-shrink-0" />
              )}
            </button>
            {expanded === s.n && (
              <div className="bg-black/25 border border-white/8 rounded-xl p-3 mt-1 text-[12px] text-white/70 leading-relaxed msg-enter">
                <div className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1">
                  <span className="text-white/40">Bill</span>
                  <span>{s.bill_title} (No. {s.bill_no}, {s.year})</span>
                  {s.ministry !== "?" && (
                    <>
                      <span className="text-white/40">Ministry</span>
                      <span>{s.ministry}</span>
                    </>
                  )}
                  <span className="text-white/40">Part</span>
                  <span>{s.clause || s.part || "—"}</span>
                  <span className="text-white/40">Tag</span>
                  <span>{s.tag}</span>
                  <span className="text-white/40">File</span>
                  <span className="break-all">{s.source_file}</span>
                </div>
                {s.text && (
                  <p className="mt-2 pt-2 border-t border-white/8 text-white/60 line-clamp-4">
                    {s.text}
                  </p>
                )}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
