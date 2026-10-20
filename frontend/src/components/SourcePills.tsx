
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
      <span
        className="text-[11px] font-medium uppercase tracking-wide"
        style={{ color: "#414141" }}
      >
        Sources
      </span>

      <div className="flex flex-wrap gap-1.5 mt-1.5">
        {sources.map((s) => (
          <div key={s.n} className="flex flex-col">
            <button
              onClick={() =>
                setExpanded(expanded === s.n ? null : s.n)
              }
              className="inline-flex items-center gap-1 px-2.5 py-1 rounded-lg border text-[12px] transition-colors duration-150 focus-ring"
              style={{
                backgroundColor: "#F2EEE5",
                borderColor: "#D8D0C4",
                color: "#242424",
              }}
              aria-label={`Source ${s.n}: ${s.bill_title}`}
            >
              <span
                className="font-medium"
                style={{ color: "#E1A46B" }}
              >
                [{s.n}]
              </span>

              <span className="truncate max-w-[180px]">
                {s.bill_title}
              </span>

              {s.clause && (
                <span
                  className="ml-0.5"
                  style={{ color: "#77716A" }}
                >
                  — {s.clause}
                </span>
              )}

              {expanded === s.n ? (
                <ChevronUp
                  size={12}
                  className="ml-0.5 flex-shrink-0"
                  color="#414141"
                />
              ) : (
                <ChevronDown
                  size={12}
                  className="ml-0.5 flex-shrink-0"
                  color="#414141"
                />
              )}
            </button>

            {expanded === s.n && (
              <div
                className="border rounded-xl p-3 mt-1 text-[12px] leading-relaxed msg-enter"
                style={{
                  backgroundColor: "#F2EEE5",
                  borderColor: "#D8D0C4",
                  color: "#242424",
                }}
              >
                <div className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1">
                  <span style={{ color: "#77716A" }}>Bill</span>
                  <span>
                    {s.bill_title} (No. {s.bill_no}, {s.year})
                  </span>

                  {s.ministry !== "?" && (
                    <>
                      <span style={{ color: "#77716A" }}>
                        Ministry
                      </span>
                      <span>{s.ministry}</span>
                    </>
                  )}

                  <span style={{ color: "#77716A" }}>Part</span>
                  <span>{s.clause || s.part || "—"}</span>

                  <span style={{ color: "#77716A" }}>Tag</span>
                  <span>{s.tag}</span>

                  <span style={{ color: "#77716A" }}>File</span>
                  <span className="break-all">{s.source_file}</span>
                </div>

                {s.text && (
                  <p
                    className="mt-2 pt-2 border-t line-clamp-4"
                    style={{
                      borderColor: "#D8D0C4",
                      color: "#414141",
                    }}
                  >
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
