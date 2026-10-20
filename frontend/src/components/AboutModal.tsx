import { X } from "lucide-react";

interface AboutModalProps {
  open: boolean;
  onClose: () => void;
}

export default function AboutModal({ open, onClose }: AboutModalProps) {
  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/30 backdrop-blur-sm"
        onClick={onClose}
        aria-hidden="true"
      />
      {/* Modal */}
      <div className="glass-card rounded-3xl p-6 max-w-md w-full relative z-10">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-ink-dark font-medium text-[15px]">About SansadBills</h2>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg hover:bg-black/5 transition-colors duration-150 focus-ring"
            aria-label="Close about dialog"
          >
            <X size={18} className="text-ink-light" />
          </button>
        </div>
        <div className="text-sm text-ink-light leading-relaxed space-y-3">
          <p>
            SansadBills is a question-answering system built on official Lok Sabha bill
            documents from 2004 to 2026. It retrieves relevant clauses, preambles and
            metadata from the corpus and generates answers grounded in the source text.
          </p>
          <p>
            Answers are generated from the available data and may contain errors. Every
            response cites the specific bill extracts it drew from. Please verify important
            information with official sources.
          </p>
        </div>
      </div>
    </div>
  );
}
