import { useEffect, useState } from "react";
import { Copy, RotateCw, FileText, ImageOff } from "lucide-react";
import GalaxyIcon from "./icons/GalaxyIcon";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import * as api from "../api/client";

// Maps markdown elements to the app's own Tailwind tokens (text-ink, text-accent, etc.)
// instead of pulling in @tailwindcss/typography's generic "prose" styles, which
// wouldn't match the existing color system.
const markdownComponents = {
  p: ({ node, ...props }) => <p className="mb-2 last:mb-0 leading-relaxed" {...props} />,
  ul: ({ node, ...props }) => <ul className="list-disc pl-5 mb-2 last:mb-0 space-y-1" {...props} />,
  ol: ({ node, ...props }) => <ol className="list-decimal pl-5 mb-2 last:mb-0 space-y-1" {...props} />,
  li: ({ node, ...props }) => <li className="leading-relaxed" {...props} />,
  strong: ({ node, ...props }) => <strong className="font-semibold text-ink" {...props} />,
  em: ({ node, ...props }) => <em className="italic" {...props} />,
  a: ({ node, ...props }) => (
    <a className="text-accent underline hover:text-accent-hover" target="_blank" rel="noopener noreferrer" {...props} />
  ),
  code: ({ node, ...props }) => (
    <code className="bg-canvas/70 rounded px-1 py-0.5 text-[13px] font-mono" {...props} />
  ),
  pre: ({ node, ...props }) => (
    <pre className="bg-canvas text-ink rounded-lg p-3 text-[13px] font-mono overflow-x-auto mb-2 last:mb-0" {...props} />
  ),
  blockquote: ({ node, ...props }) => (
    <blockquote className="border-l-2 border-accent/30 pl-3 italic text-muted mb-2 last:mb-0" {...props} />
  ),
  h1: ({ node, ...props }) => <p className="font-semibold text-ink mt-1 mb-1" {...props} />,
  h2: ({ node, ...props }) => <p className="font-semibold text-ink mt-1 mb-1" {...props} />,
  h3: ({ node, ...props }) => <p className="font-semibold text-ink mt-1 mb-1" {...props} />,
};

// Fetches one diagram image (as an authenticated blob URL) and renders it as a
// thumbnail. Isolated in its own component so each image loads/fails independently
// and the blob URL gets revoked when this particular image is no longer shown.
function DiagramImage({ filename }) {
  const [url, setUrl] = useState(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    let objectUrl = null;
    let cancelled = false;

    api
      .fetchDocumentImageUrl(filename)
      .then((u) => {
        if (cancelled) return;
        objectUrl = u;
        setUrl(u);
      })
      .catch(() => {
        if (!cancelled) setFailed(true);
      });

    return () => {
      cancelled = true;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [filename]);

  if (failed) {
    return (
      <div className="flex items-center gap-1.5 text-[11px] text-muted bg-canvas rounded-lg px-2.5 py-2">
        <ImageOff size={12} /> Image unavailable
      </div>
    );
  }
  if (!url) {
    return <div className="w-40 h-28 rounded-lg bg-canvas animate-pulse" />;
  }
  return (
    <a href={url} target="_blank" rel="noopener noreferrer">
      <img
        src={url}
        alt="Diagram from source document"
        className="max-h-48 rounded-lg border border-gray-100 hover:opacity-90 transition-opacity"
      />
    </a>
  );
}

export default function MessageBubble({ role, content, sources, isStreaming }) {
  const isUser = role === "user";

  if (isUser) {
    return (
      <div className="flex justify-end">
        <div className="max-w-[70%] bg-white rounded-2xl rounded-tr-sm px-4 py-3 shadow-sm">
          <p className="text-sm text-ink whitespace-pre-wrap">{content}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex justify-start gap-3 max-w-[80%]">
      <div className="w-8 h-8 rounded-lg bg-accent flex items-center justify-center shrink-0 mt-1">
        <GalaxyIcon size={16} className="text-white" />
      </div>
      <div className="bg-white rounded-2xl rounded-tl-sm px-4 py-3 shadow-sm">
        <div className="text-sm text-ink">
          <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
            {isStreaming ? `${content || ""}▍` : content || ""}
          </ReactMarkdown>
        </div>
        {!isStreaming && content && (
          <div className="mt-2 pt-2 border-t border-gray-100">
            {sources && sources.length > 0 && (
              <div className="flex flex-wrap gap-1.5 mb-2">
                {sources.map((s, idx) => (
                  <span
                    key={idx}
                    title={s.file}
                    className="inline-flex items-center gap-1 text-[11px] text-muted bg-canvas rounded-md px-2 py-1"
                  >
                    <FileText size={11} className="shrink-0" />
                    <span className="truncate max-w-[140px]">{s.file}</span>
                    {s.page != null && <span className="text-gray-400">· p.{s.page}</span>}
                  </span>
                ))}
              </div>
            )}
            {sources && sources.some((s) => s.image_filename) && (
              <div className="flex flex-wrap gap-2 mb-2">
                {sources
                  .filter((s) => s.image_filename)
                  .map((s, idx) => (
                    <DiagramImage key={s.image_filename || idx} filename={s.image_filename} />
                  ))}
              </div>
            )}
            <div className="flex items-center gap-3">
              <button
                onClick={() => navigator.clipboard.writeText(content)}
                className="text-gray-400 hover:text-ink transition-colors"
                title="Copy"
              >
                <Copy size={13} />
              </button>
              <button className="text-gray-400 hover:text-ink transition-colors" title="Regenerate">
                <RotateCw size={13} />
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}