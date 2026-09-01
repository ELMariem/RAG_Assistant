// Reusable popup shell: dark overlay + centered white card. Clicking the overlay
// (not the card itself) closes it, same convention as most dialog libraries.
export default function Modal({ open, onClose, title, children }) {
  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 px-4"
      onClick={onClose}
    >
      <div
        className="bg-white rounded-2xl shadow-xl w-full max-w-sm p-6"
        onClick={(e) => e.stopPropagation()}
      >
        {title && <h2 className="text-base font-semibold text-ink mb-3">{title}</h2>}
        {children}
      </div>
    </div>
  );
}