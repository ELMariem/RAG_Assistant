import { useEffect, useRef, useState } from "react";
import { Upload, FileText, Trash2, CheckCircle2, CircleDashed } from "lucide-react";
import Sidebar from "../components/Sidebar";
import * as api from "../api/client";

export default function KnowledgeDocuments() {
  const [documents, setDocuments] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState("");
  const fileRef = useRef(null);

  async function refresh() {
    const { documents } = await api.listDocuments();
    setDocuments(documents);
  }

  useEffect(() => { refresh(); }, []);

  async function handleUpload(e) {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploading(true);
    setError("");
    try {
      await api.uploadDocument(file);
      await refresh();
    } catch (err) {
      setError(err.message);
    } finally {
      setUploading(false);
      e.target.value = "";
    }
  }

  async function handleDelete(filename) {
    await api.deleteDocument(filename);
    refresh();
  }

  return (
    <div className="flex h-screen bg-canvas">
      <Sidebar conversations={[]} onNewChat={() => {}} onSelectChat={() => {}} />

      <div className="flex-1 flex flex-col min-w-0">
        <div className="flex items-center justify-between px-6 py-4 bg-white border-b border-gray-200">
          <h1 className="text-sm font-medium text-ink">Knowledge Documents</h1>
          <button
            onClick={() => fileRef.current?.click()}
            disabled={uploading}
            className="flex items-center gap-2 text-xs font-medium text-white bg-accent hover:bg-accent-hover
                       disabled:bg-gray-300 rounded-lg px-3 py-2 transition-colors"
          >
            <Upload size={14} /> {uploading ? "Uploading..." : "Upload document"}
          </button>
          <input ref={fileRef} type="file" accept=".pdf,.docx" onChange={handleUpload} className="hidden" />
        </div>

        <div className="flex-1 overflow-y-auto px-6 py-6">
          <div className="max-w-2xl mx-auto flex flex-col gap-2">
            {error && <p className="text-xs text-red-500 mb-2">{error}</p>}
            {documents.length === 0 && (
              <div className="text-center text-muted text-sm mt-20">
                No documents yet. Upload a PDF or DOCX to make it searchable.
              </div>
            )}
            {documents.map((doc) => (
              <div
                key={doc.filename}
                className="flex items-center justify-between bg-white rounded-xl px-4 py-3 shadow-sm"
              >
                <div className="flex items-center gap-3 min-w-0">
                  <FileText size={18} className="text-accent shrink-0" />
                  <span className="text-sm text-ink truncate">{doc.filename}</span>
                  {doc.indexed ? (
                    <span title="Indexed"><CheckCircle2 size={14} className="text-green-500 shrink-0" /></span>
                  ) : (
                    <span title="Not indexed yet"><CircleDashed size={14} className="text-muted shrink-0" /></span>
                  )}
                </div>
                <button
                  onClick={() => handleDelete(doc.filename)}
                  className="text-gray-400 hover:text-red-500 transition-colors shrink-0"
                  title="Delete"
                >
                  <Trash2 size={16} />
                </button>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
