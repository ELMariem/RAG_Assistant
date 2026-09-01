import { useEffect, useState } from "react";
import { User } from "lucide-react";
import Sidebar from "../components/Sidebar";
import * as api from "../api/client";

export default function UserProfile() {
  const [info, setInfo] = useState(null);

  useEffect(() => {
    api.me().then(setInfo).catch(() => {});
  }, []);

  return (
    <div className="flex h-screen bg-canvas">
      <Sidebar conversations={[]} onNewChat={() => {}} onSelectChat={() => {}} />
      <div className="flex-1 flex flex-col min-w-0">
        <div className="px-6 py-4 bg-white border-b border-gray-200">
          <h1 className="text-sm font-medium text-ink">User Profile</h1>
        </div>
        <div className="flex-1 overflow-y-auto px-6 py-6">
          <div className="max-w-md mx-auto bg-white rounded-xl shadow-sm p-6 flex flex-col items-center text-center">
            <div className="w-14 h-14 rounded-full bg-accent flex items-center justify-center mb-3">
              <User size={22} className="text-white" />
            </div>
            <h2 className="text-sm font-medium text-ink">{info?.user_id || "..."}</h2>
            <p className="text-xs text-muted mt-1">
              {info?.token_valid ? "Session valid" : "Checking session..."}
            </p>
            {info?.server_time_utc && (
              <p className="text-[11px] text-muted mt-4">Server time (UTC): {info.server_time_utc}</p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
