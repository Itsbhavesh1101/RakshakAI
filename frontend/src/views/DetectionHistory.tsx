import { useState } from 'react';
import { 
  Search, Trash2, Calendar, ShieldAlert, 
  Eye, Download, RefreshCw, AlertOctagon, Flame,
  Crosshair, Users, Shield, ExternalLink
} from 'lucide-react';
import type { LucideIcon } from 'lucide-react';

interface AlertLog {
  id: number;
  timestamp: string;
  module: string;
  severity: string;
  message: string;
  snapshot_path: string;
  ai_description: string | null;
}

interface DetectionHistoryProps {
  alertLogs: AlertLog[];
  snapshotUrl: (path: string) => string;
  clearAlertLogs: () => Promise<void>;
}

const normalizeModuleKey = (module: string) => (
  module
    .toLowerCase()
    .replace(/\s+/g, '_')
    .replace(/^trespassing_detection$/, 'trespassing_detection')
);

const MODULE_ICONS: Record<string, LucideIcon> = {
  'weapon_detection': Crosshair,
  'fire_detection': Flame,
  'fall_detection': AlertOctagon,
  'crowd_detection': Users,
  'trespassing_detection': Shield,
};

const MODULE_LABELS: Record<string, string> = {
  'weapon_detection': 'Weapon Threat',
  'fire_detection': 'Fire Hazard',
  'fall_detection': 'Person Down',
  'crowd_detection': 'Crowd Detection',
  'trespassing_detection': 'Restricted Intrusion',
};

export default function DetectionHistory({ alertLogs, snapshotUrl, clearAlertLogs }: DetectionHistoryProps) {
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedSeverity, setSelectedSeverity] = useState<string>('ALL');
  const [selectedModule, setSelectedModule] = useState<string>('ALL');
  const [selectedAlert, setSelectedAlert] = useState<AlertLog | null>(null);

  // Filter logs
  const filteredLogs = alertLogs.filter(log => {
    const matchesSearch = 
      log.module.toLowerCase().includes(searchTerm.toLowerCase()) || 
      log.message.toLowerCase().includes(searchTerm.toLowerCase()) ||
      (log.ai_description && log.ai_description.toLowerCase().includes(searchTerm.toLowerCase()));
      
    const matchesSeverity = selectedSeverity === 'ALL' || log.severity === selectedSeverity;
    
    // Normalize module names for matching
    const normalizedModule = normalizeModuleKey(log.module);
    const matchesModule = selectedModule === 'ALL' || 
      normalizedModule.includes(selectedModule.toLowerCase().replace('_detection', ''));

    return matchesSearch && matchesSeverity && matchesModule;
  });

  // Calculate statistics based on current logs
  const totalCount = alertLogs.length;
  const criticalCount = alertLogs.filter(l => l.severity === 'CRITICAL').length;
  const warningCount = alertLogs.filter(l => l.severity === 'WARNING').length;

  return (
    <div className="flex-1 flex gap-6 overflow-hidden animate-in fade-in duration-300 h-full relative">
      
      {/* A. LEFT AREA: Filters & History Logs Table List */}
      <div className="flex-1 flex flex-col gap-5 overflow-hidden h-full">
        
        {/* 1. Dynamic Bento Stats Capsules */}
        <div className="grid grid-cols-3 gap-4 flex-shrink-0">
          <div className="bg-[#0c1a24]/30 backdrop-blur-xl border border-white/5 rounded-2xl p-4 flex justify-between items-center shadow-lg hover:border-teal-500/10 transition">
            <div className="space-y-0.5">
              <span className="text-[9px] text-zinc-500 uppercase tracking-widest font-extrabold">Total Incidents</span>
              <div className="text-xl font-bold tracking-wide text-white font-mono">{totalCount}</div>
              <div className="text-[9px] text-zinc-400 font-medium">Logged in SQLite DB</div>
            </div>
            <div className="p-2.5 bg-zinc-950/40 rounded-xl border border-white/5 text-teal-400">
              <RefreshCw className="w-4 h-4 animate-spin-slow" />
            </div>
          </div>

          <div className="bg-[#0c1a24]/30 backdrop-blur-xl border border-white/5 rounded-2xl p-4 flex justify-between items-center shadow-lg hover:border-teal-500/10 transition">
            <div className="space-y-0.5">
              <span className="text-[9px] text-zinc-500 uppercase tracking-widest font-extrabold">Critical Level</span>
              <div className="text-xl font-bold tracking-wide text-rose-400 font-mono">{criticalCount}</div>
              <div className="text-[9px] text-zinc-400 font-medium">Require immediate dispatch</div>
            </div>
            <div className="p-2.5 bg-zinc-950/40 rounded-xl border border-white/5 text-rose-400">
              <ShieldAlert className="w-4 h-4 animate-pulse" />
            </div>
          </div>

          <div className="bg-[#0c1a24]/30 backdrop-blur-xl border border-white/5 rounded-2xl p-4 flex justify-between items-center shadow-lg hover:border-teal-500/10 transition">
            <div className="space-y-0.5">
              <span className="text-[9px] text-zinc-500 uppercase tracking-widest font-extrabold">Warnings logged</span>
              <div className="text-xl font-bold tracking-wide text-amber-400 font-mono">{warningCount}</div>
              <div className="text-[9px] text-zinc-400 font-medium">Low threat anomalies</div>
            </div>
            <div className="p-2.5 bg-zinc-950/40 rounded-xl border border-white/5 text-amber-400">
              <Calendar className="w-4 h-4" />
            </div>
          </div>
        </div>

        {/* 2. Unified Search & Filter Panel */}
        <div className="bg-[#0c1a24]/20 backdrop-blur-xl border border-white/5 rounded-2xl p-4 flex flex-col md:flex-row gap-4 items-center justify-between flex-shrink-0 shadow-lg">
          <div className="relative w-full md:w-72">
            <Search className="w-4 h-4 text-zinc-500 absolute left-3.5 top-1/2 -translate-y-1/2" />
            <input 
              type="text" 
              placeholder="Search by keywords, details..." 
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full bg-[#040c12]/60 border border-white/5 rounded-xl pl-9 pr-4 py-2 text-xs text-white placeholder-zinc-500 focus:outline-none focus:border-teal-500/50 font-semibold"
            />
          </div>

          <div className="flex gap-3 w-full md:w-auto items-center justify-end">
            {/* Severity Filter */}
            <select 
              value={selectedSeverity} 
              onChange={(e) => setSelectedSeverity(e.target.value)}
              className="bg-[#0c1a24]/80 border border-white/5 text-zinc-300 px-3 py-1.5 rounded-xl text-xs focus:outline-none focus:border-teal-500/50 font-bold uppercase"
            >
              <option value="ALL">All Severities</option>
              <option value="CRITICAL">Critical Only</option>
              <option value="WARNING">Warning Only</option>
              <option value="INFO">Info Only</option>
            </select>

            {/* Module Category Filter */}
            <select 
              value={selectedModule} 
              onChange={(e) => setSelectedModule(e.target.value)}
              className="bg-[#0c1a24]/80 border border-white/5 text-zinc-300 px-3 py-1.5 rounded-xl text-xs focus:outline-none focus:border-teal-500/50 font-bold uppercase"
            >
              <option value="ALL">All Threat Classes</option>
              <option value="weapon_detection">Weapon Threats</option>
              <option value="fire_detection">Fire / Smoke</option>
              <option value="fall_detection">Fall Events</option>
              <option value="trespassing_detection">Intrusion</option>
              <option value="crowd_detection">Crowd Anomalies</option>
            </select>

            {/* Clear Database Actions */}
            <button 
              onClick={clearAlertLogs}
              className="px-3.5 py-1.5 rounded-xl border border-rose-500/20 text-rose-400 hover:bg-rose-500/10 active:scale-95 text-xs font-bold transition flex items-center gap-1.5 shadow-lg"
            >
              <Trash2 className="w-3.5 h-3.5" />
              Wipe DB Logs
            </button>
          </div>
        </div>

        {/* 3. History logs table list */}
        <div className="flex-1 bg-[#0c1a24]/20 backdrop-blur-xl border border-white/5 rounded-3xl overflow-hidden shadow-2xl flex flex-col min-h-0">
          <div className="flex-1 overflow-y-auto min-h-0">
            <table className="w-full text-left border-collapse text-xs select-none">
              <thead>
                <tr className="border-b border-white/5 bg-zinc-950/40 text-[9px] font-extrabold uppercase tracking-wider text-zinc-500 sticky top-0 z-10 backdrop-blur-md">
                  <th className="py-3.5 px-5 font-mono">ID</th>
                  <th className="py-3.5 px-5 font-mono">Date / Time</th>
                  <th className="py-3.5 px-5">Threat Type</th>
                  <th className="py-3.5 px-5">Alert Details</th>
                  <th className="py-3.5 px-5 text-center">Severity</th>
                  <th className="py-3.5 px-5 text-right">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-white/5 font-semibold text-zinc-300">
                {filteredLogs.length === 0 ? (
                  <tr>
                    <td colSpan={6} className="text-center py-20 text-zinc-500 font-bold uppercase tracking-wide">
                      No logs matching selected filter criteria.
                    </td>
                  </tr>
                ) : (
                  filteredLogs.map((log) => {
                    // Match normalized module icons
                    const normalizedKey = normalizeModuleKey(log.module);
                    const MatchingIcon = MODULE_ICONS[normalizedKey] || ShieldAlert;
                    const cleanLabel = MODULE_LABELS[normalizedKey] || log.module;

                    return (
                      <tr 
                        key={log.id} 
                        onClick={() => setSelectedAlert(log)}
                        className={`hover:bg-white/[0.02] active:bg-white/[0.04] transition duration-200 cursor-pointer ${selectedAlert?.id === log.id ? 'bg-teal-950/10 border-l-2 border-l-teal-400' : ''}`}
                      >
                        <td className="py-3 px-5 text-zinc-500 font-mono font-bold">#{log.id}</td>
                        <td className="py-3 px-5 text-zinc-400 font-mono font-medium">{log.timestamp}</td>
                        <td className="py-3 px-5">
                          <div className="flex items-center gap-2">
                            <div className={`p-1.5 rounded-lg bg-zinc-950/40 border border-white/5 ${log.severity === 'CRITICAL' ? 'text-rose-400' : 'text-amber-400'}`}>
                              <MatchingIcon className="w-3.5 h-3.5" />
                            </div>
                            <span className="font-bold">{cleanLabel}</span>
                          </div>
                        </td>
                        <td className="py-3 px-5 text-zinc-400 truncate max-w-[200px]" title={log.message}>
                          {log.message}
                        </td>
                        <td className="py-3 px-5 text-center">
                          <span className={`text-[9px] font-extrabold px-2 py-0.5 rounded leading-none ${log.severity === 'CRITICAL' ? 'bg-rose-500/10 text-rose-400 border border-rose-500/20' : 'bg-amber-500/10 text-amber-400 border border-amber-500/20'}`}>
                            {log.severity}
                          </span>
                        </td>
                        <td className="py-3 px-5 text-right">
                          <button className="p-1 rounded bg-zinc-900/60 border border-white/5 text-zinc-400 hover:text-white transition hover:scale-105 active:scale-95">
                            <Eye className="w-3.5 h-3.5" />
                          </button>
                        </td>
                      </tr>
                    );
                  })
                )}
              </tbody>
            </table>
          </div>
        </div>

      </div>

      {/* B. RIGHT PANEL: Selected Log Event Viewer Details Card */}
      <div className="w-[30%] bg-[#0c1a24]/20 backdrop-blur-xl border border-white/5 rounded-3xl p-5 shadow-2xl flex flex-col justify-between overflow-y-auto h-full flex-shrink-0 animate-in slide-in-from-right-5 duration-300">
        {selectedAlert ? (
          <div className="flex flex-col gap-5 h-full justify-between">
            <div className="space-y-4">
              <div className="flex justify-between items-center pb-2 border-b border-white/5">
                <div>
                  <h3 className="text-sm font-bold text-white uppercase tracking-wider">Event Inspector</h3>
                  <p className="text-[8px] font-bold text-teal-400 uppercase tracking-widest font-mono">Alert Entity #{selectedAlert.id}</p>
                </div>
                <span className={`text-[9px] font-extrabold px-2 py-0.5 rounded leading-none ${selectedAlert.severity === 'CRITICAL' ? 'bg-rose-500/10 text-rose-400 border border-rose-500/20' : 'bg-amber-500/10 text-amber-400 border border-amber-500/20'}`}>
                  {selectedAlert.severity}
                </span>
              </div>

              {/* Event snapshot image with hover magnifier overlay */}
              <div className="relative aspect-video w-full rounded-2xl border border-white/5 overflow-hidden bg-zinc-950/60 group shadow-lg">
                {selectedAlert.snapshot_path ? (
                  <>
                    <img 
                      src={snapshotUrl(selectedAlert.snapshot_path)} 
                      alt="Threat Capture" 
                      className="w-full h-full object-cover group-hover:scale-105 transition duration-500"
                    />
                    <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-transparent opacity-100 group-hover:opacity-60 transition duration-300" />
                    <span className="absolute bottom-2 left-3 text-[8.5px] font-bold text-white bg-black/60 border border-white/5 px-2 py-0.5 rounded-full font-mono">
                      INGEST_CAM01
                    </span>
                  </>
                ) : (
                  <div className="flex flex-col items-center justify-center h-full text-zinc-600 font-bold uppercase tracking-wider text-[10px] gap-2">
                    <AlertOctagon className="w-5 h-5" />
                    No Snapshot Attached
                  </div>
                )}
              </div>

              {/* Telemetry info stack */}
              <div className="space-y-3 bg-[#040c12]/40 rounded-2xl border border-white/5 p-4 text-xs font-semibold">
                <div className="flex justify-between">
                  <span className="text-zinc-500">Incident Source:</span>
                  <span className="text-white font-bold">{selectedAlert.module}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-zinc-500">Trigger Time:</span>
                  <span className="text-zinc-300 font-mono font-medium">{selectedAlert.timestamp}</span>
                </div>
                <div className="flex flex-col gap-1 border-t border-white/5 pt-3">
                  <span className="text-zinc-500">Detector Core Event Message:</span>
                  <p className="text-zinc-300 leading-relaxed font-sans">{selectedAlert.message}</p>
                </div>
              </div>

              {/* Ollama/Moondream Vision LLM scene assessment */}
              <div className="space-y-2">
                <span className="text-[9px] text-zinc-500 uppercase tracking-widest font-extrabold block">AI Scene Interpretation Assessment</span>
                <div className="bg-teal-500/5 rounded-2xl border border-teal-500/10 p-4 text-xs leading-relaxed text-zinc-300 relative group overflow-hidden">
                  <div className="absolute inset-x-0 bottom-0 h-1 bg-gradient-to-r from-teal-500/20 to-cyan-500/20" />
                  <p className="font-sans italic">
                    {selectedAlert.ai_description || 
                      'Vision scene synthesis enrichment is queued or disabled. To enrich alerts, connect local Ollama service.'}
                  </p>
                </div>
              </div>
            </div>

            {/* Action Buttons */}
            <div className="flex gap-3 pt-3 border-t border-white/5">
              <a 
                href={selectedAlert.snapshot_path ? snapshotUrl(selectedAlert.snapshot_path) : '#'} 
                target="_blank" 
                rel="noreferrer"
                className="flex-1 py-2 rounded-xl text-[10px] font-bold bg-[#09090b]/60 border border-white/5 text-zinc-300 hover:text-white text-center hover:bg-zinc-900 active:scale-95 transition flex items-center justify-center gap-1.5 shadow-md"
              >
                <ExternalLink className="w-3 h-3" />
                Raw Snapshot
              </a>
              <button 
                onClick={() => {
                  if (selectedAlert.snapshot_path) {
                    const link = document.createElement('a');
                    link.href = snapshotUrl(selectedAlert.snapshot_path);
                    link.download = `threat_capture_${selectedAlert.id}.jpg`;
                    link.click();
                  }
                }}
                disabled={!selectedAlert.snapshot_path}
                className="flex-1 py-2 rounded-xl text-[10px] font-bold bg-teal-500 hover:bg-teal-400 disabled:opacity-50 text-[#040c12] hover:scale-[1.02] active:scale-95 transition flex items-center justify-center gap-1.5 shadow-lg shadow-teal-500/10"
              >
                <Download className="w-3 h-3" />
                Download JPEG
              </button>
            </div>
          </div>
        ) : (
          <div className="h-full flex flex-col items-center justify-center text-center text-zinc-600 gap-3 px-4 py-20 select-none">
            <div className="p-3 bg-zinc-950/40 rounded-full border border-white/5 animate-pulse-slow">
              <Eye className="w-6 h-6 text-zinc-500" strokeWidth={1.5} />
            </div>
            <div className="space-y-1">
              <div className="text-xs font-bold uppercase tracking-wider text-zinc-500">Operator Inspector Idle</div>
              <p className="text-[10px] leading-relaxed text-zinc-500 font-semibold uppercase">
                Select a threat log from the index to display snapshots, vision LLM context, and alert telemetry diagnostics.
              </p>
            </div>
          </div>
        )}
      </div>

    </div>
  );
}
