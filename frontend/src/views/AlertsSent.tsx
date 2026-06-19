import { useState } from 'react';
import { 
  ShieldAlert, Search, ChevronRight, CheckCircle, Wifi,
  MessageSquare, Smartphone, Signal, Battery, Eye
} from 'lucide-react';

interface AlertLog {
  id: number;
  timestamp: string;
  module: string;
  severity: string;
  message: string;
  snapshot_path: string;
  ai_description: string | null;
}

interface AlertsSentProps {
  alertLogs: AlertLog[];
  snapshotUrl: (path: string) => string;
  smsEnabled?: boolean;
  toPhone?: string;
}

export default function AlertsSent({ alertLogs, snapshotUrl, smsEnabled = false, toPhone = '' }: AlertsSentProps) {
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedSMS, setSelectedSMS] = useState<AlertLog | null>(null);

  // Filter logs based on search term
  const dispatchLogs = alertLogs.filter(log => {
    const matchesSearch = 
      log.module.toLowerCase().includes(searchTerm.toLowerCase()) || 
      log.message.toLowerCase().includes(searchTerm.toLowerCase()) ||
      (log.ai_description && log.ai_description.toLowerCase().includes(searchTerm.toLowerCase()));
    return matchesSearch;
  });

  const totalDispatched = dispatchLogs.length;
  const criticalDispatched = dispatchLogs.filter(d => d.severity === 'CRITICAL').length;

  // Generate standard Twilio SMS message body
  const getSMSBody = (log: AlertLog) => {
    const severity = log.severity.toUpperCase();
    const type = log.module.replace(' Detection', '').toUpperCase();
    const description = log.ai_description || log.message;
    return (
      `RAKSHAK ALERT: ${severity} THREAT\n` +
      `Type: ${type}\n` +
      `Time: ${log.timestamp}\n` +
      `Assessment: ${description}\n` +
      `Action Required: Please login to the security dashboard immediately.`
    );
  };

  return (
    <div className="flex-1 flex gap-6 overflow-hidden animate-in fade-in duration-300 h-full relative z-10">
      
      {/* A. LEFT AREA: Outbox Feed Table */}
      <div className="flex-1 flex flex-col gap-5 overflow-hidden h-full">
        
        {/* 1. Bento stats row */}
        <div className="grid grid-cols-3 gap-4 flex-shrink-0">
          {/* Card 1: Outbox Count */}
          <div className="bg-[#0c1a24]/30 backdrop-blur-xl border border-white/5 rounded-2xl p-4 flex justify-between items-center shadow-lg hover:border-teal-500/10 transition duration-300">
            <div className="space-y-0.5">
              <span className="text-[9px] text-zinc-500 uppercase tracking-widest font-extrabold">SMS Outbox Dispatch</span>
              <div className="text-xl font-bold tracking-wide text-white font-mono">{totalDispatched} Messages</div>
              <div className="text-[9px] text-teal-400 font-semibold uppercase tracking-wider font-mono">Dispatched successfully</div>
            </div>
            <div className="p-2.5 bg-teal-500/10 rounded-xl border border-teal-500/20 text-teal-400">
              <MessageSquare className="w-4 h-4" />
            </div>
          </div>

          {/* Card 2: Emergency Dispatched */}
          <div className="bg-[#0c1a24]/30 backdrop-blur-xl border border-white/5 rounded-2xl p-4 flex justify-between items-center shadow-lg hover:border-teal-500/10 transition duration-300">
            <div className="space-y-0.5">
              <span className="text-[9px] text-zinc-500 uppercase tracking-widest font-extrabold">Critical Vectors SMS</span>
              <div className="text-xl font-bold tracking-wide text-rose-400 font-mono">{criticalDispatched} Alarms</div>
              <div className="text-[9px] text-rose-400/80 font-bold uppercase tracking-wider text-[8px]">High priority overrides</div>
            </div>
            <div className="p-2.5 bg-rose-500/10 rounded-xl border border-rose-500/20 text-rose-400">
              <ShieldAlert className="w-4 h-4 animate-pulse" />
            </div>
          </div>

          {/* Card 3: Twilio Status */}
          <div className="bg-[#0c1a24]/30 backdrop-blur-xl border border-white/5 rounded-2xl p-4 flex justify-between items-center shadow-lg hover:border-teal-500/10 transition duration-300">
            <div className="space-y-0.5">
              <span className="text-[9px] text-zinc-500 uppercase tracking-widest font-extrabold">Twilio SMS Gateway</span>
              <div className={`text-sm font-bold tracking-wide font-mono ${smsEnabled ? 'text-teal-400' : 'text-amber-500'}`}>
                {smsEnabled ? 'GATEWAY CONNECTED' : 'STANDBY MODE'}
              </div>
              <div className="text-[9px] text-zinc-400 font-medium">
                {smsEnabled ? `Active: ${toPhone || 'operator term'}` : 'SMS dispatches paused'}
              </div>
            </div>
            <div className={`p-2.5 rounded-xl border ${smsEnabled ? 'bg-teal-500/10 border-teal-500/20 text-teal-400' : 'bg-amber-500/10 border-amber-500/20 text-amber-500'}`}>
              <Wifi className={`w-4 h-4 ${smsEnabled ? 'animate-pulse' : ''}`} />
            </div>
          </div>
        </div>

        {/* 2. Search & Filter Outbox */}
        <div className="bg-[#0c1a24]/20 backdrop-blur-xl border border-white/5 rounded-2xl p-4 flex gap-4 items-center justify-between flex-shrink-0 shadow-lg">
          <div className="relative w-full">
            <Search className="w-4 h-4 text-zinc-500 absolute left-3.5 top-1/2 -translate-y-1/2" />
            <input 
              type="text" 
              placeholder="Search transmitted alerts, mobile nodes, payload logs..." 
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full bg-[#040c12]/60 border border-white/5 rounded-xl pl-9 pr-4 py-2 text-xs text-white placeholder-zinc-500 focus:outline-none focus:border-teal-500/50 font-semibold"
            />
          </div>
        </div>

        {/* 3. SMS Logs List */}
        <div className="flex-1 overflow-y-auto space-y-3 pr-1.5 min-h-0">
          {dispatchLogs.length === 0 ? (
            <div className="bg-[#0c1a24]/10 rounded-3xl border border-white/5 p-20 text-center text-zinc-500 font-bold uppercase tracking-widest text-xs">
              No SMS alerts registered in active memory logs.
            </div>
          ) : (
            dispatchLogs.map((log) => {
              const selected = selectedSMS?.id === log.id;
              const isCritical = log.severity === 'CRITICAL';
              const cleanType = log.module.replace(' Detection', '').toUpperCase();

              return (
                <div 
                  key={log.id}
                  onClick={() => setSelectedSMS(log)}
                  className={`bg-[#0c1a24]/20 border rounded-2xl p-4 cursor-pointer hover:border-teal-500/30 transition duration-300 flex justify-between items-center gap-6 group shadow-lg ${selected ? 'border-teal-400/40 bg-teal-950/5 ring-1 ring-teal-400/10' : 'border-white/5'}`}
                >
                  <div className="flex items-center gap-4 min-w-0">
                    {/* Log status icon badge */}
                    <div className={`p-3 rounded-xl border flex-shrink-0 transition-transform duration-300 group-hover:scale-105 ${selected ? 'bg-teal-500/20 border-teal-500/30 text-teal-400' : 'bg-zinc-950/40 border-white/5 text-zinc-500'}`}>
                      <MessageSquare className="w-5 h-5" strokeWidth={1.5} />
                    </div>

                    <div className="space-y-1 min-w-0">
                      <div className="flex items-center gap-2.5 flex-wrap">
                        <span className={`text-[8.5px] font-extrabold px-1.5 py-0.5 rounded leading-none ${isCritical ? 'bg-rose-500/10 text-rose-400 border border-rose-500/20 animate-pulse' : 'bg-amber-500/10 text-amber-400 border border-amber-500/20'}`}>
                          {log.severity}
                        </span>
                        <span className="text-[10px] text-zinc-500 font-mono font-medium">{log.timestamp}</span>
                        <span className="text-[10px] text-teal-400 font-mono font-semibold">SMS Dispatch Active</span>
                      </div>
                      
                      <h4 className="text-xs font-bold text-white tracking-wide truncate group-hover:text-teal-300 transition">
                         RAKSHAK SMS ALERT: {cleanType} VIOLATION DETECTED
                      </h4>
                      
                      <p className="text-[11px] text-zinc-400 truncate max-w-[400px]">
                        {log.ai_description || log.message}
                      </p>
                    </div>
                  </div>

                  <div className="flex items-center gap-4 flex-shrink-0">
                    <div className="flex flex-col items-end gap-1 font-mono text-[9px] text-zinc-500 font-semibold">
                      <div className="flex items-center gap-1 text-zinc-400">
                        <Smartphone className="w-3.5 h-3.5 text-teal-500" />
                        <span>Twilio API</span>
                      </div>
                      <span>AES-256 routed</span>
                    </div>
                    <ChevronRight className="w-4 h-4 text-zinc-600 group-hover:text-white transition-transform group-hover:translate-x-0.5" />
                  </div>
                </div>
              );
            })
          )}
        </div>

      </div>

      {/* B. RIGHT AREA: High-Fidelity Mobile Smartphone Simulator View */}
      <div className="w-[38%] bg-[#0c1a24]/20 backdrop-blur-xl border border-white/5 rounded-3xl p-5 shadow-2xl flex flex-col justify-between overflow-y-auto h-full flex-shrink-0 animate-in slide-in-from-right-5 duration-300 select-none">
        {selectedSMS ? (
          <div className="flex flex-col gap-4 h-full justify-between items-center w-full">
            
            {/* Operator mobile terminal mockup heading */}
            <div className="w-full pb-2 border-b border-white/5 flex justify-between items-center">
              <div>
                <h3 className="text-xs font-extrabold text-zinc-400 uppercase tracking-widest">Simulated Operator Terminal</h3>
                <p className="text-[8px] font-bold text-teal-400 uppercase tracking-widest font-mono">Mobile Node Node_01</p>
              </div>
              <span className="text-[8.5px] font-extrabold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 px-2 py-0.5 rounded flex items-center gap-1">
                <CheckCircle className="w-3 h-3" />
                DELIVERED
              </span>
            </div>

            {/* Premium iPhone/Android Frame Simulation */}
            <div className="w-full max-w-[280px] aspect-[9/18] bg-zinc-950 rounded-[40px] border-4 border-zinc-800 relative shadow-2xl flex flex-col overflow-hidden shadow-teal-500/5 my-2">
              
              {/* Phone Speaker Notch */}
              <div className="absolute top-0 left-1/2 -translate-x-1/2 w-28 h-5 bg-zinc-800 rounded-b-2xl z-50 flex items-center justify-center">
                <div className="w-10 h-1 bg-black rounded-full" />
              </div>

              {/* Status Bar */}
              <div className="h-9 bg-zinc-900 border-b border-white/5 flex justify-between items-end px-6 pb-1 text-[9px] font-bold font-mono text-zinc-400 relative z-30 select-none">
                <span>09:41</span>
                <div className="flex items-center gap-1.5">
                  <Signal className="w-3 h-3 text-teal-400" />
                  <span className="text-[8px]">5G</span>
                  <Battery className="w-3.5 h-3.5" />
                </div>
              </div>

              {/* Messages Screen Content */}
              <div className="flex-1 bg-zinc-900 p-3 flex flex-col justify-between overflow-y-auto space-y-3 relative z-20 min-h-0 select-text">
                {/* Scrollable Conversation area */}
                <div className="space-y-4 flex-1 flex flex-col justify-end pb-2">
                  
                  {/* SMS Conversation Timestamp */}
                  <div className="text-[8px] text-zinc-600 font-bold uppercase tracking-wider text-center py-1">
                    Today at {selectedSMS.timestamp.split(' ')[1] || '09:41'}
                  </div>

                  {/* Message Bubble 1 (Incoming system update) */}
                  <div className="flex flex-col gap-1 items-start max-w-[85%] self-start animate-in slide-in-from-left duration-300">
                    <div className="text-[7.5px] font-extrabold text-zinc-500 uppercase tracking-widest ml-1 leading-none">
                      RAKSHAK CORE
                    </div>
                    <div className="bg-zinc-800 border border-white/5 text-zinc-300 rounded-2xl rounded-tl-sm px-3 py-2 text-[10px] leading-snug shadow-md">
                      Secure telemetry interface is online. Watching active threat polygons.
                    </div>
                  </div>

                  {/* Message Bubble 2 (Active alert SMS alert) */}
                  <div className="flex flex-col gap-1 items-end max-w-[88%] self-end animate-in slide-in-from-right duration-300">
                    <div className="text-[7.5px] font-extrabold text-teal-400 uppercase tracking-widest mr-1 leading-none">
                      DELIVERED
                    </div>
                    <div className="bg-teal-500 text-zinc-950 font-bold rounded-2xl rounded-tr-sm px-3.5 py-2.5 text-[9.5px] leading-relaxed shadow-lg font-mono whitespace-pre-wrap select-text break-words">
                      {getSMSBody(selectedSMS)}
                    </div>
                    <span className="text-[8px] text-zinc-500 font-semibold tracking-wider uppercase font-mono mr-1">
                      Delivered - Just now
                    </span>
                  </div>

                </div>

                {/* Simulated Input field at bottom of phone viewport */}
                <div className="bg-zinc-950 rounded-full border border-white/5 p-1 flex items-center justify-between flex-shrink-0">
                  <div className="text-[9px] text-zinc-600 px-3 py-1 font-bold uppercase font-sans tracking-wide">
                    Text Message
                  </div>
                  <button className="w-5 h-5 rounded-full bg-teal-500 text-zinc-950 flex items-center justify-center font-bold text-xs hover:scale-105 active:scale-95 transition">
                    ^
                  </button>
                </div>
              </div>
            </div>

            {/* Quick action buttons for testing */}
            <div className="w-full flex gap-3 mt-1.5">
              {/* Button to view visual snapshot proof */}
              <a 
                href={snapshotUrl(selectedSMS.snapshot_path)} 
                target="_blank" 
                rel="noreferrer" 
                className="flex-1 bg-[#0c1a24]/60 hover:bg-teal-500/10 border border-teal-500/20 text-teal-400 py-2 rounded-xl text-[10px] font-extrabold uppercase tracking-widest transition hover:scale-[1.02] active:scale-95 flex items-center justify-center gap-1.5 shadow"
              >
                <Eye className="w-3.5 h-3.5" />
                Snapshot Proof
              </a>
            </div>

            {/* Footer gateway info */}
            <div className="pt-2.5 border-t border-white/5 text-[8.5px] text-center text-zinc-600 font-bold uppercase select-none w-full tracking-wide">
              Twilio REST Dispatch Matrix V1.0 - TLS encrypted
            </div>

          </div>
        ) : (
          <div className="h-full flex flex-col items-center justify-center text-center text-zinc-600 gap-3.5 px-4 py-20 select-none">
            <div className="p-3 bg-zinc-950/40 rounded-full border border-white/5 animate-pulse">
              <Smartphone className="w-6 h-6 text-zinc-500" strokeWidth={1.5} />
            </div>
            <div className="space-y-1">
              <div className="text-xs font-bold uppercase tracking-wider text-zinc-500">Outbox SMS Matrix Idle</div>
              <p className="text-[10px] leading-relaxed text-zinc-500 font-semibold uppercase">
                Select an automated system SMS alert message from the outbox log to preview the delivered text payload on the operator's simulated mobile terminal.
              </p>
            </div>
          </div>
        )}
      </div>

    </div>
  );
}
