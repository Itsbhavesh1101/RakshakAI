import { 
  TrendingUp, AlertTriangle, Clock, Calendar, 
  ArrowUpRight, Download, Filter
} from 'lucide-react';
import { mockIncidentLogs, mockChartData } from '../utils/mockData';

export default function Analytics() {
  return (
    <div className="flex flex-col gap-6 h-full overflow-hidden animate-in fade-in duration-300">
      
      {/* 1. BENTO STATS ROW */}
      <div className="grid grid-cols-3 gap-6">
        {[
          { label: 'Total Anomalies (24h)', value: '24 Incidents', detail: '+12% from last week', icon: TrendingUp, color: 'text-rose-400', glow: 'shadow-rose-950/10' },
          { label: 'Most Active Threat Type', value: 'Perimeter Intrusion', detail: 'CAM_PERIMETER_03 dominates', icon: AlertTriangle, color: 'text-amber-400', glow: 'shadow-amber-950/10' },
          { label: 'Peak Activity Hours', value: '18:00 - 20:00', detail: 'Schedules correspond to shift switches', icon: Clock, color: 'text-teal-400', glow: 'shadow-teal-950/10' }
        ].map((stat, idx) => {
          const Icon = stat.icon;
          return (
            <div 
              key={idx}
              className={`bg-[#0c1a24]/30 backdrop-blur-xl border border-white/5 rounded-2xl p-4 flex justify-between items-center shadow-lg hover:border-teal-500/20 transition duration-300 ${stat.glow} group`}
            >
              <div className="space-y-1">
                <span className="text-[10px] text-zinc-500 uppercase tracking-widest font-extrabold">{stat.label}</span>
                <div className={`text-xl font-bold tracking-wide ${stat.color}`}>{stat.value}</div>
                <div className="text-[9px] text-zinc-400 font-semibold">{stat.detail}</div>
              </div>
              <div className="p-3 bg-zinc-950/40 rounded-xl border border-white/5 group-hover:scale-105 transition text-zinc-400">
                <Icon className="w-5 h-5" strokeWidth={1.5} />
              </div>
            </div>
          );
        })}
      </div>

      {/* 2. INCIDENT FREQUENCY OVER TIME (HIGH-FIDELITY CHART PLACEHOLDER) */}
      <div className="bg-[#0c1a24]/20 backdrop-blur-xl border border-white/5 rounded-3xl p-5 shadow-2xl flex flex-col gap-4">
        <div className="flex justify-between items-center">
          <div>
            <h3 className="text-sm font-bold text-white uppercase tracking-wider">Incident Frequency Over Time</h3>
            <p className="text-[9px] text-zinc-500 font-bold uppercase tracking-widest mt-0.5 font-mono">Aggregated AI threat signals across 2-hour windows</p>
          </div>
          <div className="flex gap-2.5">
            <button className="px-3 py-1.5 rounded-lg bg-zinc-900/60 border border-white/5 text-[9px] font-bold text-zinc-400 hover:text-white flex items-center gap-1.5 transition">
              <Filter className="w-3.5 h-3.5" /> Filter Modules
            </button>
            <button className="px-3 py-1.5 rounded-lg bg-teal-500 hover:bg-teal-400 text-[#040c12] text-[9px] font-extrabold uppercase flex items-center gap-1.5 transition shadow-lg shadow-teal-500/10">
              <Download className="w-3.5 h-3.5" /> Export Report
            </button>
          </div>
        </div>

        {/* High-Fidelity SVG Line Chart Grid */}
        <div className="h-[180px] w-full bg-zinc-950/40 rounded-2xl border border-white/5 p-4 flex flex-col justify-between relative overflow-hidden group">
          <div className="absolute inset-0 bg-gradient-to-t from-teal-500/5 to-transparent opacity-0 group-hover:opacity-100 transition duration-500 pointer-events-none" />

          {/* SVG Canvas drawing bars and paths */}
          <div className="absolute inset-x-12 top-6 bottom-8 z-0">
            <svg className="w-full h-full" preserveAspectRatio="none">
              {/* Grid guide lines */}
              <line x1="0%" y1="0%" x2="100%" y2="0%" stroke="rgba(255,255,255,0.03)" strokeWidth="1" strokeDasharray="4" />
              <line x1="0%" y1="33%" x2="100%" y2="33%" stroke="rgba(255,255,255,0.03)" strokeWidth="1" strokeDasharray="4" />
              <line x1="0%" y1="66%" x2="100%" y2="66%" stroke="rgba(255,255,255,0.03)" strokeWidth="1" strokeDasharray="4" />
              <line x1="0%" y1="100%" x2="100%" y2="100%" stroke="rgba(255,255,255,0.05)" strokeWidth="1" />

              {/* Chart Line Path: Trespassing data points (height is inverted, max value = 6) */}
              {/* Points: 08:00 (2), 10:00 (4), 12:00 (1), 14:00 (3), 16:00 (0), 18:00 (5), 20:00 (2) */}
              <path 
                d="M0,66.6 L16.6,33.3 L33.3,83.3 L50,50 L66.6,100 L83.3,16.6 L100,66.6" 
                className="fill-none stroke-teal-400 stroke-[2] transition duration-500" 
                vectorEffect="non-scaling-stroke"
              />
              {/* Linear gradient shadow beneath path */}
              <path 
                d="M0,66.6 L16.6,33.3 L33.3,83.3 L50,50 L66.6,100 L83.3,16.6 L100,66.6 L100,100 L0,100 Z" 
                className="fill-teal-500/5 stroke-none transition duration-500" 
                vectorEffect="non-scaling-stroke"
              />

              {/* Data points dots */}
              <circle cx="0%" cy="66.6%" r="4" className="fill-teal-400 stroke-zinc-950 stroke-[2]" />
              <circle cx="16.6%" cy="33.3%" r="4" className="fill-teal-400 stroke-zinc-950 stroke-[2]" />
              <circle cx="33.3%" cy="83.3%" r="4" className="fill-teal-400 stroke-zinc-950 stroke-[2]" />
              <circle cx="50%" cy="50%" r="4" className="fill-teal-400 stroke-zinc-950 stroke-[2]" />
              <circle cx="66.6%" cy="100%" r="4" className="fill-teal-400 stroke-zinc-950 stroke-[2]" />
              <circle cx="83.3%" cy="16.6%" r="4" className="fill-teal-400 stroke-zinc-950 stroke-[2]" />
              <circle cx="100%" cy="66.6%" r="4" className="fill-teal-400 stroke-zinc-950 stroke-[2]" />
            </svg>
          </div>

          {/* Left Y Axis labels */}
          <div className="flex flex-col justify-between text-[8px] font-mono text-zinc-500 font-bold h-[142px] z-10 select-none">
            <span>6 max</span>
            <span>4 mid</span>
            <span>2 min</span>
            <span>0</span>
          </div>

          {/* Bottom X Axis time intervals */}
          <div className="flex justify-between items-center pl-12 pr-4 text-[8px] font-mono text-zinc-500 font-extrabold uppercase select-none z-10">
            {mockChartData.map((d, i) => (
              <span key={i}>{d.time}</span>
            ))}
          </div>

        </div>
      </div>

      {/* 3. POST-INCIDENT EVENT LOG DATA TABLE */}
      <div className="flex-1 flex flex-col gap-3 overflow-hidden">
        <div className="text-[10px] text-zinc-400 uppercase font-extrabold tracking-widest flex justify-between items-center px-1">
          <span>Historical Threat Incident Log</span>
          <div className="flex items-center gap-1.5 text-zinc-500 text-[9px] font-bold font-mono uppercase">
            <Calendar className="w-3.5 h-3.5" /> Showing past 24 hours
          </div>
        </div>

        <div className="flex-1 bg-[#0c1a24]/20 backdrop-blur-xl border border-white/5 rounded-2xl overflow-hidden shadow-2xl flex flex-col">
          <div className="flex-1 overflow-y-auto">
            <table className="w-full text-left border-collapse text-xs select-none">
              <thead>
                <tr className="border-b border-white/5 bg-zinc-950/40 text-[9px] font-extrabold uppercase tracking-wider text-zinc-500">
                  <th className="py-3 px-4 font-mono">Date / Time</th>
                  <th className="py-3 px-4">Camera ID</th>
                  <th className="py-3 px-4">Classification</th>
                  <th className="py-3 px-4 text-center">Confidence</th>
                  <th className="py-3 px-4">Action Taken</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-white/5 font-medium text-zinc-300">
                {mockIncidentLogs.map((log) => (
                  <tr key={log.id} className="hover:bg-white/[0.02] active:bg-white/[0.04] transition duration-200 cursor-pointer">
                    <td className="py-3 px-4 text-zinc-400 font-mono">{log.timestamp}</td>
                    <td className="py-3 px-4 text-zinc-400 font-mono text-[10px] font-semibold">{log.cameraId}</td>
                    <td className="py-3 px-4">
                      <div className="flex items-center gap-2">
                        <span className={`w-1.5 h-1.5 rounded-full ${log.severity === 'CRITICAL' ? 'bg-rose-500 animate-pulse' : log.severity === 'WARNING' ? 'bg-amber-500' : 'bg-teal-500'}`} />
                        <span>{log.threatClassification}</span>
                      </div>
                    </td>
                    <td className="py-3 px-4 text-center font-mono font-bold text-teal-400">{log.confidence}%</td>
                    <td className="py-3 px-4 text-[10px] text-zinc-400 italic font-sans flex items-center justify-between">
                      <span>{log.actionTaken}</span>
                      <ArrowUpRight className="w-3.5 h-3.5 text-zinc-600 hover:text-white transition" />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

    </div>
  );
}
