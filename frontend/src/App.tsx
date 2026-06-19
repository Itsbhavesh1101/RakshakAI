import { useState, useEffect, useRef } from 'react';
import type { MouseEvent } from 'react';
import { 
  Users, ChevronLeft, ChevronRight, Bell, AlertOctagon, 
  Flame, Crosshair, RefreshCw, Monitor, History, Send,
  Search, MoreVertical, Disc, Wifi, Sparkles,
  Activity, Eye, EyeOff, ShieldAlert, Cpu, Mail,
  Smartphone, UserCheck, Plus, Trash2
} from 'lucide-react';

import Analytics from './views/Analytics';
import DetectionHistory from './views/DetectionHistory';
import AlertsSent from './views/AlertsSent';

interface AlertLog {
  id: number;
  timestamp: string;
  module: string;
  severity: string;
  message: string;
  snapshot_path: string;
  ai_description: string | null;
}

interface TelemetryAlert {
  camera_id: string;
  timestamp: string;
  anomaly_detected: boolean;
  type: string;
  severity: string;
  confidence: number;
  bounding_boxes: number[][];
  live_count: number | null;
}

interface SystemStats {
  crowd_count: number;
  inference_ms?: number;
  inference_fps?: number;
}

interface SystemConfig {
  active_modules: string[];
  confidence_thresholds: Record<string, number>;
  cooldown_periods: Record<string, number>;
  restricted_zone_coords: [number, number][];
  authorized_people: AuthorizedPerson[];
  source_type: string;
  video_filepath: string;
  smtp_enabled: boolean;
  smtp_server: string;
  to_address: string;
  sms_enabled: boolean;
  to_phone: string;
  camera_device_id: number;
  target_fps: number;
  inference_fps: number;
  jpeg_quality: number;
  model_imgsz: number;
}

interface AuthorizedPerson {
  id: string;
  name: string;
  role: string;
  access_zones: string[];
  enabled: boolean;
  present: boolean;
  intrusion_bypass: boolean;
}

type ViewName = 'dashboard' | 'analytics' | 'history' | 'alerts_sent';
type SubTab = 'control' | 'analytics' | 'intrusion' | 'authorized' | 'ptz' | 'filters' | 'notifications' | 'logs';
type DisplayMode = 'standard' | 'night' | 'heatmap' | 'tracking';

const API_BASE_URL = (() => {
  const configured = import.meta.env.VITE_API_BASE_URL as string | undefined;
  if (configured) return configured.replace(/\/$/, '');

  if (window.location.port === '5173') {
    return `${window.location.protocol}//${window.location.hostname}:8000`;
  }

  return window.location.origin;
})();

const WS_BASE_URL = API_BASE_URL.replace(/^http/, 'ws');
const apiUrl = (path: string) => `${API_BASE_URL}${path}`;
const snapshotUrl = (path: string) => `${API_BASE_URL}/${path.replace(/^\/+/, '')}`;

const SUB_TABS: Array<{ id: SubTab; label: string }> = [
  { id: 'control', label: 'Model Control' },
  { id: 'analytics', label: 'Analytics Config' },
  { id: 'intrusion', label: 'Intrusion Zones' },
  { id: 'authorized', label: 'Authorized Persons' },
  { id: 'notifications', label: 'Notification Routing' },
  { id: 'ptz', label: 'PTZ Stream Matrix' },
  { id: 'filters', label: 'Active Detection Filters' },
  { id: 'logs', label: 'Recent Anomaly Logs' },
];

const DISPLAY_MODES: Array<{ id: DisplayMode; label: string }> = [
  { id: 'standard', label: 'Standard View' },
  { id: 'night', label: 'Night Vision' },
  { id: 'heatmap', label: 'Heat Map Overlay' },
  { id: 'tracking', label: 'AI Frame Tracking' },
];

const DETECTION_FILTERS = [
  { id: 'weapon_detection', label: 'Weapon', icon: Crosshair },
  { id: 'fire_detection', label: 'Fire', icon: Flame },
  { id: 'fall_detection', label: 'Fall', icon: AlertOctagon },
  { id: 'crowd_detection', label: 'Human', icon: Users },
  { id: 'crowd_detection', label: 'Crowd', icon: ShieldAlert },
  { id: 'fire_detection', label: 'Smoke', icon: Disc },
  { id: 'trespassing_detection', label: 'Intrusion', icon: ShieldAlert },
];

const createAuthorizedPerson = (): AuthorizedPerson => ({
  id: `person-${Date.now()}`,
  name: '',
  role: 'Security Staff',
  access_zones: ['CAM_MAIN_ENTRANCE_01'],
  enabled: true,
  present: false,
  intrusion_bypass: false,
});

export default function App() {
  // Main layout views routing
  const [activeView, setActiveView] = useState<ViewName>('dashboard');

  // Header Dropdown States
  const [isSearchOpen, setIsSearchOpen] = useState(false);
  const [isNotificationsOpen, setIsNotificationsOpen] = useState(false);
  const [isMenuOpen, setIsMenuOpen] = useState(false);

  // Global Toast Notifications
  const [toast, setToast] = useState<string | null>(null);

  const triggerToast = (message: string) => {
    setToast(message);
    setTimeout(() => setToast(null), 3000);
  };

  const toggleSearch = () => {
    setIsSearchOpen(!isSearchOpen);
    setIsNotificationsOpen(false);
    setIsMenuOpen(false);
  };

  const toggleNotifications = () => {
    setIsNotificationsOpen(!isNotificationsOpen);
    setIsSearchOpen(false);
    setIsMenuOpen(false);
  };

  const toggleMenu = () => {
    setIsMenuOpen(!isMenuOpen);
    setIsSearchOpen(false);
    setIsNotificationsOpen(false);
  };

  // Sub-tabs (ML control parameters spec)
  const [activeSubTab, setActiveSubTab] = useState<SubTab>('control');
  
  // Connection and Live Data
  const [connected, setConnected] = useState(false);
  const [frame, setFrame] = useState<string | null>(null);
  const [liveAlerts, setLiveAlerts] = useState<TelemetryAlert[]>([]);
  const [stats, setStats] = useState<SystemStats>({ crowd_count: 0, inference_ms: 0, inference_fps: 2 });
  const [criticalActive, setCriticalActive] = useState(false);
  
  // Config States
  const [activeModules, setActiveModules] = useState<string[]>([]);
  const [confidenceThresholds, setConfidenceThresholds] = useState<Record<string, number>>({});
  const [restrictedZoneCoords, setRestrictedZoneCoords] = useState<[number, number][]>([]);
  const [authorizedPeople, setAuthorizedPeople] = useState<AuthorizedPerson[]>([]);
  const [smtpEnabled, setSmtpEnabled] = useState(false);
  const [toAddress, setToAddress] = useState('');
  const [smsEnabled, setSmsEnabled] = useState(false);
  const [toPhone, setToPhone] = useState('');
  const [sourceType, setSourceType] = useState('webcam');
  const [videoFilepath, setVideoFilepath] = useState('');
  const [cameraDeviceId, setCameraDeviceId] = useState(0);
  const [targetFps, setTargetFps] = useState(20);
  const [inferenceFps, setInferenceFps] = useState(2);

  // Persistent Alert Logs History
  const [alertLogs, setAlertLogs] = useState<AlertLog[]>([]);
  
  // Bounding Box overlays toggle state
  const [analyticsOverlaysEnabled, setAnalyticsOverlaysEnabled] = useState(true);

  // Replaying specific event snapshot
  const [replaySnapshot, setReplaySnapshot] = useState<string | null>(null);

  // Active Camera selection (feeds switching)
  const [activeCamIndex, setActiveCamIndex] = useState(0);
  const cameras = [
    { id: 'CAM_MAIN_ENTRANCE_01', name: 'Cam 1 - Entrance', lastCheck: '1s ago', operator: 'Agent Hawkins', status: 'Live' },
    { id: 'CAM_BACK_LOBBY_02', name: 'Cam 2 - Back Lobby', lastCheck: '4s ago', operator: 'Agent Hawkins', status: 'Live' },
    { id: 'CAM_PERIMETER_03', name: 'Cam 3 - Perimeter', lastCheck: '12s ago', operator: 'Agent Hawkins', status: 'Live' }
  ];

  // PTZ Controls
  const [pan, setPan] = useState(120);
  const [tilt, setTilt] = useState(120);
  const [zoom, setZoom] = useState(100);

  // Feed Display Mode
  const [displayMode, setDisplayMode] = useState<DisplayMode>('standard');

  const socketRef = useRef<WebSocket | null>(null);
  const videoCardRef = useRef<HTMLDivElement | null>(null);

  // Fetch Configurations on Startup
  const fetchConfig = async () => {
    try {
      const res = await fetch(apiUrl('/api/config'));
      if (!res.ok) throw new Error(`Config request failed with ${res.status}`);
      const data: SystemConfig = await res.json();
      setActiveModules(data.active_modules || []);
      setConfidenceThresholds(data.confidence_thresholds || {});
      setRestrictedZoneCoords(data.restricted_zone_coords || []);
      setAuthorizedPeople(data.authorized_people || []);
      setSmtpEnabled(!!data.smtp_enabled);
      setToAddress(data.to_address || '');
      setSmsEnabled(!!data.sms_enabled);
      setToPhone(data.to_phone || '');
      setSourceType(data.source_type || 'webcam');
      setVideoFilepath(data.video_filepath || '');
      setCameraDeviceId(data.camera_device_id || 0);
      setTargetFps(data.target_fps || 20);
      setInferenceFps(data.inference_fps || 2.0);
    } catch (err) {
      console.error('Error fetching configurations:', err);
    }
  };

  // Fetch Alert Logs History
  const fetchAlertLogs = async () => {
    try {
      const res = await fetch(apiUrl('/api/alerts?limit=100'));
      if (!res.ok) throw new Error(`Alerts request failed with ${res.status}`);
      const data: AlertLog[] = await res.json();
      setAlertLogs(data);
    } catch (err) {
      console.error('Error fetching alerts history:', err);
    }
  };

  // Save Settings to Backend
  const saveConfig = async (updatedFields: Partial<SystemConfig>) => {
    try {
      const res = await fetch(apiUrl('/api/config'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(updatedFields),
      });
      if (!res.ok) throw new Error(`Config save failed with ${res.status}`);
      void fetchConfig();
    } catch (err) {
      console.error('Error saving configs:', err);
    }
  };

  // Clear Alerts History
  const clearAlertLogs = async () => {
    if (window.confirm('Are you sure you want to permanently clear all logs and visual snapshots?')) {
      try {
        const res = await fetch(apiUrl('/api/alerts/clear'), { method: 'POST' });
        if (!res.ok) throw new Error(`Clear request failed with ${res.status}`);
        void fetchAlertLogs();
      } catch (err) {
        console.error('Error clearing alert logs:', err);
      }
    }
  };

  // Connect WebSockets
  useEffect(() => {
    let closedByComponent = false;
    let reconnectTimer: number | undefined;

    queueMicrotask(() => {
      void fetchConfig();
      void fetchAlertLogs();
    });
    
    const logsInterval = window.setInterval(() => {
      void fetchAlertLogs();
    }, 5000);

    const connectWS = () => {
      if (closedByComponent) return;

      const wsUrl = `${WS_BASE_URL}/api/ws/telemetry`;
      const ws = new WebSocket(wsUrl);
      socketRef.current = ws;

      ws.onopen = () => {
        setConnected(true);
        console.log('WebSocket connection successfully opened!');
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.frame) {
            setFrame(data.frame);
          }
          setLiveAlerts(data.alerts || []);
          setStats(data.stats || { crowd_count: 0, inference_ms: 0, inference_fps: 2 });
          
          const critical = (data.alerts || []).some((a: TelemetryAlert) => a.severity === 'CRITICAL');
          setCriticalActive(critical);
        } catch (err) {
          console.error('Error parsing WS message:', err);
        }
      };

      ws.onclose = () => {
        setConnected(false);
        if (!closedByComponent) {
          console.log('WebSocket disconnected. Reconnecting in 3 seconds...');
          reconnectTimer = window.setTimeout(connectWS, 3000);
        }
      };

      ws.onerror = (err) => {
        console.error('WebSocket Error:', err);
        ws.close();
      };
    };

    connectWS();

    return () => {
      closedByComponent = true;
      window.clearInterval(logsInterval);
      if (reconnectTimer) window.clearTimeout(reconnectTimer);
      if (socketRef.current) socketRef.current.close();
      socketRef.current = null;
    };
  }, []);

  const toggleModule = (module: string) => {
    const updated = activeModules.includes(module)
      ? activeModules.filter(m => m !== module)
      : [...activeModules, module];
    setActiveModules(updated);
    saveConfig({ active_modules: updated });
  };

  const handleConfChange = (module: string, val: number) => {
    const updatedConf = { ...confidenceThresholds, [module]: val };
    setConfidenceThresholds(updatedConf);
    saveConfig({ confidence_thresholds: updatedConf });
  };

  const handleCoordChange = (index: number, axis: 'x' | 'y', val: number) => {
    const nextCoords = [...restrictedZoneCoords];
    const original = nextCoords[index];
    if (axis === 'x') {
      nextCoords[index] = [val, original[1]];
    } else {
      nextCoords[index] = [original[0], val];
    }
    setRestrictedZoneCoords(nextCoords);
  };

  const saveAuthorizedPeople = (people: AuthorizedPerson[]) => {
    const normalized = people.map(person => ({
      ...person,
      name: person.name.trim() || 'Authorized Person',
      role: person.role.trim() || 'Authorized',
      access_zones: person.access_zones.length ? person.access_zones : ['CAM_MAIN_ENTRANCE_01'],
    }));
    setAuthorizedPeople(normalized);
    saveConfig({ authorized_people: normalized });
  };

  const updateAuthorizedPerson = (id: string, updates: Partial<AuthorizedPerson>) => {
    setAuthorizedPeople(prev => prev.map(person => (
      person.id === id ? { ...person, ...updates } : person
    )));
  };

  const removeAuthorizedPerson = (id: string) => {
    const nextPeople = authorizedPeople.filter(person => person.id !== id);
    saveAuthorizedPeople(nextPeople);
    triggerToast("AUTHORIZED ROSTER UPDATED");
  };

  // Handle feed switches (left/right arrows)
  const prevCamera = () => {
    setActiveCamIndex(prev => (prev === 0 ? cameras.length - 1 : prev - 1));
  };

  const nextCamera = () => {
    setActiveCamIndex(prev => (prev === cameras.length - 1 ? 0 : prev + 1));
  };

  const currentCam = cameras[activeCamIndex];

  // Dispatch Alarm critical action (SMS/Email triggers)
  const dispatchPatrol = async () => {
    try {
      triggerToast("DISPATCH ALARM INITIATED");
      // Call backend to enrich or flag emergency
      await fetch(apiUrl('/api/config'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ smtp_enabled: true }),
      });
    } catch (e) {
      console.error(e);
    }
  };

  // Canvas Click Coordinate Handler (Draw Polygon dynamically for Restricted Intrusion Zones)
  const handleCanvasClick = (e: MouseEvent<HTMLDivElement>) => {
    if (activeSubTab !== 'intrusion' || !videoCardRef.current) return;
    
    const rect = videoCardRef.current.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const clickY = e.clientY - rect.top;
    
    // Scale local click relative coordinates (matching 640x480 resolution frame)
    const scaledX = Math.round((clickX / rect.width) * 640);
    const scaledY = Math.round((clickY / rect.height) * 480);
    
    // We maintain a 4-point polygon. Let's cycle and replace the nodes sequentially
    const nextCoords = [...restrictedZoneCoords];
    if (nextCoords.length < 4) {
      nextCoords.push([scaledX, scaledY]);
    } else {
      // Cycle replacing coordinates
      nextCoords.shift();
      nextCoords.push([scaledX, scaledY]);
    }
    
    setRestrictedZoneCoords(nextCoords);
  };

  // PTZ camera motorized gimbal translation & dynamic display mode filters
  const getStreamStyles = () => {
    let filterStr = '';
    if (displayMode === 'night') {
      filterStr = 'sepia(1) hue-rotate(80deg) saturate(3) brightness(1.2) contrast(1.2)';
    } else if (displayMode === 'heatmap') {
      filterStr = 'saturate(3.5) hue-rotate(240deg) contrast(1.5) brightness(0.9)';
    } else if (displayMode === 'tracking') {
      filterStr = 'grayscale(1) invert(0.95) contrast(3.5) brightness(0.9)';
    }

    const panOffset = (pan - 120) * 0.7;
    const tiltOffset = (tilt - 120) * 0.7;

    return {
      filter: filterStr || undefined,
      transform: `scale(${zoom / 100}) translate(${panOffset}px, ${tiltOffset}px)`,
      transition: 'transform 0.15s cubic-bezier(0.25, 0.46, 0.45, 0.94)',
    };
  };

  return (
    <div className="h-screen w-screen bg-[#040c12] text-[#f4f4f5] flex overflow-hidden p-4 md:p-6 select-none font-sans relative">
      
      {/* BACKGROUND FLOATING TEAL NEON GLOW (Inspired by image_1.png) */}
      <div className="absolute top-0 right-0 w-[500px] h-[500px] bg-teal-500/10 rounded-full blur-[160px] pointer-events-none z-0" />
      <div className="absolute bottom-0 left-0 w-[400px] h-[400px] bg-cyan-500/5 rounded-full blur-[130px] pointer-events-none z-0" />

      {/* 1. NARROW LEFT SIDEBAR (Adapted from banking UI to Surveillance layout) */}
      <aside className="w-[84px] bg-[#0c1a24]/30 backdrop-blur-xl border border-white/5 rounded-3xl shadow-2xl flex flex-col items-center py-6 justify-start gap-12 h-full z-10 mr-6">
        
        {/* Top interlocking circle rings logo */}
        <div className="relative w-8 h-8 flex items-center justify-center">
          <div className="w-5 h-5 rounded-full border border-white absolute left-1" />
          <div className="w-5 h-5 rounded-full border border-white absolute right-1" />
        </div>

        {/* Vertical Icon Menu */}
        <div className="flex flex-col gap-8 items-center w-full">
          <button 
            onClick={() => setActiveView('dashboard')}
            className="flex flex-col items-center gap-1.5 group outline-none relative"
          >
            {activeView !== 'dashboard' && <div className="w-1.5 h-1.5 rounded-full bg-teal-400 absolute -top-1 right-3 animate-ping" />}
            <Monitor className={`w-5 h-5 transition ${activeView === 'dashboard' ? 'text-teal-400' : 'text-zinc-400 group-hover:text-white'}`} strokeWidth={1.5} />
            <span className={`text-[9px] tracking-wider transition ${activeView === 'dashboard' ? 'text-teal-400' : 'text-zinc-500 group-hover:text-zinc-200'}`}>Dashboard</span>
            {activeView === 'dashboard' && <div className="w-1 h-4 bg-teal-400 rounded-r absolute -left-7 top-1" />}
          </button>
          
          <button 
            onClick={() => setActiveView('analytics')}
            className="flex flex-col items-center gap-1.5 group outline-none relative"
          >
            <Activity className={`w-5 h-5 transition ${activeView === 'analytics' ? 'text-teal-400' : 'text-zinc-400 group-hover:text-white'}`} strokeWidth={1.5} />
            <span className={`text-[9px] tracking-wider transition ${activeView === 'analytics' ? 'text-teal-400 font-semibold' : 'text-zinc-500 group-hover:text-zinc-200'}`}>Analytics</span>
            {activeView === 'analytics' && <div className="w-1 h-4 bg-teal-400 rounded-r absolute -left-7 top-1" />}
          </button>

          <button 
            onClick={() => setActiveView('history')}
            className="flex flex-col items-center gap-1.5 group outline-none relative"
          >
            <History className={`w-5 h-5 transition ${activeView === 'history' ? 'text-teal-400' : 'text-zinc-400 group-hover:text-white'}`} strokeWidth={1.5} />
            <span className={`text-[9px] tracking-wide text-center transition ${activeView === 'history' ? 'text-teal-400 font-semibold' : 'text-zinc-500 group-hover:text-zinc-200'}`}>Detection History</span>
            {activeView === 'history' && <div className="w-1 h-4 bg-teal-400 rounded-r absolute -left-7 top-1" />}
          </button>

          <button 
            onClick={() => setActiveView('alerts_sent')}
            className="flex flex-col items-center gap-1.5 group outline-none relative"
          >
            <Send className={`w-5 h-5 transition ${activeView === 'alerts_sent' ? 'text-teal-400' : 'text-zinc-400 group-hover:text-white'}`} strokeWidth={1.5} />
            <span className={`text-[9px] tracking-wider transition ${activeView === 'alerts_sent' ? 'text-teal-400 font-semibold' : 'text-zinc-500 group-hover:text-zinc-200'}`}>Alert Sents</span>
            {activeView === 'alerts_sent' && <div className="w-1 h-4 bg-teal-400 rounded-r absolute -left-7 top-1" />}
          </button>
        </div>

      </aside>

      {/* MAIN CONTAINER PANEL */}
      <div className="flex-1 flex flex-col gap-6 z-10 overflow-hidden">
        
        {/* HEADER SECTION (Adapted from image_1.png) */}
        <header className="flex justify-between items-center pb-2">
          <div>
            <h1 className="text-2xl font-bold text-white tracking-wide uppercase">Rakshak</h1>
            <p className="text-[10px] text-zinc-500 tracking-widest font-extrabold uppercase mt-0.5">Autonomous Threat Intelligence Matrix</p>
          </div>

          <div className="flex items-center gap-4">
            {/* Try premium -> Dispatch Alarm critical action */}
            <button 
              onClick={dispatchPatrol}
              className="px-4 py-2 rounded-full text-xs font-bold bg-[#0c1a24]/60 border border-teal-500/30 text-teal-400 hover:bg-teal-500/10 active:scale-95 transition-all flex items-center gap-2 shadow-lg shadow-teal-500/5 hover:scale-105"
            >
              <Sparkles className="w-3.5 h-3.5" strokeWidth={1.5} />
              Dispatch Patrol Unit
            </button>

            <div className="relative">
              <button 
                onClick={toggleSearch}
                className={`p-2.5 rounded-full border transition hover:scale-105 active:scale-95 ${isSearchOpen ? 'bg-teal-500/20 border-teal-500/40 text-teal-400 animate-pulse' : 'bg-[#0c1a24]/60 border-white/5 text-zinc-400 hover:text-white'}`}
              >
                <Search className="w-4 h-4" strokeWidth={1.5} />
              </button>
              {isSearchOpen && (
                <div className="absolute right-0 top-12 z-50 bg-[#0c1a24]/90 border border-white/10 rounded-2xl p-3 shadow-2xl w-64 backdrop-blur-xl animate-in slide-in-from-top-2 duration-200">
                  <input 
                    type="text" 
                    placeholder="Search cameras, alerts, logs..." 
                    className="w-full bg-[#040c12] border border-white/10 rounded-xl px-3 py-2 text-xs text-white placeholder-zinc-500 focus:outline-none focus:border-teal-500/50 font-semibold"
                    autoFocus
                  />
                </div>
              )}
            </div>

            <div className="relative">
              <button 
                onClick={toggleNotifications}
                className={`p-2.5 rounded-full border relative transition hover:scale-105 active:scale-95 ${isNotificationsOpen ? 'bg-teal-500/20 border-teal-500/40 text-teal-400' : 'bg-[#0c1a24]/60 border-white/5 text-zinc-400 hover:text-white'}`}
              >
                {activeView === 'dashboard' && <span className="w-2 h-2 rounded-full bg-rose-500 absolute top-0.5 right-0.5 animate-ping" />}
                <Bell className="w-4 h-4" strokeWidth={1.5} />
              </button>
              {isNotificationsOpen && (
                <div className="absolute right-0 top-12 z-50 bg-[#0c1a24]/90 border border-white/10 rounded-2xl p-4 shadow-2xl w-60 backdrop-blur-xl animate-in slide-in-from-top-2 duration-200 text-xs font-semibold text-zinc-400">
                  <div className="pb-2 border-b border-white/5 text-white font-bold uppercase tracking-wider text-[9px]">Active Notifications</div>
                  <div className="pt-2 flex flex-col gap-2">
                    <div className="flex items-center gap-2 text-zinc-500">
                      <span className="w-1.5 h-1.5 rounded-full bg-zinc-500 animate-pulse" />
                      <span>No new critical alerts</span>
                    </div>
                  </div>
                </div>
              )}
            </div>

            <div className="relative">
              <button 
                onClick={toggleMenu}
                className={`p-2.5 rounded-full border transition hover:scale-105 active:scale-95 ${isMenuOpen ? 'bg-teal-500/20 border-teal-500/40 text-teal-400' : 'bg-[#0c1a24]/60 border-white/5 text-zinc-400 hover:text-white'}`}
              >
                <MoreVertical className="w-4 h-4" strokeWidth={1.5} />
              </button>
              {isMenuOpen && (
                <div className="absolute right-0 top-12 z-50 bg-[#0c1a24]/90 border border-white/10 rounded-2xl p-2 shadow-2xl w-48 backdrop-blur-xl animate-in slide-in-from-top-2 duration-200 flex flex-col text-xs font-semibold">
                  <button className="w-full text-left px-3 py-2.5 hover:bg-white/5 rounded-xl text-zinc-300 hover:text-white transition">System Diagnostics</button>
                  <button className="w-full text-left px-3 py-2.5 hover:bg-white/5 rounded-xl text-zinc-300 hover:text-white transition">Export Logs</button>
                  <button className="w-full text-left px-3 py-2.5 hover:bg-white/5 rounded-xl text-zinc-300 hover:text-white transition">Help & Docs</button>
                </div>
              )}
            </div>
          </div>
        </header>

        {/* SUB-TABS LAYER (Renamed to reflect actual ML control parameters) */}
        {activeView === 'dashboard' && (
          <div className="flex border-b border-white/5 text-xs text-zinc-400 font-bold uppercase tracking-wider gap-8 pb-0.5">
            {SUB_TABS.map(tab => (
              <button 
                key={tab.id}
                onClick={() => setActiveSubTab(tab.id)}
                className={`pb-3 border-b-2 transition-all outline-none ${activeSubTab === tab.id ? 'text-teal-400 border-teal-400' : 'border-transparent hover:text-zinc-200'}`}
              >
                {tab.label}
              </button>
            ))}
          </div>
        )}

        {/* 2-COLUMN SURVEILLANCE STAGE */}
        {activeView === 'dashboard' ? (
        <div className="flex-1 flex gap-6 overflow-hidden">
          
          {/* A. LEFT COLUMN (The Primary Surveillance Stream Card & Bottom Feeds Grid) */}
          <div className="w-[48%] flex flex-col gap-6 h-full overflow-hidden">
            
            {/* Primary camera metadata tag */}
            <div className="text-[10px] text-zinc-400 uppercase font-extrabold tracking-widest flex justify-between items-center">
              <span>Primary viewport stream</span>
              <span className="font-mono text-teal-400 bg-teal-500/10 px-2 py-0.5 rounded border border-teal-500/20">
                {currentCam.id}
              </span>
            </div>

            {/* Central Vertical Primary Video Feed Card */}
            <div 
              ref={videoCardRef}
              onClick={handleCanvasClick}
              className={`flex-1 bg-[#0c1a24]/20 backdrop-blur-xl border rounded-3xl p-4 relative overflow-hidden flex flex-col group justify-between shadow-2xl transition-all duration-300 ${activeSubTab === 'intrusion' ? 'cursor-crosshair border-teal-500/40 ring-1 ring-teal-500/30' : 'border-white/5'}`}
            >
              
              {/* Event Pulsing Overlay (Soft warm glowing amber/red overlay for anomalies) */}
              <div className={`absolute inset-0 bg-gradient-to-t from-rose-500/10 to-amber-500/10 transition-opacity duration-500 pointer-events-none z-10 ${criticalActive || liveAlerts.length > 0 ? 'opacity-100 animate-pulse' : 'opacity-0'}`} />

              {/* Top Corners: OSD metadata overlays & System Ingestion Speed Metrics */}
              <div className="flex justify-between items-start z-20">
                {/* Top-Left: Camera origin capsule */}
                <div className="flex items-center gap-2 bg-black/45 backdrop-blur-md border border-white/5 px-3 py-1.5 rounded-full shadow-lg">
                  <Wifi className="w-3.5 h-3.5 text-teal-400 animate-pulse" />
                  <span className="text-[10px] font-bold text-white tracking-wide">{currentCam.name}</span>
                </div>
                
                {/* Top-Right: Ingest technical details (Inference overlay metric spec) */}
                <div className="flex items-center gap-2.5 bg-black/45 backdrop-blur-md border border-white/5 px-3.5 py-1.5 rounded-full text-[9px] font-bold text-zinc-300 font-mono">
                  <Cpu className="w-3.5 h-3.5 text-teal-400" />
                  <span className={connected ? 'text-teal-400' : 'text-rose-400'}>{connected ? 'LIVE' : 'OFFLINE'}</span>
                  <span className="text-zinc-600">|</span>
                  <span>{targetFps} FPS</span>
                  <span className="text-zinc-600">|</span>
                  <span>AI {Math.round(stats.inference_ms ?? 0)}ms/{inferenceFps} FPS</span>
                  <span className="text-zinc-600">|</span>
                  <span>People {stats.crowd_count}</span>
                  <span className="text-zinc-600">|</span>
                  <span
                    className="text-teal-400 max-w-[120px] truncate"
                    title={sourceType === 'webcam' ? `Camera ${cameraDeviceId}` : videoFilepath}
                  >
                    {sourceType === 'webcam' ? `CAM ${cameraDeviceId}` : videoFilepath.split(/[\\/]/).pop() || 'FILE'}
                  </span>
                </div>
              </div>

              {/* Streaming Content Ingestion Layer */}
              <div className="absolute inset-0 bg-[#040c12]/80 flex items-center justify-center z-0 overflow-hidden">
                {replaySnapshot ? (
                  <img 
                    src={replaySnapshot} 
                    alt="Replayed Event Snapshot" 
                    style={getStreamStyles()}
                    className="w-full h-full object-cover transition duration-500 group-hover:scale-[1.02]" 
                  />
                ) : activeCamIndex === 0 && frame ? (
                  <img 
                    src={frame} 
                    alt="Surveillance Feed" 
                    style={getStreamStyles()}
                    className={`w-full h-full object-cover transition duration-500 group-hover:scale-[1.02] ${!analyticsOverlaysEnabled ? 'contrast-125 saturate-100' : ''}`} 
                  />
                ) : (
                  <div className="flex flex-col items-center gap-2.5 text-zinc-600 font-bold uppercase tracking-wider text-xs">
                    <RefreshCw className="w-6 h-6 animate-spin text-zinc-500" strokeWidth={1.5} />
                    Loading ingestion stream...
                  </div>
                )}

                {/* Night Vision Scanlines Overlay */}
                {displayMode === 'night' && (
                  <div className="absolute inset-0 pointer-events-none z-10 bg-[linear-gradient(rgba(18,16,16,0)_50%,rgba(0,0,0,0.25)_50%),linear-gradient(90deg,rgba(255,0,0,0.06),rgba(0,255,0,0.02),rgba(0,0,255,0.06))] bg-[length:100%_4px,3px_100%] opacity-40 animate-pulse" />
                )}
              </div>

              {/* Interactive Perimeter custom drawing SVG overlay */}
              {activeSubTab === 'intrusion' && (
                <svg className="absolute inset-0 w-full h-full z-10 pointer-events-none">
                  {/* Polygon connecting drawn coordinates */}
                  {restrictedZoneCoords.length >= 2 && (
                    <polygon 
                      points={restrictedZoneCoords.map(p => `${(p[0] / 640) * 100}%, ${(p[1] / 480) * 100}%`).join(' ')}
                      className="fill-teal-500/10 stroke-teal-400 stroke-[2] stroke-dasharray-[4]"
                    />
                  )}
                  {/* Dots for vertices */}
                  {restrictedZoneCoords.map((point, idx) => (
                    <g key={idx}>
                      <circle 
                        cx={`${(point[0] / 640) * 100}%`} 
                        cy={`${(point[1] / 480) * 100}%`} 
                        r="5" 
                        className="fill-teal-400 stroke-zinc-950 stroke-[2] pointer-events-auto cursor-pointer"
                      />
                      <text 
                        x={`${(point[0] / 640) * 100}%`} 
                        y={`${(point[1] / 480) * 100 - 8}%`} 
                        className="fill-teal-300 text-[10px] font-mono font-bold text-center"
                      >
                        N{idx + 1}
                      </text>
                    </g>
                  ))}
                </svg>
              )}

              {/* Feed navigation arrows */}
              <div className="absolute inset-x-4 top-1/2 -translate-y-1/2 flex justify-between z-20 pointer-events-none">
                <button 
                  onClick={prevCamera}
                  className="w-9 h-9 rounded-full bg-black/55 backdrop-blur-md border border-white/5 flex items-center justify-center text-white hover:bg-black/80 hover:scale-105 active:scale-95 transition pointer-events-auto shadow-xl"
                >
                  <ChevronLeft className="w-4 h-4" />
                </button>
                <button 
                  onClick={nextCamera}
                  className="w-9 h-9 rounded-full bg-black/55 backdrop-blur-md border border-white/5 flex items-center justify-center text-white hover:bg-black/80 hover:scale-105 active:scale-95 transition pointer-events-auto shadow-xl"
                >
                  <ChevronRight className="w-4 h-4" />
                </button>
              </div>

              {/* Bottom OSD Pill details (Clean corners layout to avoid obscuring frame center) */}
              <div className="flex justify-between items-center z-20 mt-auto">
                <div className="flex items-center gap-1.5 bg-black/45 backdrop-blur-md border border-white/5 px-3 py-1.5 rounded-full text-[10px] font-bold text-zinc-300 font-mono">
                  <Disc className="w-3.5 h-3.5 text-rose-500 animate-pulse" />
                  Recording
                </div>

                {replaySnapshot ? (
                  <div className="bg-rose-500/20 backdrop-blur-md border border-rose-500/40 text-rose-300 px-4 py-1.5 text-[9px] font-extrabold uppercase tracking-wider rounded-full shadow-lg flex items-center gap-3">
                    <span>REPLAYING EVENT SNAPSHOT</span>
                    <button 
                      onClick={(e) => {
                        e.stopPropagation();
                        setReplaySnapshot(null);
                      }}
                      className="bg-rose-500 text-white font-bold px-2 py-0.5 rounded-full hover:bg-rose-400 active:scale-95 transition pointer-events-auto"
                    >
                      RETURN LIVE
                    </button>
                  </div>
                ) : activeSubTab === 'intrusion' ? (
                  <div className="bg-teal-500/20 backdrop-blur-md border border-teal-500/40 text-teal-300 px-3 py-1 text-[9px] font-bold uppercase tracking-wider rounded-full shadow-lg">
                    Click card to draw restricted polygon zone!
                  </div>
                ) : null}

                <div className="flex items-center gap-1.5 bg-black/45 backdrop-blur-md border border-white/5 px-3 py-1.5 rounded-full text-[10px] font-bold text-teal-400 font-mono">
                  <Wifi className="w-3.5 h-3.5 text-teal-400" />
                  Latency: 42ms
                </div>
              </div>

            </div>

            {/* Bottom Cards become Live Feed Thumbnail Grid */}
            <div className="grid grid-cols-3 gap-4 h-[78px] flex-shrink-0">
              {cameras.map((cam, idx) => {
                const selected = idx === activeCamIndex;
                return (
                  <div 
                    key={cam.id}
                    onClick={() => setActiveCamIndex(idx)}
                    className={`bg-[#0c1a24]/30 backdrop-blur-md rounded-2xl p-3 border transition duration-300 cursor-pointer flex flex-col justify-between hover:scale-[1.02] hover:border-teal-500/20 select-none ${selected ? 'border-teal-500/40 bg-teal-950/10 shadow-lg shadow-teal-500/5' : 'border-white/5'}`}
                  >
                    <div className="flex justify-between items-center">
                      <span className="text-[9px] font-mono text-zinc-400 font-bold truncate max-w-[80px]">{cam.name}</span>
                      <span className={`w-1.5 h-1.5 rounded-full ${selected ? 'bg-teal-400 animate-pulse' : 'bg-zinc-600'}`} />
                    </div>
                    <div className="flex justify-between items-end">
                      <span className="text-[9px] text-zinc-500 font-bold font-mono uppercase">Check-in</span>
                      <span className="text-xs font-mono font-semibold text-teal-400 leading-none">{cam.lastCheck}</span>
                    </div>
                  </div>
                );
              })}
            </div>

          </div>

          {/* B. RIGHT COLUMN (Surveillance Settings, View Modes, Sliders, Category Toggles & Recent Anomalies) */}
          <div className="flex-1 bg-[#0c1a24]/20 backdrop-blur-xl border border-white/5 rounded-3xl p-6 flex flex-col justify-between shadow-2xl overflow-y-auto">
            
            {/* 1. Model display view toggle (Settings parameters based on active tab) */}
            {activeSubTab === 'control' && (
              <div className="space-y-6 animate-in fade-in duration-200">
                <div className="space-y-4">
                  <div className="text-[10px] text-zinc-400 uppercase font-extrabold tracking-widest">Feed Display Mode</div>
                  <div className="grid grid-cols-2 gap-3">
                    {DISPLAY_MODES.map(mode => {
                      const selected = displayMode === mode.id;
                      return (
                        <button 
                          key={mode.id}
                          onClick={() => setDisplayMode(mode.id)}
                          className={`h-11 rounded-xl text-[10px] font-bold uppercase tracking-wider flex items-center justify-center text-center transition ${selected ? 'bg-gradient-to-r from-teal-500/20 to-cyan-500/20 text-teal-300 border-teal-500/40 border shadow-md' : 'bg-zinc-900/40 border border-white/5 text-zinc-400 hover:bg-zinc-900/60'}`}
                        >
                          {mode.label}
                        </button>
                      );
                    })}
                  </div>
                </div>

                <div className="space-y-4 pt-4 border-t border-white/5">
                  <div className="text-[10px] text-zinc-400 uppercase font-extrabold tracking-widest">Ingestion Source Management</div>
                  <div className="bg-[#040c12]/40 rounded-2xl border border-white/5 p-4 space-y-4">
                    {/* Source Selection Toggle */}
                    <div className="space-y-1.5">
                      <span className="text-[9px] text-zinc-500 font-extrabold uppercase tracking-wider">Source Type</span>
                      <div className="grid grid-cols-2 gap-2 bg-zinc-950/60 p-1 rounded-xl border border-white/5">
                        <button
                          onClick={() => setSourceType('webcam')}
                          className={`py-1.5 rounded-lg text-[9px] font-bold uppercase transition ${sourceType === 'webcam' ? 'bg-teal-500/20 text-teal-300 border border-teal-500/20 shadow-md' : 'text-zinc-500 hover:text-zinc-300'}`}
                        >
                          Webcam Feed
                        </button>
                        <button
                          onClick={() => setSourceType('file')}
                          className={`py-1.5 rounded-lg text-[9px] font-bold uppercase transition ${sourceType === 'file' ? 'bg-teal-500/20 text-teal-300 border border-teal-500/20 shadow-md' : 'text-zinc-500 hover:text-zinc-300'}`}
                        >
                          Video File
                        </button>
                      </div>
                    </div>

                    {/* Conditional inputs */}
                    {sourceType === 'webcam' ? (
                      <div className="space-y-1.5">
                        <span className="text-[9px] text-zinc-500 font-extrabold uppercase tracking-wider block">Camera Device ID</span>
                        <input
                          type="number"
                          min="0"
                          max="10"
                          value={cameraDeviceId}
                          onChange={(e) => setCameraDeviceId(parseInt(e.target.value) || 0)}
                          className="w-full bg-zinc-950/60 border border-white/5 rounded-xl px-3 py-2 text-xs text-white focus:outline-none focus:border-teal-500/50 font-semibold font-mono"
                        />
                      </div>
                    ) : (
                      <div className="space-y-1.5">
                        <span className="text-[9px] text-zinc-500 font-extrabold uppercase tracking-wider block">Video Filepath / Stream URI</span>
                        <input
                          type="text"
                          value={videoFilepath}
                          placeholder="e.g. F:/minipro2/uploads/threat.mp4"
                          onChange={(e) => setVideoFilepath(e.target.value)}
                          className="w-full bg-zinc-950/60 border border-white/5 rounded-xl px-3 py-2 text-xs text-white placeholder-zinc-600 focus:outline-none focus:border-teal-500/50 font-semibold font-mono"
                        />
                      </div>
                    )}

                    {/* Frame Rates Adjustments */}
                    <div className="grid grid-cols-2 gap-4">
                      <div className="space-y-1.5">
                        <div className="flex justify-between text-[9px] font-extrabold uppercase tracking-wider">
                          <span className="text-zinc-500">Ingest FPS:</span>
                          <span className="text-teal-400 font-mono font-bold">{targetFps}</span>
                        </div>
                        <input
                          type="range"
                          min="5"
                          max="30"
                          value={targetFps}
                          onChange={(e) => setTargetFps(parseInt(e.target.value) || 20)}
                          className="w-full accent-teal-400 bg-zinc-800 h-1.5 rounded-lg cursor-pointer"
                        />
                      </div>

                      <div className="space-y-1.5">
                        <div className="flex justify-between text-[9px] font-extrabold uppercase tracking-wider">
                          <span className="text-zinc-500">Inference FPS:</span>
                          <span className="text-teal-400 font-mono font-bold">{(inferenceFps || 2.0).toFixed(1)}</span>
                        </div>
                        <input
                          type="range"
                          min="0.2"
                          max="10.0"
                          step="0.2"
                          value={inferenceFps || 2.0}
                          onChange={(e) => setInferenceFps(parseFloat(e.target.value) || 2.0)}
                          className="w-full accent-teal-400 bg-zinc-800 h-1.5 rounded-lg cursor-pointer"
                        />
                      </div>
                    </div>

                    {/* Action Button */}
                    <button
                      onClick={() => {
                        saveConfig({
                          source_type: sourceType,
                          video_filepath: videoFilepath,
                          camera_device_id: cameraDeviceId,
                          target_fps: targetFps,
                          inference_fps: inferenceFps,
                        });
                        triggerToast("SOURCE CONFIG DISPATCHED");
                      }}
                      className="w-full bg-teal-500 hover:bg-teal-400 text-[#040c12] text-xs font-bold py-2.5 rounded-xl transition shadow-lg shadow-teal-500/10 hover:scale-[1.02] active:scale-95"
                    >
                      Apply Ingestion Changes
                    </button>
                  </div>
                </div>
              </div>
            )}

            {activeSubTab === 'analytics' && (
              <div className="space-y-4">
                <div className="text-[10px] text-zinc-400 uppercase font-extrabold tracking-widest">Model Precision Limits</div>
                <div className="space-y-3.5">
                  {Object.entries(confidenceThresholds).map(([mod, val]) => (
                    <div key={mod} className="space-y-1.5">
                      <div className="flex justify-between text-xs font-semibold capitalize">
                        <span className="text-zinc-400">{mod.replace('_detection', '')} Margin:</span>
                        <span className="text-teal-400 font-bold font-mono">{val * 100}%</span>
                      </div>
                      <input 
                        type="range"
                        min="0.10"
                        max="0.95"
                        step="0.05"
                        value={val}
                        onChange={(e) => handleConfChange(mod, parseFloat(e.target.value))}
                        className="w-full accent-teal-400 bg-zinc-800 h-1.5 rounded-lg cursor-pointer"
                      />
                    </div>
                  ))}
                </div>
              </div>
            )}

            {activeSubTab === 'intrusion' && (
              <div className="space-y-4">
                <div className="text-[10px] text-zinc-400 uppercase font-extrabold tracking-widest">Restricted Zone Perimeter coordinates</div>
                <div className="bg-[#040c12]/40 rounded-2xl border border-white/5 p-4 space-y-3">
                  {restrictedZoneCoords.map((point, idx) => (
                    <div key={idx} className="flex items-center gap-2 text-xs font-semibold justify-between">
                      <span className="text-teal-400 font-mono">Point {idx + 1}:</span>
                      <div className="flex gap-2">
                        <div className="flex items-center gap-1 bg-[#18181b]/50 border border-white/5 rounded-xl px-2 py-0.5">
                          <span className="text-zinc-600 text-[10px] font-mono">X:</span>
                          <input 
                            type="number"
                            value={point[0]}
                            onChange={(e) => handleCoordChange(idx, 'x', parseInt(e.target.value) || 0)}
                            className="bg-transparent w-10 text-white focus:outline-none text-right font-mono"
                          />
                        </div>
                        <div className="flex items-center gap-1 bg-[#18181b]/50 border border-white/5 rounded-xl px-2 py-0.5">
                          <span className="text-zinc-600 text-[10px] font-mono">Y:</span>
                          <input 
                            type="number"
                            value={point[1]}
                            onChange={(e) => handleCoordChange(idx, 'y', parseInt(e.target.value) || 0)}
                            className="bg-transparent w-10 text-white focus:outline-none text-right font-mono"
                          />
                        </div>
                      </div>
                    </div>
                  ))}
                  
                  <button 
                    onClick={() => {
                      saveConfig({ restricted_zone_coords: restrictedZoneCoords });
                      triggerToast("RESTRICTED ZONE SAVED");
                    }}
                    className="w-full bg-teal-500 hover:bg-teal-400 text-[#040c12] text-xs font-bold py-2 rounded-xl transition shadow-lg shadow-teal-500/10 hover:scale-[1.02]"
                  >
                    Apply Coordinates
                  </button>
                </div>
              </div>
            )}

            {activeSubTab === 'authorized' && (
              <div className="space-y-4 animate-in fade-in duration-200">
                <div className="flex items-center justify-between">
                  <div className="text-[10px] text-zinc-400 uppercase font-extrabold tracking-widest">Authorized Restricted-Zone Access</div>
                  <button
                    onClick={() => {
                      setAuthorizedPeople(prev => [...prev, createAuthorizedPerson()]);
                      triggerToast("AUTHORIZED PERSON ADDED");
                    }}
                    className="h-8 w-8 rounded-xl bg-teal-500/15 border border-teal-500/30 text-teal-300 hover:bg-teal-500/25 flex items-center justify-center transition"
                    title="Add authorized person"
                  >
                    <Plus className="w-4 h-4" strokeWidth={1.8} />
                  </button>
                </div>

                <div className="bg-[#040c12]/40 rounded-2xl border border-white/5 p-4 space-y-3 max-h-[430px] overflow-y-auto">
                  {authorizedPeople.length === 0 ? (
                    <div className="py-8 text-center text-xs text-zinc-600 font-semibold">No authorized people configured.</div>
                  ) : (
                    authorizedPeople.map(person => (
                      <div key={person.id} className="bg-zinc-950/40 border border-white/5 rounded-2xl p-3 space-y-3">
                        <div className="flex items-center justify-between gap-3">
                          <div className="flex items-center gap-2 min-w-0">
                            <UserCheck className={`w-4 h-4 flex-shrink-0 ${person.enabled && person.present ? 'text-teal-400' : 'text-zinc-600'}`} strokeWidth={1.6} />
                            <input
                              type="text"
                              value={person.name}
                              placeholder="Person name"
                              onChange={(e) => updateAuthorizedPerson(person.id, { name: e.target.value })}
                              className="bg-transparent text-sm text-white font-semibold focus:outline-none placeholder-zinc-600 min-w-0"
                            />
                          </div>
                          <button
                            onClick={() => removeAuthorizedPerson(person.id)}
                            className="h-7 w-7 rounded-lg bg-rose-500/10 border border-rose-500/20 text-rose-400 hover:bg-rose-500/20 flex items-center justify-center transition"
                            title="Remove authorized person"
                          >
                            <Trash2 className="w-3.5 h-3.5" strokeWidth={1.8} />
                          </button>
                        </div>

                        <div className="grid grid-cols-2 gap-2">
                          <input
                            type="text"
                            value={person.role}
                            placeholder="Role"
                            onChange={(e) => updateAuthorizedPerson(person.id, { role: e.target.value })}
                            className="bg-[#09090b]/60 border border-white/5 rounded-xl px-3 py-2 text-xs text-white focus:outline-none focus:border-teal-500/40"
                          />
                          <input
                            type="text"
                            value={person.access_zones.join(',')}
                            placeholder="CAM_MAIN_ENTRANCE_01"
                            onChange={(e) => updateAuthorizedPerson(person.id, {
                              access_zones: e.target.value.split(',').map(zone => zone.trim()).filter(Boolean),
                            })}
                            className="bg-[#09090b]/60 border border-white/5 rounded-xl px-3 py-2 text-xs text-white focus:outline-none focus:border-teal-500/40 font-mono"
                          />
                        </div>

                        <div className="grid grid-cols-3 gap-2">
                          <button
                            onClick={() => updateAuthorizedPerson(person.id, { enabled: !person.enabled })}
                            className={`py-2 rounded-xl text-[9px] font-extrabold uppercase tracking-wider border transition ${person.enabled ? 'bg-teal-500/15 text-teal-300 border-teal-500/30' : 'bg-zinc-900/40 text-zinc-500 border-white/5'}`}
                          >
                            {person.enabled ? 'Access Enabled' : 'Access Disabled'}
                          </button>
                          <button
                            onClick={() => updateAuthorizedPerson(person.id, { present: !person.present })}
                            className={`py-2 rounded-xl text-[9px] font-extrabold uppercase tracking-wider border transition ${person.present ? 'bg-cyan-500/15 text-cyan-300 border-cyan-500/30' : 'bg-zinc-900/40 text-zinc-500 border-white/5'}`}
                          >
                            {person.present ? 'Currently Present' : 'Not Present'}
                          </button>
                          <button
                            onClick={() => updateAuthorizedPerson(person.id, { intrusion_bypass: !person.intrusion_bypass })}
                            className={`py-2 rounded-xl text-[9px] font-extrabold uppercase tracking-wider border transition ${person.intrusion_bypass ? 'bg-amber-500/15 text-amber-300 border-amber-500/30' : 'bg-zinc-900/40 text-zinc-500 border-white/5'}`}
                          >
                            {person.intrusion_bypass ? 'Bypass On' : 'Bypass Off'}
                          </button>
                        </div>
                      </div>
                    ))
                  )}

                  <button
                    onClick={() => {
                      saveAuthorizedPeople(authorizedPeople);
                      triggerToast("AUTHORIZED ROSTER SAVED");
                    }}
                    className="w-full bg-teal-500 hover:bg-teal-400 text-[#040c12] text-xs font-bold py-2.5 rounded-xl transition shadow-lg shadow-teal-500/10 hover:scale-[1.02] active:scale-95"
                  >
                    Apply Authorized Roster
                  </button>
                </div>
              </div>
            )}

            {activeSubTab === 'notifications' && (
              <div className="space-y-5 animate-in fade-in duration-200">
                <div className="text-[10px] text-zinc-400 uppercase font-extrabold tracking-widest">Notification Dispatch Channels</div>
                <div className="space-y-4">
                  {/* SMS Alerts Dispatch (Primary Channel) */}
                  <div className="space-y-2.5">
                    <div className="flex items-center justify-between bg-zinc-900/20 p-4 rounded-2xl border border-white/5">
                      <div className="flex items-center gap-2">
                        <Smartphone className="w-4 h-4 text-teal-400" strokeWidth={1.5} />
                        <div className="flex flex-col">
                          <span className="text-xs font-bold text-zinc-200 animate-pulse">SMS Mobile Alerts</span>
                          <span className="text-[9px] text-zinc-500 font-medium">Route critical events to mobile network</span>
                        </div>
                      </div>
                      <button 
                        onClick={() => {
                          const next = !smsEnabled;
                          setSmsEnabled(next);
                          saveConfig({ sms_enabled: next });
                        }}
                        className={`w-9 h-5 rounded-full transition-all duration-300 relative ${smsEnabled ? 'bg-teal-500 shadow-lg shadow-teal-500/20' : 'bg-zinc-800'}`}
                      >
                        <div className={`w-3 h-3 rounded-full bg-white absolute top-1 transition-all ${smsEnabled ? 'left-5' : 'left-1'}`} />
                      </button>
                    </div>

                    {smsEnabled && (
                      <div className="space-y-1.5 animate-in slide-in-from-top-1 duration-200">
                        <span className="text-[9px] text-zinc-500 font-extrabold uppercase tracking-wider block">On-Duty Operator Mobile Number</span>
                        <input 
                          type="tel"
                          value={toPhone}
                          placeholder="e.g. +14155552671"
                          onChange={(e) => setToPhone(e.target.value)}
                          onBlur={() => saveConfig({ to_phone: toPhone })}
                          className="w-full bg-[#09090b]/50 border border-white/5 rounded-xl px-3 py-2 text-xs text-white focus:outline-none focus:border-teal-500/50 font-semibold font-mono"
                        />
                      </div>
                    )}
                  </div>

                  {/* Email Alerts Routing (Secondary Channel) */}
                  <div className="space-y-2.5 border-t border-white/5 pt-4">
                    <div className="flex items-center justify-between bg-zinc-900/20 p-4 rounded-2xl border border-white/5">
                      <div className="flex items-center gap-2">
                        <Mail className="w-4 h-4 text-zinc-400" strokeWidth={1.5} />
                        <div className="flex flex-col">
                          <span className="text-xs font-semibold text-zinc-300">Email SMTP Routing</span>
                          <span className="text-[9px] text-zinc-600 font-medium">Backup reporting via SMTP server</span>
                        </div>
                      </div>
                      <button 
                        onClick={() => {
                          const next = !smtpEnabled;
                          setSmtpEnabled(next);
                          saveConfig({ smtp_enabled: next });
                        }}
                        className={`w-9 h-5 rounded-full transition-all duration-300 relative ${smtpEnabled ? 'bg-teal-500' : 'bg-zinc-800'}`}
                      >
                        <div className={`w-3 h-3 rounded-full bg-white absolute top-1 transition-all ${smtpEnabled ? 'left-5' : 'left-1'}`} />
                      </button>
                    </div>

                    {smtpEnabled && (
                      <div className="space-y-1.5 animate-in slide-in-from-top-1 duration-200">
                        <span className="text-[9px] text-zinc-600 font-extrabold uppercase tracking-wider block">Dispatch Email address</span>
                        <input 
                          type="email"
                          value={toAddress}
                          placeholder="e.g. duty-officer@domain.com"
                          onChange={(e) => setToAddress(e.target.value)}
                          onBlur={() => saveConfig({ to_address: toAddress })}
                          className="w-full bg-[#09090b]/50 border border-white/5 rounded-xl px-3 py-2 text-xs text-white focus:outline-none focus:border-teal-500/50 font-semibold"
                        />
                      </div>
                    )}
                  </div>

                  {/* Security Warn Banner */}
                  {smsEnabled && (
                    <div className="p-3 bg-teal-500/10 border border-teal-500/20 rounded-2xl text-[9px] text-teal-400 font-semibold tracking-wide uppercase leading-normal">
                      SMS Channel Active: When critical threat vectors are compiled, automated outbox dispatches will push alert payloads to {toPhone || 'the configured number'} via Twilio REST gateway.
                    </div>
                  )}
                </div>
              </div>
            )}

            {activeSubTab === 'ptz' && (
              <div className="space-y-4 animate-in fade-in duration-200">
                <div className="flex justify-between items-center">
                  <div className="text-[10px] text-zinc-400 uppercase font-extrabold tracking-widest">PTZ Stream Matrix</div>
                  {/* Bounding Box Visibility Toggle (Suggested New Functionalities spec) */}
                  <button 
                    onClick={() => setAnalyticsOverlaysEnabled(!analyticsOverlaysEnabled)}
                    className="flex items-center gap-1.5 text-[9px] font-extrabold tracking-wider uppercase text-zinc-400 hover:text-white transition"
                  >
                    {analyticsOverlaysEnabled ? (
                      <>
                        <Eye className="w-3.5 h-3.5 text-teal-400" /> Toggle overlays
                      </>
                    ) : (
                      <>
                        <EyeOff className="w-3.5 h-3.5 text-zinc-500" /> overlays hidden
                      </>
                    )}
                  </button>
                </div>
                
                <div className="grid grid-cols-2 gap-x-6 gap-y-3">
                  <div className="space-y-1.5">
                    <div className="flex justify-between text-xs font-semibold">
                      <span className="text-zinc-400">Pan Adjust:</span>
                      <span className="text-teal-400 font-mono font-bold">{pan} deg</span>
                    </div>
                    <input 
                      type="range"
                      min="0"
                      max="240"
                      value={pan}
                      onChange={(e) => setPan(parseInt(e.target.value))}
                      className="w-full accent-teal-400 bg-zinc-800 h-1 rounded-lg cursor-pointer"
                    />
                  </div>

                  <div className="space-y-1.5">
                    <div className="flex justify-between text-xs font-semibold">
                      <span className="text-zinc-400">Tilt Angle:</span>
                      <span className="text-teal-400 font-mono font-bold">{tilt} deg</span>
                    </div>
                    <input 
                      type="range"
                      min="0"
                      max="240"
                      value={tilt}
                      onChange={(e) => setTilt(parseInt(e.target.value))}
                      className="w-full accent-teal-400 bg-zinc-800 h-1 rounded-lg cursor-pointer"
                    />
                  </div>

                  <div className="space-y-1.5 col-span-2">
                    <div className="flex justify-between text-xs font-semibold">
                      <span className="text-zinc-400">Zoom Focus:</span>
                      <span className="text-teal-400 font-mono font-bold">{zoom}%</span>
                    </div>
                    <input 
                      type="range"
                      min="100"
                      max="300"
                      value={zoom}
                      onChange={(e) => setZoom(parseInt(e.target.value))}
                      className="w-full accent-teal-400 bg-zinc-800 h-1 rounded-lg cursor-pointer"
                    />
                  </div>
                </div>
              </div>
            )}

            {/* 3. Object Classification Categories (Toggles for Weapon, Fire, Fall, Human, Crowd, Smoke) - Active Filters Tab */}
            {activeSubTab === 'filters' && (
              <div className="space-y-5 animate-in fade-in duration-200">
                <div className="space-y-3">
                  <div className="text-[10px] text-zinc-400 uppercase font-extrabold tracking-widest">Active Detection Filters</div>
                  <div className="grid grid-cols-3 gap-2">
                    {DETECTION_FILTERS.map(m => {
                      const Icon = m.icon;
                      const mappedId = m.id;
                      const active = activeModules.includes(mappedId);
                      return (
                        <div 
                          key={m.id}
                          onClick={() => toggleModule(mappedId)}
                          className={`h-11 rounded-xl flex flex-col justify-center items-center gap-0.5 cursor-pointer transition select-none ${active ? 'bg-gradient-to-r from-teal-500/10 to-cyan-500/10 border border-teal-500/30 text-teal-400 shadow-md shadow-teal-500/5 hover:scale-[1.02]' : 'bg-zinc-900/40 border border-white/5 text-zinc-500 hover:text-zinc-300 hover:scale-[1.02]'}`}
                        >
                          <Icon className="w-4 h-4" strokeWidth={1.5} />
                          <span className="text-[8px] font-bold uppercase tracking-wide leading-none">{m.label}</span>
                        </div>
                      );
                    })}
                  </div>
                </div>

                {/* Action buttons (Direct replication of Reset / Apply buttons from image_1.png) */}
                <div className="flex gap-4 pt-4 border-t border-white/5">
                  <button 
                    onClick={() => {
                      fetchConfig();
                      triggerToast("PARAMETERS RESET");
                    }}
                    className="flex-1 py-2.5 rounded-xl text-xs font-bold bg-transparent border border-zinc-700 text-zinc-300 hover:bg-zinc-900 active:scale-95 transition"
                  >
                    Reset Filters
                  </button>
                  <button 
                    onClick={() => {
                      saveConfig({ active_modules: activeModules });
                      triggerToast("APPLIED CHANGES");
                    }}
                    className="flex-1 py-2.5 rounded-xl text-xs font-bold bg-teal-500 hover:bg-teal-400 text-[#040c12] hover:scale-[1.02] active:scale-95 transition shadow-lg shadow-teal-500/15"
                  >
                    Apply Filters
                  </button>
                </div>
              </div>
            )}

            {/* 4. Recent Anomalies Feed (Logs, clicking log focuses stream) - Recent Anomaly Logs Tab */}
            {activeSubTab === 'logs' && (
              <div className="space-y-4 animate-in fade-in duration-200">
                <div className="text-[10px] text-zinc-400 uppercase font-extrabold tracking-widest flex justify-between items-center">
                  <span>Recent Anomalies Log</span>
                  <button 
                    onClick={clearAlertLogs}
                    className="text-[9px] font-extrabold tracking-wider text-rose-400 hover:text-rose-300 uppercase"
                  >
                    Clear Logs
                  </button>
                </div>

                <div className="bg-[#040c12]/40 rounded-2xl border border-white/5 p-3.5 h-[340px] overflow-y-auto space-y-2">
                  {alertLogs.length === 0 ? (
                    <div className="text-center py-5 text-zinc-600 text-xs font-semibold">No critical incidents registered.</div>
                  ) : (
                    alertLogs.map(log => (
                      <div 
                        key={log.id} 
                        onClick={() => {
                          setActiveCamIndex(0);
                          if (log.snapshot_path) {
                            setReplaySnapshot(snapshotUrl(log.snapshot_path));
                          }
                        }}
                        className="flex justify-between items-center text-xs border-b border-white/5 pb-1.5 last:border-0 last:pb-0 hover:bg-white/5 p-1 rounded transition cursor-pointer"
                      >
                        <div className="flex items-center gap-2">
                          <span className={`w-1.5 h-1.5 rounded-full ${log.severity === 'CRITICAL' ? 'bg-rose-500 animate-ping' : 'bg-amber-500'}`} />
                          <span className="font-semibold text-zinc-200">{log.module.replace(' Detection', '')}</span>
                        </div>
                        <div className="flex items-center gap-3">
                          <span className="text-[9px] text-zinc-500 font-mono">{log.timestamp.split(' ')[1] || log.timestamp}</span>
                          <span className={`text-[9px] font-extrabold px-1.5 py-0.5 rounded leading-none ${log.severity === 'CRITICAL' ? 'bg-rose-500/10 text-rose-400 border border-rose-500/20' : 'bg-amber-500/10 text-amber-400 border border-amber-500/20'}`}>
                            {log.severity}
                          </span>
                        </div>
                      </div>
                    ))
                  )}
                </div>
              </div>
            )}

          </div>

        </div>
        ) : activeView === 'analytics' ? (
          <Analytics />
        ) : activeView === 'history' ? (
          <DetectionHistory alertLogs={alertLogs} snapshotUrl={snapshotUrl} clearAlertLogs={clearAlertLogs} />
        ) : (
          <AlertsSent alertLogs={alertLogs} snapshotUrl={snapshotUrl} smsEnabled={smsEnabled} toPhone={toPhone} />
        )}

      </div>

      {/* Global Toast Notification System */}
      {toast && (
        <div className="fixed bottom-10 left-1/2 -translate-x-1/2 z-50 bg-teal-500/20 border border-teal-500/50 text-teal-400 px-6 py-3 rounded-full shadow-lg backdrop-blur-md animate-in slide-in-from-bottom-5 fade-in duration-300 text-xs font-extrabold uppercase tracking-widest flex items-center gap-2">
          <span>OK</span>
          <span>{toast}</span>
        </div>
      )}

    </div>
  );
}
