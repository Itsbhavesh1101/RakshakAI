export interface MockIncidentLog {
  id: number;
  timestamp: string;
  cameraId: string;
  threatClassification: string;
  severity: 'CRITICAL' | 'WARNING' | 'MEDIUM' | 'LOW';
  confidence: number;
  actionTaken: string;
}

export interface MockLiveAlert {
  id: number;
  type: 'weapon' | 'fire' | 'fall' | 'trespassing' | 'crowd';
  severity: 'CRITICAL' | 'WARNING' | 'MEDIUM';
  timestamp: string;
  cameraId: string;
  description: string;
}

export const mockIncidentLogs: MockIncidentLog[] = [
  {
    id: 1,
    timestamp: '2026-05-25 11:20:45',
    cameraId: 'CAM_MAIN_ENTRANCE_01',
    threatClassification: 'Weapon Detection',
    severity: 'CRITICAL',
    confidence: 89.4,
    actionTaken: 'Patrol dispatched & SMTP email sent'
  },
  {
    id: 2,
    timestamp: '2026-05-25 09:15:32',
    cameraId: 'CAM_PERIMETER_03',
    threatClassification: 'Trespassing Violation',
    severity: 'CRITICAL',
    confidence: 94.2,
    actionTaken: 'Siren triggered & backup notified'
  },
  {
    id: 3,
    timestamp: '2026-05-25 08:44:12',
    cameraId: 'CAM_MAIN_ENTRANCE_01',
    threatClassification: 'Crowd Anomaly',
    severity: 'WARNING',
    confidence: 82.5,
    actionTaken: 'Operator logged warning state'
  },
  {
    id: 4,
    timestamp: '2026-05-24 23:10:04',
    cameraId: 'CAM_BACK_LOBBY_02',
    threatClassification: 'Fall Detected',
    severity: 'MEDIUM',
    confidence: 76.8,
    actionTaken: 'Medical dispatch notified via terminal'
  },
  {
    id: 5,
    timestamp: '2026-05-24 19:30:15',
    cameraId: 'CAM_PERIMETER_03',
    threatClassification: 'Smoke Detection',
    severity: 'CRITICAL',
    confidence: 91.0,
    actionTaken: 'Sprinkler routing verify and local email'
  },
  {
    id: 6,
    timestamp: '2026-05-24 15:45:50',
    cameraId: 'CAM_MAIN_ENTRANCE_01',
    threatClassification: 'Weapon Detection',
    severity: 'CRITICAL',
    confidence: 88.0,
    actionTaken: 'Tactical intercept logged'
  },
  {
    id: 7,
    timestamp: '2026-05-24 12:04:12',
    cameraId: 'CAM_BACK_LOBBY_02',
    threatClassification: 'Human Intrusion',
    severity: 'LOW',
    confidence: 72.1,
    actionTaken: 'Ignored (authorized staff access)'
  }
];

export const mockLiveAlerts: MockLiveAlert[] = [
  {
    id: 101,
    type: 'weapon',
    severity: 'CRITICAL',
    timestamp: '11:20:45 AM',
    cameraId: 'CAM_MAIN_ENTRANCE_01',
    description: 'Active weapon brandishing detected near gate A.'
  },
  {
    id: 102,
    type: 'fire',
    severity: 'CRITICAL',
    timestamp: '10:52:12 AM',
    cameraId: 'CAM_PERIMETER_03',
    description: 'Structural flame and smoke plume triggered.'
  },
  {
    id: 103,
    type: 'fall',
    severity: 'MEDIUM',
    timestamp: '10:14:02 AM',
    cameraId: 'CAM_BACK_LOBBY_02',
    description: 'Individual slip-and-fall detected near rear stairs.'
  },
  {
    id: 104,
    type: 'trespassing',
    severity: 'CRITICAL',
    timestamp: '09:15:32 AM',
    cameraId: 'CAM_PERIMETER_03',
    description: 'Polygon restricted trespassing intrusion violation.'
  },
  {
    id: 105,
    type: 'crowd',
    severity: 'WARNING',
    timestamp: '08:44:12 AM',
    cameraId: 'CAM_MAIN_ENTRANCE_01',
    description: 'Overcrowding capacity threshold limit breached.'
  }
];

export const mockChartData = [
  { time: '08:00', weapons: 0, fire: 0, fall: 1, trespassing: 2 },
  { time: '10:00', weapons: 1, fire: 1, fall: 0, trespassing: 4 },
  { time: '12:00', weapons: 0, fire: 0, fall: 2, trespassing: 1 },
  { time: '14:00', weapons: 2, fire: 0, fall: 1, trespassing: 3 },
  { time: '16:00', weapons: 0, fire: 1, fall: 0, trespassing: 0 },
  { time: '18:00', weapons: 1, fire: 0, fall: 1, trespassing: 5 },
  { time: '20:00', weapons: 0, fire: 0, fall: 0, trespassing: 2 }
];
