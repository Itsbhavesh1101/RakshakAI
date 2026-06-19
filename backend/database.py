import sqlite3
from typing import List, Dict, Any, Optional
from contextlib import contextmanager

from config import DB_PATH, SNAPSHOTS_DIR

SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


@contextmanager
def _connect():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()

def init_db():
    """Initializes the SQLite alerts database table schema."""
    with _connect() as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                module TEXT,
                severity TEXT,
                message TEXT,
                snapshot_path TEXT,
                ai_description TEXT
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts(timestamp)")
    print(f"SQLite DB initialized successfully at '{DB_PATH}'")

def log_alert_to_db(
    timestamp: str, 
    module_name: str, 
    severity: str, 
    message: str, 
    snapshot_filename: str
) -> int:
    """Logs a raw alert entry and returns its unique ID."""
    with _connect() as conn:
        cursor = conn.execute("""
            INSERT INTO alerts (timestamp, module, severity, message, snapshot_path, ai_description)
            VALUES (?, ?, ?, ?, ?, NULL)
        """, (timestamp, module_name, severity, message, snapshot_filename))
        return int(cursor.lastrowid)

def update_alert_description(alert_id: int, ai_description: str):
    """Enriches an alert entry with Ollama AI description text."""
    with _connect() as conn:
        conn.execute("""
            UPDATE alerts
            SET ai_description = ?
            WHERE id = ?
        """, (ai_description, alert_id))
    print(f"SQLite DB enriched alert #{alert_id} with description.")

def fetch_alert_history(limit: Optional[int] = 100) -> List[Dict[str, Any]]:
    """Retrieves all alert history logs in chronological order."""
    if not DB_PATH.exists():
        return []

    sql = """
        SELECT id, timestamp, module, severity, message, snapshot_path, ai_description
        FROM alerts
        ORDER BY id DESC
    """
    params: tuple[Any, ...] = ()
    if limit is not None and limit > 0:
        sql += " LIMIT ?"
        params = (limit,)

    with _connect() as conn:
        rows = conn.execute(sql, params).fetchall()

    alerts = []
    for r in rows:
        alerts.append({
            "id": r["id"],
            "timestamp": r["timestamp"],
            "module": r["module"],
            "severity": r["severity"],
            "message": r["message"],
            "snapshot_path": r["snapshot_path"],
            "ai_description": r["ai_description"]
        })
    return alerts

def clear_alerts_table():
    """Wipes all rows inside the alerts table and deletes physical snapshot JPEG files."""
    if DB_PATH.exists():
        with _connect() as conn:
            conn.execute("DELETE FROM alerts")
        
    for file_path in SNAPSHOTS_DIR.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            try:
                file_path.unlink()
            except Exception as e:
                print(f"Error removing snapshot file {file_path}: {e}")
                
    print("Database logs and snapshots cleared.")
