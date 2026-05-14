"""
yfinance_lock.py
Lock global para prevenir runs concurrentes de scripts que llaman a yfinance
desde la misma IP. Yahoo Finance rate-limita por IP, no por proceso: dos
scripts simultaneos golpeando yfinance = doble carga + bloqueo casi seguro.

Uso al inicio de un script (cron_diario, recovery_incremental, snapshot):

    from src.utils.yfinance_lock import acquire
    acquire("cron_diario --step precios")
    # ... resto del script

Si otro script tiene el lock activo -> exit(1) con mensaje claro.
El lock se libera automaticamente al terminar (atexit).
Locks de procesos crasheados (>2 hs sin actualizar) se ignoran como stale.
"""
import os
import sys
import atexit
import time
from pathlib import Path

# Lockfile en logs/ del proyecto (cross-platform, persistente, inspeccionable)
_ROOT = Path(__file__).resolve().parent.parent.parent
_LOCK_PATH = _ROOT / "logs" / "yfinance.lock"
_STALE_AFTER_SEC = 7200  # 2 horas: si el lock es mas viejo, lo consideramos stale


def acquire(script_name: str) -> None:
    """
    Adquiere el lock global de yfinance.
    Si otro script lo tiene, imprime mensaje y exit(1).
    """
    _LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)

    if _LOCK_PATH.exists():
        try:
            content = _LOCK_PATH.read_text(encoding="utf-8").strip()
            parts  = content.split("|")
            old_pid = int(parts[0])
            ts      = float(parts[1])
            owner   = parts[2] if len(parts) > 2 else "unknown"
            age     = time.time() - ts

            if age > _STALE_AFTER_SEC:
                print(f"  [YF-LOCK] Stale lock de '{owner}' (PID {old_pid}, edad {age:.0f}s). Removiendo.")
                _LOCK_PATH.unlink()
            elif _process_alive(old_pid):
                print(f"  [YF-LOCK] yfinance YA EN USO por '{owner}' (PID {old_pid}, edad {age:.0f}s).")
                print(f"  [YF-LOCK] Abortando para evitar carga concurrente sobre la IP.")
                print(f"  [YF-LOCK] Lockfile: {_LOCK_PATH}")
                sys.exit(1)
            else:
                print(f"  [YF-LOCK] PID {old_pid} no existe. Removiendo lock huerfano.")
                _LOCK_PATH.unlink()
        except (ValueError, IndexError, OSError) as e:
            print(f"  [YF-LOCK] Lockfile corrupto ({e}). Removiendo.")
            try:
                _LOCK_PATH.unlink()
            except OSError:
                pass

    # Crear lock con info del proceso
    pid = os.getpid()
    _LOCK_PATH.write_text(f"{pid}|{time.time()}|{script_name}", encoding="utf-8")
    atexit.register(_release)
    print(f"  [YF-LOCK] Adquirido por '{script_name}' (PID {pid})")


def _process_alive(pid: int) -> bool:
    """Cross-platform: True si el PID corresponde a un proceso vivo."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def _release() -> None:
    """Libera el lock (llamado automaticamente por atexit)."""
    try:
        if _LOCK_PATH.exists():
            _LOCK_PATH.unlink()
    except OSError:
        pass
