import json
import os
from pathlib import Path
from typing import Optional


class BotStateStore:
    """Minimal persistent state for the daemon.

    Purpose
    -------
    In daemon mode, WebSocket streams can reconnect and re-deliver the same
    "kline closed" event. We therefore need an idempotency key that survives
    process restarts. We persist the most recently *processed* trigger timestamp
    (bar close time, ms).

    Notes
    -----
    - The file is written atomically (write temp + replace) to reduce corruption
      risk on crashes.
    - Only store non-sensitive operational state here.
    """

    def __init__(self, path: str = "./runtime/state.json"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load_last_processed(self) -> Optional[int]:
        """Return last processed trigger close timestamp (ms), or None."""
        if not self.path.exists():
            return None
        data = json.loads(self.path.read_text(encoding="utf-8"))
        v = data.get("last_processed_bar_close_ts_ms")
        return int(v) if v is not None else None

    def save_last_processed(self, ts_ms: int) -> None:
        """Persist last processed trigger close timestamp (ms), atomically."""
        payload = json.dumps(
            {"last_processed_bar_close_ts_ms": int(ts_ms)},
            ensure_ascii=False,
            indent=2,
        )
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(payload, encoding="utf-8")
        # Best-effort fsync on POSIX; harmless on others.
        try:
            fd = os.open(tmp, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        except Exception:
            pass
        tmp.replace(self.path)
