from __future__ import annotations

import time
class RuntimeLogger:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self.start_time = time.perf_counter()
        self.last_time = self.start_time
        self.last_step = 0

    def event(self, message: str) -> None:
        if self.enabled:
            print(f"[dia2] {message}")

    def progress(self, step: int, total: Optional[int] = None) -> None:
        if not self.enabled:
            return
        now = time.perf_counter()
        delta_t = max(now - self.last_time, 1e-6)
        delta_steps = max(step - self.last_step, 1)
        speed = delta_steps / delta_t
        if total is None:
            self.event(f"step {step} :: {speed:.1f} toks/s")
        else:
            self.event(f"step {step}/{total} :: {speed:.1f} toks/s")
        self.last_time = now
        self.last_step = step

    def elapsed(self) -> float:
        return time.perf_counter() - self.start_time


__all__ = ["RuntimeLogger"]
