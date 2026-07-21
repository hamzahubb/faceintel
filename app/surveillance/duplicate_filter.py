"""
Duplicate Filter — thread-safe deduplication and cooldown management
for attendance writes and recognition logs across multiple camera threads.
"""

import threading
import time


class DuplicateFilter:
    """
    Prevents duplicate database writes by enforcing per-employee cooldowns.

    Thread-safe: multiple camera threads can call methods concurrently.
    The database's UNIQUE KEY (employee_id, date) on the attendance table
    acts as a safety net, but this filter reduces unnecessary DB calls.
    """

    def __init__(self, attendance_cooldown: int = 300, log_cooldown: int = 10):
        """
        Args:
            attendance_cooldown: Minimum seconds between attendance writes
                                 for the same employee (default 5 minutes).
            log_cooldown: Minimum seconds between recognition log writes
                          for the same employee (default 10 seconds).
        """
        self._attendance_cooldown = attendance_cooldown
        self._log_cooldown = log_cooldown

        self._attendance_timestamps: dict[str, float] = {}
        self._log_timestamps: dict[str, float] = {}
        self._checkout_timestamps: dict[str, float] = {}
        self._lock = threading.Lock()

    def should_record_attendance(self, employee_id: str) -> bool:
        """Check if enough time has passed to write an attendance record."""
        with self._lock:
            last = self._attendance_timestamps.get(employee_id, 0.0)
            return (time.time() - last) > self._attendance_cooldown

    def mark_attendance_recorded(self, employee_id: str):
        """Mark that an attendance record was just written."""
        with self._lock:
            self._attendance_timestamps[employee_id] = time.time()

    def should_log_recognition(self, employee_id: str) -> bool:
        """Check if enough time has passed to log a recognition event."""
        with self._lock:
            last = self._log_timestamps.get(employee_id, 0.0)
            return (time.time() - last) > self._log_cooldown

    def mark_recognition_logged(self, employee_id: str):
        """Mark that a recognition log was just written."""
        with self._lock:
            self._log_timestamps[employee_id] = time.time()

    def should_record_checkout(self, employee_id: str) -> bool:
        """Check if enough time has passed to write a checkout record."""
        with self._lock:
            last = self._checkout_timestamps.get(employee_id, 0.0)
            return (time.time() - last) > self._attendance_cooldown

    def mark_checkout_recorded(self, employee_id: str):
        """Mark that a checkout record was just written."""
        with self._lock:
            self._checkout_timestamps[employee_id] = time.time()

    def reset(self):
        """Clear all tracked state."""
        with self._lock:
            self._attendance_timestamps.clear()
            self._log_timestamps.clear()
            self._checkout_timestamps.clear()
