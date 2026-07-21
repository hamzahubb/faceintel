"""
CCTV Detection Log — supplementary database table for surveillance-specific
detection records, including camera source information.

This is purely additive — the existing recognition_logs table is untouched.
"""

import sys
import os

# Add project root to path so we can import database module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import get_connection
from pymysql import Error
from datetime import datetime


def init_cctv_tables() -> bool:
    """
    Create the cctv_detection_log table if it doesn't exist.
    Does NOT modify or drop any existing tables.
    """
    conn = get_connection()
    if conn is None:
        print("[Surveillance DB] Cannot initialize CCTV tables — no database connection.")
        return False

    cursor = conn.cursor()
    try:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cctv_detection_log (
                id INT AUTO_INCREMENT PRIMARY KEY,
                employee_id VARCHAR(50) NOT NULL,
                employee_name VARCHAR(100) NOT NULL,
                camera_id VARCHAR(50) NOT NULL,
                camera_name VARCHAR(100) NOT NULL,
                expression VARCHAR(50) DEFAULT NULL,
                confidence FLOAT NOT NULL,
                detected_at DATETIME NOT NULL,
                INDEX idx_employee_date (employee_id, detected_at),
                INDEX idx_camera (camera_id, detected_at)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)
        conn.commit()
        print("[Surveillance DB] CCTV detection log table initialized.")
        return True
    except Error as e:
        print(f"[Surveillance DB Error] Failed to create CCTV tables: {e}")
        return False
    finally:
        cursor.close()
        conn.close()


def log_cctv_detection(
    employee_id: str,
    employee_name: str,
    camera_id: str,
    camera_name: str,
    expression: str | None,
    confidence: float,
) -> bool:
    """
    Log a CCTV detection event with camera context.

    Args:
        employee_id: Detected employee's ID.
        employee_name: Detected employee's full name.
        camera_id: ID of the camera that captured the detection.
        camera_name: Human-readable camera name.
        expression: Dominant facial expression (or None if unavailable).
        confidence: Recognition confidence score (0.0 – 1.0).

    Returns:
        True if the log was written successfully.
    """
    conn = get_connection()
    if conn is None:
        return False

    cursor = conn.cursor()
    now = datetime.now()
    try:
        cursor.execute(
            """
            INSERT INTO cctv_detection_log
                (employee_id, employee_name, camera_id, camera_name,
                 expression, confidence, detected_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (employee_id, employee_name, camera_id, camera_name,
             expression or "unknown", confidence, now),
        )
        conn.commit()
        return True
    except Error as e:
        print(f"[Surveillance DB Error] Failed to log CCTV detection: {e}")
        return False
    finally:
        cursor.close()
        conn.close()
