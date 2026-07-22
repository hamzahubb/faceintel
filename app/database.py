"""
Database — MySQL/MariaDB connection and table management.
Connects to the user's existing XAMPP database and auto-creates
required tables if they don't already exist.
"""

import pymysql
import pymysql.cursors
from pymysql import Error
from datetime import datetime, date, time

# ──────────────────────────────────────────────────────────────
# Connection Configuration (XAMPP MariaDB)
# ──────────────────────────────────────────────────────────────
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "root",
    "database": "facial detector",
}


def get_connection():
    """Create and return a new MySQL connection trying ports 3306/3307 and passwords."""
    # First try connecting directly to database
    for port in [3307, 3306]:
        for pwd in ["", "root"]:
            try:
                cfg = dict(DB_CONFIG)
                cfg["port"] = port
                cfg["password"] = pwd
                return pymysql.connect(**cfg)
            except Error:
                pass

    # If database missing, attempt to connect to server and create database
    for port in [3307, 3306]:
        for pwd in ["", "root"]:
            try:
                conn = pymysql.connect(host=DB_CONFIG["host"], port=port, user=DB_CONFIG["user"], password=pwd)
                cursor = conn.cursor()
                cursor.execute("CREATE DATABASE IF NOT EXISTS `facial detector`")
                conn.commit()
                conn.close()
                # Retry connection to created database
                cfg = dict(DB_CONFIG)
                cfg["port"] = port
                cfg["password"] = pwd
                return pymysql.connect(**cfg)
            except Error:
                pass

    print("[DB Error] Failed to connect to MySQL on ports 3306 and 3307.")
    return None


def init_tables():
    """
    Create the `employees` and `recognition_logs` tables if they
    don't already exist. Does NOT modify or drop any existing tables.
    """
    conn = get_connection()
    if conn is None:
        print("[DB Error] Cannot initialize tables — no database connection.")
        return False

    cursor = conn.cursor()
    try:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS employees (
                id INT AUTO_INCREMENT PRIMARY KEY,
                employee_id VARCHAR(50) UNIQUE NOT NULL,
                full_name VARCHAR(100) NOT NULL,
                department VARCHAR(100) DEFAULT NULL,
                embedding LONGBLOB DEFAULT NULL,
                image_count INT DEFAULT 0,
                registered_at DATETIME DEFAULT CURRENT_TIMESTAMP
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)

        # Recognition logs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS recognition_logs (
                id INT AUTO_INCREMENT PRIMARY KEY,
                employee_id VARCHAR(50) NOT NULL,
                employee_name VARCHAR(100) NOT NULL,
                expression VARCHAR(50) NOT NULL,
                confidence FLOAT NOT NULL,
                log_date DATE NOT NULL,
                log_time TIME NOT NULL
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)

        # Attendance table — one row per employee per day
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS attendance (
                id INT AUTO_INCREMENT PRIMARY KEY,
                employee_id VARCHAR(50) NOT NULL,
                employee_name VARCHAR(100) NOT NULL,
                department VARCHAR(100) DEFAULT NULL,
                check_in DATETIME NOT NULL,
                check_out DATETIME DEFAULT NULL,
                date DATE NOT NULL,
                status VARCHAR(20) DEFAULT 'present',
                UNIQUE KEY unique_daily (employee_id, date)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)

        # Users table — login/authentication accounts
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(50) UNIQUE NOT NULL,
                password_hash VARCHAR(256) NOT NULL,
                full_name VARCHAR(100) NOT NULL,
                face_embedding LONGBLOB DEFAULT NULL,
                avatar TEXT DEFAULT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)

        conn.commit()
        print("[DB] Tables initialized successfully.")
        return True
    except Error as e:
        print(f"[DB Error] Failed to create tables: {e}")
        return False
    finally:
        cursor.close()
        conn.close()


# ──────────────────────────────────────────────────────────────
# Employee CRUD Operations
# ──────────────────────────────────────────────────────────────

def save_employee(employee_id: str, full_name: str, department: str,
                  embedding_bytes: bytes, image_count: int) -> bool:
    """Insert a new employee record with their averaged face embedding."""
    conn = get_connection()
    if conn is None:
        return False

    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO employees (employee_id, full_name, department, embedding, image_count)
            VALUES (%s, %s, %s, %s, %s)
        """, (employee_id, full_name, department or None, embedding_bytes, image_count))
        conn.commit()
        print(f"[DB] Employee '{full_name}' ({employee_id}) registered.")
        return True
    except Error as e:
        print(f"[DB Error] Failed to save employee: {e}")
        return False
    finally:
        cursor.close()
        conn.close()


def get_all_employees() -> list[dict]:
    """Retrieve all employees with their embeddings."""
    conn = get_connection()
    if conn is None:
        return []

    cursor = conn.cursor(pymysql.cursors.DictCursor)
    try:
        cursor.execute("""
            SELECT id, employee_id, full_name, department, embedding, image_count, registered_at
            FROM employees
        """)
        rows = cursor.fetchall()
        return rows
    except Error as e:
        print(f"[DB Error] Failed to fetch employees: {e}")
        return []
    finally:
        cursor.close()
        conn.close()


def get_employee_list() -> list[dict]:
    """Retrieve employees without embeddings (for admin listing)."""
    conn = get_connection()
    if conn is None:
        return []

    cursor = conn.cursor(pymysql.cursors.DictCursor)
    try:
        cursor.execute("""
            SELECT id, employee_id, full_name, department, image_count, registered_at
            FROM employees
        """)
        rows = cursor.fetchall()
        # Convert datetime objects to strings for JSON serialization
        for row in rows:
            if row.get("registered_at"):
                row["registered_at"] = row["registered_at"].strftime("%Y-%m-%d %H:%M:%S")
        return rows
    except Error as e:
        print(f"[DB Error] Failed to fetch employee list: {e}")
        return []
    finally:
        cursor.close()
        conn.close()


def delete_employee(employee_id: str) -> bool:
    """Delete an employee and their corresponding user account by their employee_id."""
    conn = get_connection()
    if conn is None:
        return False

    cursor = conn.cursor()
    try:
        # Delete from employees
        cursor.execute("DELETE FROM employees WHERE employee_id = %s", (employee_id,))
        emp_deleted = cursor.rowcount > 0
        
        # Delete from users
        cursor.execute("DELETE FROM users WHERE username = %s", (employee_id,))
        user_deleted = cursor.rowcount > 0
        
        conn.commit()
        deleted = emp_deleted or user_deleted
        if deleted:
            print(f"[DB] Deleted employee: {emp_deleted}, user account: {user_deleted} for '{employee_id}'")
        return deleted
    except Error as e:
        print(f"[DB Error] Failed to delete employee/user: {e}")
        return False
    finally:
        cursor.close()
        conn.close()


def employee_exists(employee_id: str) -> bool:
    """Check if an employee with the given ID already exists."""
    conn = get_connection()
    if conn is None:
        return False

    cursor = conn.cursor()
    try:
        cursor.execute("SELECT 1 FROM employees WHERE employee_id = %s", (employee_id,))
        return cursor.fetchone() is not None
    except Error as e:
        print(f"[DB Error] Failed to check employee: {e}")
        return False
    finally:
        cursor.close()
        conn.close()


# ──────────────────────────────────────────────────────────────
# Recognition Log Operations
# ──────────────────────────────────────────────────────────────

def log_recognition(employee_id: str, employee_name: str,
                    expression: str, confidence: float) -> bool:
    """Log a recognition event to the database."""
    conn = get_connection()
    if conn is None:
        return False

    cursor = conn.cursor()
    now = datetime.now()
    try:
        cursor.execute("""
            INSERT INTO recognition_logs (employee_id, employee_name, expression, confidence, log_date, log_time)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (employee_id, employee_name, expression, confidence,
              now.date(), now.time().replace(microsecond=0)))
        conn.commit()
        return True
    except Error as e:
        print(f"[DB Error] Failed to log recognition: {e}")
        return False
    finally:
        cursor.close()
        conn.close()


# ──────────────────────────────────────────────────────────────
# Attendance Operations
# ──────────────────────────────────────────────────────────────

def check_in_employee(employee_id: str, employee_name: str,
                      department: str, status: str = "present") -> bool:
    """
    Record a check-in for today. Uses INSERT IGNORE so duplicate
    daily entries are silently skipped (employee already checked in).
    """
    conn = get_connection()
    if conn is None:
        return False

    cursor = conn.cursor()
    now = datetime.now()
    try:
        cursor.execute("""
            INSERT IGNORE INTO attendance
                (employee_id, employee_name, department, check_in, date, status)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (employee_id, employee_name, department or None,
              now, now.date(), status))
        conn.commit()
        inserted = cursor.rowcount > 0
        if inserted:
            print(f"[Attendance] CHECK-IN: {employee_name} ({employee_id}) at {now.strftime('%H:%M:%S')} — {status}")
        return inserted
    except Error as e:
        print(f"[DB Error] Failed to check-in employee: {e}")
        return False
    finally:
        cursor.close()
        conn.close()


def update_check_out(employee_id: str) -> bool:
    """Update the check-out time for today's attendance row."""
    conn = get_connection()
    if conn is None:
        return False

    cursor = conn.cursor()
    now = datetime.now()
    today = now.date()
    try:
        cursor.execute("""
            UPDATE attendance SET check_out = %s
            WHERE employee_id = %s AND date = %s
        """, (now, employee_id, today))
        conn.commit()
        return cursor.rowcount > 0
    except Error as e:
        print(f"[DB Error] Failed to update check-out: {e}")
        return False
    finally:
        cursor.close()
        conn.close()


def get_attendance_by_date(target_date: date) -> list[dict]:
    """Get all attendance records for a specific date."""
    conn = get_connection()
    if conn is None:
        return []

    cursor = conn.cursor(pymysql.cursors.DictCursor)
    try:
        cursor.execute("""
            SELECT employee_id, employee_name, department,
                   check_in, check_out, date, status
            FROM attendance
            WHERE date = %s
            ORDER BY check_in ASC
        """, (target_date,))
        rows = cursor.fetchall()

        # Convert datetime objects for JSON serialization
        for row in rows:
            if row.get("check_in"):
                row["check_in"] = row["check_in"].strftime("%Y-%m-%d %H:%M:%S")
            if row.get("check_out"):
                row["check_out"] = row["check_out"].strftime("%Y-%m-%d %H:%M:%S")
            if row.get("date"):
                row["date"] = row["date"].strftime("%Y-%m-%d")

            # Compute total_hours
            if row.get("check_out") and row.get("check_in"):
                ci = datetime.strptime(row["check_in"], "%Y-%m-%d %H:%M:%S")
                co = datetime.strptime(row["check_out"], "%Y-%m-%d %H:%M:%S")
                diff = co - ci
                total_minutes = int(diff.total_seconds() / 60)
                hours = total_minutes // 60
                minutes = total_minutes % 60
                row["total_hours"] = f"{hours}h {minutes}m"
            else:
                row["total_hours"] = "—"

        return rows
    except Error as e:
        print(f"[DB Error] Failed to fetch attendance: {e}")
        return []
    finally:
        cursor.close()
        conn.close()


def get_all_employee_ids() -> list[str]:
    """Get all registered employee IDs (for computing absent count)."""
    conn = get_connection()
    if conn is None:
        return []

    cursor = conn.cursor()
    try:
        cursor.execute("SELECT employee_id FROM employees")
        return [row[0] for row in cursor.fetchall()]
    except Error as e:
        print(f"[DB Error] Failed to fetch employee IDs: {e}")
        return []
    finally:
        cursor.close()
        conn.close()


def get_employee_attendance_today(employee_id: str) -> dict | None:
    """
    Check if an employee has an attendance record for today.

    Returns a dict with keys: check_in, check_out, status
    or None if no record exists (not checked in today).
    """
    conn = get_connection()
    if conn is None:
        return None

    cursor = conn.cursor(pymysql.cursors.DictCursor)
    try:
        today = date.today().isoformat()
        cursor.execute(
            "SELECT check_in, check_out, status FROM attendance WHERE employee_id = %s AND date = %s",
            (employee_id, today),
        )
        row = cursor.fetchone()
        if row:
            return {
                "check_in": str(row["check_in"]) if row["check_in"] else None,
                "check_out": str(row["check_out"]) if row["check_out"] else None,
                "status": row["status"],
            }
        return None
    except Error as e:
        print(f"[DB Error] Failed to fetch today's attendance for {employee_id}: {e}")
        return None
    finally:
        cursor.close()
        conn.close()


# ──────────────────────────────────────────────────────────────
# System Logs Operations
# ──────────────────────────────────────────────────────────────

def get_all_logs(target_date, event_type=None):
    """
    Retrieve all system logs for a given date, merging data from
    employees (registrations), attendance (check-in/check-out),
    recognition_logs, and cctv_detection_log tables.
    Returns a list of dicts sorted by datetime descending.
    """
    conn = get_connection()
    if conn is None:
        return []

    cursor = conn.cursor(pymysql.cursors.DictCursor)
    logs = []
    try:
        # 1) Registrations
        if event_type is None or event_type in ("all", "registration"):
            cursor.execute("""
                SELECT employee_id, full_name AS employee_name, department,
                       registered_at AS event_time
                FROM employees
                WHERE DATE(registered_at) = %s
            """, (target_date,))
            for row in cursor.fetchall():
                logs.append({
                    "datetime": row["event_time"].strftime("%Y-%m-%d %H:%M:%S") if row["event_time"] else "",
                    "event_type": "Registration",
                    "employee_id": row["employee_id"],
                    "employee_name": row["employee_name"],
                    "details": f"Dept: {row['department'] or '—'}",
                })

        # 2) Check-ins
        if event_type is None or event_type in ("all", "check-in"):
            cursor.execute("""
                SELECT employee_id, employee_name, department,
                       check_in AS event_time, status
                FROM attendance
                WHERE date = %s AND check_in IS NOT NULL
            """, (target_date,))
            for row in cursor.fetchall():
                logs.append({
                    "datetime": row["event_time"].strftime("%Y-%m-%d %H:%M:%S") if row["event_time"] else "",
                    "event_type": "Check-in",
                    "employee_id": row["employee_id"],
                    "employee_name": row["employee_name"],
                    "details": f"Status: {row['status'] or 'present'} · Dept: {row['department'] or '—'}",
                })

        # 3) Check-outs
        if event_type is None or event_type in ("all", "check-out"):
            cursor.execute("""
                SELECT employee_id, employee_name, department,
                       check_out AS event_time
                FROM attendance
                WHERE date = %s AND check_out IS NOT NULL
            """, (target_date,))
            for row in cursor.fetchall():
                logs.append({
                    "datetime": row["event_time"].strftime("%Y-%m-%d %H:%M:%S") if row["event_time"] else "",
                    "event_type": "Check-out",
                    "employee_id": row["employee_id"],
                    "employee_name": row["employee_name"],
                    "details": f"Dept: {row['department'] or '—'}",
                })

        # 4) Recognition logs
        if event_type is None or event_type in ("all", "recognition"):
            cursor.execute("""
                SELECT employee_id, employee_name, expression,
                       confidence, log_date, log_time
                FROM recognition_logs
                WHERE log_date = %s
            """, (target_date,))
            for row in cursor.fetchall():
                dt_str = ""
                if row["log_date"] and row["log_time"]:
                    log_time = row["log_time"]
                    if isinstance(log_time, time):
                        dt = datetime.combine(row["log_date"], log_time)
                    else:
                        dt = datetime.combine(row["log_date"], (datetime.min + log_time).time())
                    dt_str = dt.strftime("%Y-%m-%d %H:%M:%S")
                logs.append({
                    "datetime": dt_str,
                    "event_type": "Recognition",
                    "employee_id": row["employee_id"],
                    "employee_name": row["employee_name"],
                    "details": f"Expression: {row['expression']} · Confidence: {round(row['confidence'] * 100)}%",
                })

        # 5) CCTV detection logs
        if event_type is None or event_type in ("all", "cctv"):
            cursor.execute("""
                SELECT employee_id, employee_name, camera_name,
                       expression, confidence, detected_at
                FROM cctv_detection_log
                WHERE DATE(detected_at) = %s
            """, (target_date,))
            for row in cursor.fetchall():
                logs.append({
                    "datetime": row["detected_at"].strftime("%Y-%m-%d %H:%M:%S") if row["detected_at"] else "",
                    "event_type": "CCTV Detection",
                    "employee_id": row["employee_id"],
                    "employee_name": row["employee_name"],
                    "details": f"Camera: {row['camera_name']} · Expression: {row['expression'] or '—'} · Confidence: {round(row['confidence'] * 100)}%",
                })

        # Sort all logs by datetime descending
        logs.sort(key=lambda x: x["datetime"], reverse=True)
        return logs

    except Error as e:
        print(f"[DB Error] Failed to fetch logs: {e}")
        return []
    finally:
        cursor.close()
        conn.close()


# ──────────────────────────────────────────────────────────────
# User Authentication CRUD
# ──────────────────────────────────────────────────────────────

def create_user(username: str, password_hash: str, full_name: str,
                face_embedding_bytes: bytes = None) -> bool:
    """Insert a new user account for login/authentication."""
    conn = get_connection()
    if conn is None:
        return False

    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO users (username, password_hash, full_name, face_embedding)
            VALUES (%s, %s, %s, %s)
        """, (username, password_hash, full_name, face_embedding_bytes))
        conn.commit()
        return True
    except Error as e:
        print(f"[DB Error] Failed to create user: {e}")
        return False
    finally:
        cursor.close()
        conn.close()


def get_user_by_username(username: str) -> dict | None:
    """Retrieve a user record by username."""
    conn = get_connection()
    if conn is None:
        return None

    cursor = conn.cursor(pymysql.cursors.DictCursor)
    try:
        cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
        return cursor.fetchone()
    except Error as e:
        print(f"[DB Error] Failed to get user: {e}")
        return None
    finally:
        cursor.close()
        conn.close()


def get_user_by_id(user_id: int) -> dict | None:
    """Retrieve a user record by id."""
    conn = get_connection()
    if conn is None:
        return None

    cursor = conn.cursor(pymysql.cursors.DictCursor)
    try:
        cursor.execute("SELECT id, username, full_name, avatar, created_at FROM users WHERE id = %s", (user_id,))
        return cursor.fetchone()
    except Error as e:
        print(f"[DB Error] Failed to get user by id: {e}")
        return None
    finally:
        cursor.close()
        conn.close()


def get_all_users_with_embedding() -> list[dict]:
    """Retrieve all users that have a face embedding stored (for face login matching)."""
    conn = get_connection()
    if conn is None:
        return []

    cursor = conn.cursor(pymysql.cursors.DictCursor)
    try:
        cursor.execute("""
            SELECT id, username, full_name, face_embedding
            FROM users
            WHERE face_embedding IS NOT NULL
        """)
        return cursor.fetchall()
    except Error as e:
        print(f"[DB Error] Failed to get users with embeddings: {e}")
        return []
    finally:
        cursor.close()
        conn.close()
