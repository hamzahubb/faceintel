#!/usr/bin/env python3
"""
Infigo FaceIntel — Background CCTV Surveillance Service

Runs independently of the Flask web server, continuously monitoring
IP CCTV cameras for automated attendance and expression detection.

Usage:
    python run_surveillance.py                        # Default config
    python run_surveillance.py --config custom.json   # Custom config
    python run_surveillance.py --offline              # Offline video only

The Flask dashboard (python app.py) can run simultaneously for
administration and reporting. Both services share the same database.
"""

from surveillance.camera_service import main

if __name__ == "__main__":
    main()
