"""
Helper script to run the Streamlit dashboard.
"""

import subprocess
import sys
from config import DASHBOARD_PORT

if __name__ == "__main__":
    print("="*70)
    print("Starting Network Traffic Classification Dashboard")
    print("="*70)
    print(f"Dashboard will open automatically in your browser")
    print(f"If it doesn't open, visit: http://localhost:{DASHBOARD_PORT}")
    print("="*70)
    print()

    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        "dashboard.py",
        f"--server.port={DASHBOARD_PORT}",
        "--server.address=localhost"
    ])
