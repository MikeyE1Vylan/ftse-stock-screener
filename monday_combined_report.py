
"""
Optional Monday command-line report.

For the richer interactive experience, use app.py.
This script is intentionally lightweight and can be scheduled with Windows Task Scheduler.
"""
from datetime import datetime
from pathlib import Path
import subprocess
import sys

print("FTSE combined report")
print("Open the Streamlit app to run the latest 7-day or 30-day combined screen:")
print("  streamlit run app.py")
print()
print("Generated:", datetime.now().isoformat(timespec="seconds"))
print()
print("Tip: schedule RUN_APP.bat or launch the app on Monday, then click 'Run combined report'.")
