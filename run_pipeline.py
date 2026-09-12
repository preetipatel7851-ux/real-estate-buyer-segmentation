"""Run the complete machine-learning pipeline."""

import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

steps = [
    [sys.executable, str(BASE_DIR / "src" / "data_cleaning.py")],
    [sys.executable, str(BASE_DIR / "src" / "eda.py")],
    [sys.executable, str(BASE_DIR / "src" / "clustering.py")],
    [sys.executable, str(BASE_DIR / "src" / "investment_profiling.py")],
]

for step in steps:
    subprocess.run(step, check=True)

print("\nPipeline completed successfully.")
