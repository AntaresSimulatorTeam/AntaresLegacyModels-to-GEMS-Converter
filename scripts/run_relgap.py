from pathlib import Path
import sys

# Ensure project root is on sys.path so imports work when running script directly
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
# Also ensure the `src` folder is on sys.path so packages under `src/` import by name
src_path = repo_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from src.antares_runner.antares_runner import AntaresHybridStudyBenchmarker
# helper to add hybrid behaviour to a study (copies template input files)
from tests.antares_historic.utils import addHybridBehaviorPerso
from shutil import copy2
from datetime import datetime

# User-provided values (edit if needed)
exec_path = Path(r"C:\Users\jeannecor\Documents\1-PROJECTS\OPEN SOURCE\rte-antares-9.3.6-installer-64bits\bin")
study1 = Path(r"C:\Users\jeannecor\Documents\1-PROJECTS\OPEN SOURCE\BP25")
study2 = Path(r"C:\Users\jeannecor\Documents\1-PROJECTS\OPEN SOURCE\antares-studies-converted\BP25")
solver = None  # or 'mumps', 'ma57', etc.

print(f"Using antares exec: {exec_path}")
print(f"Study 1: {study1}")
print(f"Study 2: {study2}")

# Backup study1 generaldata.ini before addHybridBehaviorPerso (it will be overwritten)
settings_file = study1 / "settings" / "generaldata.ini"
if settings_file.exists():
    backup_dir = repo_root / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / f"{study1.name}_generaldata_{datetime.now().strftime('%Y%m%dT%H%M%S')}.ini"
    copy2(settings_file, backup_path)
    print(f"Backed up {settings_file} to {backup_path}")
else:
    print(f"No generaldata.ini found at {settings_file}; nothing to back up.")

# Ensure hybrid behaviour files are present for study1 so Antares produces simulation tables
print("Ensuring hybrid behaviour files are present for study1...")
addHybridBehaviorPerso(study1)

bench = AntaresHybridStudyBenchmarker(exec_path, study1, study2, solver)
print("Running antares solver for both studies (this may take a while)...")
bench.run()
rel1, rel2 = bench.weekly_rel_gaps()
print("Weekly relative gaps (optim 1):")
print(rel1)
print("Weekly relative gaps (optim 2):")
print(rel2)

# Optionally print max rel gap
import numpy as np
if rel1.size:
    print("Max weekly rel gap (optim 1):", np.max(rel1))
if rel2.size:
    print("Max weekly rel gap (optim 2):", np.max(rel2))
