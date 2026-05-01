"""Quick smoke-test for multi_actor.py with --dataset bbh (~10 questions)."""
import json
import shutil
import subprocess
import sys

COMBINED = "combined_bbh_correct.json"
BACKUP = "combined_bbh_correct_backup.json"
N_TEST = 10

# Shrink the dataset to N_TEST questions for the duration of the test
shutil.copy(COMBINED, BACKUP)
data = json.load(open(COMBINED))
json.dump(data[:N_TEST], open(COMBINED, "w"), indent=2)

def run(n, extra_flags=()):
    cmd = [sys.executable, "multi_actor.py", str(n), "--dataset", "bbh"] + list(extra_flags)
    print(f"\n>>> {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

try:
    run(1)           # solo, no actors
    run(2)           # single conformity actor
    run(3)           # multi-actor
finally:
    shutil.copy(BACKUP, COMBINED)
    print(f"\nRestored {COMBINED}")
