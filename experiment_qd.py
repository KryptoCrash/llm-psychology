import subprocess
import sys

for n in range(1, 11):
    print(f"Running n={n}...")
    subprocess.run([sys.executable, "multi_actor.py", str(n), "--qd"], check=True)
