import subprocess
import sys
import time

def run_timed(label, *args):
    cmd = [sys.executable, "multi_actor.py"] + list(args)
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print('='*60)
    start = time.time()
    subprocess.run(cmd, check=True)
    elapsed = time.time() - start
    minutes, seconds = divmod(int(elapsed), 60)
    print(f"\n[{label}] Time: {minutes}m {seconds:02d}s")
    return elapsed

t1 = run_timed("baseline", "--mode", "1", "--model", "llama", "--dataset", "mmlu")
t2 = run_timed("explain",  "--mode", "1", "--model", "llama", "--dataset", "mmlu", "--explain")

print(f"\n{'='*60}")
print(f"Summary:")
m1, s1 = divmod(int(t1), 60)
m2, s2 = divmod(int(t2), 60)
total = t1 + t2
mt, st = divmod(int(total), 60)
print(f"  baseline : {m1}m {s1:02d}s")
print(f"  explain  : {m2}m {s2:02d}s")
print(f"  total    : {mt}m {st:02d}s")
print('='*60)
