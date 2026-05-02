import subprocess
import sys
import itertools

MODEL = "qwen"
DATASETS = ["mmlu", "bbh"]
MODES = [str(i) for i in range(1, 11)] + ["qd", "da"]
EXPLAINS = [False, True]

combos = list(itertools.product(DATASETS, MODES, EXPLAINS))
assert len(combos) == 48

for i, (dataset, mode, explain) in enumerate(combos, 1):
    cmd = [sys.executable, "multi_actor.py", "--mode", mode, "--model", MODEL, "--dataset", dataset]
    if explain:
        cmd.append("--explain")

    print(f"\n[{i}/48] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

print("\nAll 48 qwen experiments complete.")
