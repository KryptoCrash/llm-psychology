import subprocess
import sys
import itertools

MODELS = ["llama", "qwen"]
DATASETS = ["mmlu", "bbh"]
MODES = [str(i) for i in range(1, 11)] + ["qd", "da"]
EXPLAINS = [False, True]

combos = list(itertools.product(MODELS, DATASETS, MODES, EXPLAINS))
assert len(combos) == 96

for i, (model, dataset, mode, explain) in enumerate(combos, 1):
    cmd = [sys.executable, "multi_actor.py", "--mode", mode, "--model", model, "--dataset", dataset]
    if explain:
        cmd.append("--explain")

    print(f"\n[{i}/96] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

print("\nAll 96 experiments complete.")
