import itertools
from multi_actor import load_model, run_experiment

MODEL = "qwen"
DATASETS = ["mmlu", "bbh"]
MODES = [str(i) for i in range(1, 11)] + ["qd", "da"]
EXPLAINS = [False, True]

combos = list(itertools.product(DATASETS, MODES, EXPLAINS))
assert len(combos) == 48

model, device = load_model(MODEL)

for i, (dataset, mode, explain) in enumerate(combos, 1):
    print(f"\n[{i}/48] model={MODEL} dataset={dataset} mode={mode} explain={explain}")
    run_experiment(MODEL, dataset, mode, explain=explain, model=model, device=device)

print("\nAll 48 qwen experiments complete.")
