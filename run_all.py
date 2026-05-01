import subprocess
import sys

def run(script, *args):
    cmd = [sys.executable, script] + list(args)
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print('='*60)
    result = subprocess.run(cmd, check=True)
    return result

# run("llama_mmlu.py")
# run("qwen_mmlu.py")
run("llama_bbh.py")
run("qwen_bbh.py")

run("combine_correct_answers.py", "llama_mmlu_correct.json", "qwen_mmlu_correct.json", "combined_mmlu_correct.json")
run("combine_correct_answers.py", "llama_bbh_correct.json", "qwen_bbh_correct.json", "combined_bbh_correct.json")
