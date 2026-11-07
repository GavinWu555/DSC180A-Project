"""
Resumable experiment runner for tasks2.py (math_dependent_mid).
 
Runs every (task x condition x think-setting x repeat) cell, scores it, and appends one JSON
line per cell to --out. Re-running the same command skips cells that are already done, so you
can stop and restart freely.
 
Examples
  # smoke test the pipeline with no model at all
  python run_experiment2.py --dry-run --n 2 --ks 4 --out dry.jsonl
 
  # the confirmation run: 100 items, 12 steps, fresh task seed, thinking off
  python run_experiment2.py --model qwen3.5:4b --n 100 --ks 12 --task-seed 1 --think off --out confirm.jsonl
 
  # sampled run for robustness: 3 repeats with different seeds
  python run_experiment2.py --model qwen3.5:4b --n 20 --ks 12 --temperature 0.7 --top-p 0.8 --top-k 20 --repeats 3 --out sampled.jsonl
"""
import argparse
import json
import os
import sys
import time
 
import tasks as T
 
 
def dry_llm(messages):
    """Stub model: exercises the pipeline without calling anything."""
    dry_llm.usage["calls"] += 1
    return "Step 1: 0\nStep 2: 0"
 
 
dry_llm.usage = {"calls": 0, "prompt_tokens": 0, "output_tokens": 0, "truncated": 0}
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5:4b")
    ap.add_argument("--n", type=int, default=20, help="items per chain length k")
    ap.add_argument("--ks", type=int, nargs="+", default=[12], help="chain lengths (steps per task)")
    ap.add_argument("--conditions", nargs="+", default=T.CONDITIONS, choices=T.CONDITIONS)
    ap.add_argument("--think", nargs="+", default=["off"], choices=["off", "on"])
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--task-seed", type=int, default=0)
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--out", default="results.jsonl")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
 
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            r = json.loads(line)
            done.add((r["id"], r["condition"], r["think"], r["repeat"], r.get("task_seed", 0)))
    print(f"{len(done)} cells already done")
 
    jobs = []
    for k in a.ks:
        for i in range(a.n):
            for think in a.think:
                for rep in range(a.repeats):
                    for cond in a.conditions:
                        jobs.append((T.makeTask(k, i, a.task_seed), cond, think, rep))
    print(f"{len(jobs)} cells total")
 
    llms = {}
    t0 = time.time()
    consecutive_errors = 0
    with open(a.out, "a") as f:
        for n, (task, cond, think, rep) in enumerate(jobs, 1):
            if (task["id"], cond, think, rep, a.task_seed) in done:
                continue
            key = (think, rep)
            if key not in llms:
                llms[key] = dry_llm if a.dry_run else T.ollama_llm(
                    a.model, think=(think == "on"), temperature=a.temperature, top_p=a.top_p,
                    top_k=a.top_k, seed=1000 + rep, host=a.host)
            llm = llms[key]
            before = dict(llm.usage)
            start = time.time()
            try:
                responses = T.runCondition(task, cond, llm)
                s = T.score(task, responses, cond)
                consecutive_errors = 0
            except Exception as e:
                # Failed cells are NOT written to the results file (so a rerun retries them);
                # they go to <out>.errors, and repeated failures stop the run instead of
                # silently producing hours of empty results.
                consecutive_errors += 1
                print(f"ERROR on {task['id']} {cond}: {e!r}", file=sys.stderr)
                with open(a.out + ".errors", "a") as ef:
                    ef.write(json.dumps({"id": task["id"], "condition": cond, "think": think,
                                         "repeat": rep, "task_seed": a.task_seed, "error": repr(e)}) + "\n")
                if consecutive_errors >= 3:
                    sys.exit("Stopping: 3 errors in a row (see above). Fix the problem and rerun to resume.")
                continue
            usage = {k_: llm.usage[k_] - before[k_] for k_ in before}
            rec = {**s, "task_seed": a.task_seed, "think": think, "repeat": rep,
                   "model": "dry" if a.dry_run else a.model, "temperature": a.temperature,
                   "seconds": round(time.time() - start, 2), "usage": usage,
                   "responses": responses, "error": None}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if n % 10 == 0:
                print(f"[{n}/{len(jobs)}] {time.time() - t0:.0f}s  last={task['id']} {cond} "
                      f"think={think} correct={s['all_correct']}")
    print("done")
 
 
if __name__ == "__main__":
    main()