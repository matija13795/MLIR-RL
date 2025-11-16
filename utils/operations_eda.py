from dotenv import load_dotenv
load_dotenv()

import argparse, json, os, sys, glob
from statistics import mean, median
from collections import Counter, defaultdict
from pathlib import Path
from utils.config import Config
from rl_autoschedular.state import extract_bench_features_from_file


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Analyze MLIR benchmarks referenced in collected.json: "
            "counts number of operations per program and summarizes how many "
            "schedule-time measurements each program has."
        )
    )
    ap.add_argument("--mlir-dir", type=str, help="Folder containing .mlir files.")
    ap.add_argument("--collected-json", type=str, help=(
            "Path to collected.json (containing schedule-execution time data) to intersect with .mlir files."
        )
    )
    args = ap.parse_args()

    mlir_dir = Path(args.mlir_dir)
    if not mlir_dir.is_dir():
        print(f"ERROR: --mlir-dir not found: {mlir_dir}", file=sys.stderr)
        sys.exit(1)
    mlir_files = glob.glob(str(mlir_dir / "*.mlir"))  # Find all files ending in .mlir inside the directory mlir_dir.
    if not mlir_files:
        print(f"No .mlir files found under {mlir_dir}")
        return

    collected_path = Path(args.collected_json)
    if not collected_path.is_file():
        print(f"ERROR: collected.json not found at: {collected_path}", file=sys.stderr)
        sys.exit(1)
    with open(collected_path) as f:
        schedules_by_prog = json.load(f)

    total = 0
    op_count_hist = Counter()
    schedules_count_by_ops = defaultdict(list)

    for fpath in mlir_files:
        bench_name = Path(fpath).stem
        if bench_name not in schedules_by_prog: # Doing a join based on collected-json file.
            continue                            # We don't want stats for datapoints we don't have labels for.
        try:
            bf = extract_bench_features_from_file(bench_name, fpath, root_execution_time=0)
        except Exception as e:
            print(f"[SKIP] Failed to extract features for {bench_name}: {e}", file=sys.stderr)
            continue
        total += 1
        n_ops = len(bf.operation_tags)
        op_count_hist[n_ops] += 1
        schedules_count_by_ops[n_ops].append(len(schedules_by_prog[bench_name]))

    if total == 0:
        print("No programs matched (check --mlir-dir and/or --collected-json path/IDs).")
        return

    def pct(x):
        return f"{100.0 * x / total:.1f}%"
    print(f"\nTotal programs considered: {total}")
    print("Programs grouped by number of operations (k):")
    print("  k  |    count   (%)   |  schedules per program (median / mean / p90)")
    print("-----+------------------+---------------------------------------------")
    for k in sorted(op_count_hist):
        count = op_count_hist[k]
        arr = schedules_count_by_ops.get(k, [])
        p90 = sorted(arr)[int(0.9 * len(arr)) - 1]
        print(f"  {k:>2} | {count:>6}  ({pct(count):>6}) |         {int(median(arr)):>5} / {mean(arr):>6.1f} /  {p90}")

if __name__ == "__main__":
    main()