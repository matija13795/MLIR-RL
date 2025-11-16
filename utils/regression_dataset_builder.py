from dotenv import load_dotenv
load_dotenv()

import argparse, json, glob, sys
from pathlib import Path
from typing import Optional
from utils.config import Config
from rl_autoschedular.state import BenchmarkFeatures, extract_bench_features_from_file
from rl_autoschedular.observation import OpFeatures, ProducerOpFeatures, ActionHistory

# Determines the relationship between OperationFeatures from a given BenchmarkFeature
def determine_consumer_producer(bf: BenchmarkFeature):
    tags = bf.operation_tags
    if len(tags) == 1:
        t = tags[0]
        return bf.operations[t], None

    if len(tags) == 2:
        a, b = tags
        A = bf.operations[a]; B = bf.operations[b]
        a_uses_b = any(p == b for p, _ in A.producers)
        b_uses_a = any(p == a for p, _ in B.producers)
        if a_uses_b and not b_uses_a:
            return A, B   # A is consumer, B is producer
        if b_uses_a and not a_uses_b:
            return B, A   # B is consumer, A is producer


def main():
    ap = argparse.ArgumentParser(description="")
    ap.add_argument("--mlir-dir", type=str, help="Folder containing .mlir files.")
    ap.add_argument("--collected-json", type=str, help="Path to schedule-execution_time data).")
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

    for fpath in mlir_files:
        bench_name = Path(fpath).stem
        if bench_name not in schedules_by_prog: # Doing a join based on collected-json file.
            continue                            # Don't want datapoints we don't have labels for.
        try:
            bf = extract_bench_features_from_file(bench_name, fpath, root_execution_time=0)
        except Exception as e:
            print(f"[SKIP] Failed to extract features for {bench_name}: {e}", file=sys.stderr)
            continue

        # Create producer and consumer features
        consumer_of, producer_of = determine_consumer_producer(bf)
        consumer_encoding = OpFeatures._from_features(consumer_of)
        if producer_of is None:
            producer_encoding = torch.zeros(OpFeatures.size())
        else:
            producer_encoding = OpFeatures._from_features(producer_of)

        # TODO: get schedules embedding
        schedule_encoding = None

        # Pack exactly in the order LSTMEmbedding expects
        obs = torch.cat([consumer_encoding, producer_encoding, schedule_encoding], dim=0).unsqueeze(0)  # shape [1, total]

        # TODO: store obs + labels

if __name__ == "__main__":
    main()