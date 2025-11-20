from dotenv import load_dotenv
load_dotenv()

import argparse, json, glob, sys, torch, re
from pathlib import Path
from typing import Optional
from utils.config import Config
from rl_autoschedular.state import BenchmarkFeatures, extract_bench_features_from_file
from rl_autoschedular.observation import OpFeatures, ProducerOpFeatures, ActionHistory
from rl_autoschedular.env import Env
from rl_autoschedular.actions import Action, ActionSpace

# Determines the relationship between OperationFeatures from a given BenchmarkFeature
def determine_consumer_producer(bf: BenchmarkFeatures):
    tags = bf.operation_tags
    if len(tags) == 1:
        t = tags[0]
        return bf.operations[t], None
    elif len(tags) == 2:
        a, b = tags
        A = bf.operations[a]; B = bf.operations[b]
        a_uses_b = any(p == b for p, _ in A.producers)
        b_uses_a = any(p == a for p, _ in B.producers)
        if a_uses_b and not b_uses_a:
            return A, B   # A is consumer, B is producer
        if b_uses_a and not a_uses_b:
            return B, A   # B is consumer, A is producer

# NOTE: keep in sync with Execution.{decode_cache_key} if upstream changes.
def split_program_schedule(program_schedule: str) -> list[list[str]]:
    """
    Split a program schedule string into a list of operation schedules.
    Each operation schedule is represented as a list of
    transformation strings like "I(0,1,2)", "T(32,2,0)", "NT()", etc.

    Example:
        "I(0,1,2)T(32,2,0)NT()|I(1,3,0,2)TP(2,64,16,1)NT()"

    becomes:
        [
          ["I(0,1,2)", "T(32,2,0)", "NT()"],
          ["I(1,3,0,2)", "TP(2,64,16,1)", "NT()"]
        ]
    """
    operation_schedules: list[list[str]] = []
    for operation_schedule in program_schedule.split("|"):
        transformation_strings = re.findall(r"\w+\([^\)]*\)", operation_schedule)
        operation_schedules.append(transformation_strings)
    return operation_schedules

# NOTE: keep in sync with Execution.{get_code_cache_key} if upstream changes.
# Example input:   [[Interchange(2, 0, 1, operation_tag = operation_0), TiledParallelization(16, 8, 1, operation_tag = operation_0, iterators = ['reduction', 'parallel', 'parallel']), Tiling(2, 0, 0, operation_tag = operation_0), NoTransformation(operation_tag = operation_0)], [Interchange(0, 1, operation_tag = operation_1), Tiling(4, 4, operation_tag = operation_1), NoTransformation(operation_tag = operation_1)]]
# Example ouutput: I(2,0,1)TP(16,8,1)T(2,0,0)NT()|I(0,1)T(4,4)NT()
def encode_program_schedule(seq: list[list[Action]]) -> str:
    ops_codes = []
    for op_seq in seq:
        ops_codes.append(''.join(map(str, op_seq)))
    return '|'.join(ops_codes)

def build_action_history(bench: BenchmarkFeatures, program_schedule: str) -> torch.Tensor:
    """
    Given a BenchmarkFeatures and a program schedule,
    reconstruct the sequence of Action objects via the Env and ActionSpace,
    and return the ActionHistory tensor exactly as MLIR-RL expects in LSTMEmbedding.
    """
    env = Env()
    operation_schedules = split_program_schedule(program_schedule)
    state = env.reset(bench)

    actions: list[Action] = []                # flat with sub_actions
    seq_actions_rev: list[list[Action]] = []  # per-op, only main actions, reversed order

    for operation_schedule in reversed(operation_schedules):  # Why reversed? ...
        if state is None:
            raise RuntimeError("State unexpectedly None while parsing schedule.")

        op_actions_for_string: list[Action] = []
        for transformation_string in operation_schedule:
            symbol = transformation_string.split('(')[0]
            action_type = ActionSpace.action_type_by_symbol(symbol)

            # Handle param-less actions (like NT(), V()) without parsing params
            if action_type.params_size() == 0:
                global_action = action_type(state)
            else:
                global_action = ActionSpace.action_from_str(state, transformation_string)

            op_actions_for_string.append(global_action)

            # expanded sequence for env + ActionHistory
            for action in global_action.sub_actions + [global_action]:
                actions.append(action)
                state = env.step(state, action)

        seq_actions_rev.append(op_actions_for_string)
        state = env.get_next_op_state(state)

    if state is not None:
        raise RuntimeError("Parsing ended but final state is not None.")

    # ===== ROUND-TRIP TEST (program_schedule -> Actions -> program_schedule) =====
    # Reverse op order back to original
    seq_actions_for_key = list(reversed(seq_actions_rev))
    reconstructed = encode_program_schedule(seq_actions_for_key)
    if reconstructed != program_schedule:
        raise RuntimeError(
            f"Schedule round-trip mismatch:\n"
            f"  original:     {program_schedule}\n"
            f"  reconstructed:{reconstructed}"
        )
    # ===== END ROUND-TRIP TEST =====

    # Finally, build the ActionHistory tensor for the model (LSTMEmbedding expects such a tensor)
    return ActionSpace.action_history(actions)

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

        # For now, we are only interested in 1-op and 2-op programs.
        if len(bf.operation_tags) != 1 and len(bf.operation_tags) != 2:
            continue

        # Create producer and consumer features
        consumer_of, producer_of = determine_consumer_producer(bf)
        consumer_encoding = OpFeatures._from_features(consumer_of)
        if producer_of is None:
            producer_encoding = torch.zeros(OpFeatures.size())
        else:
            producer_encoding = OpFeatures._from_features(producer_of)

        # Build schedule encoding for each schedule in this program
        for program_schedule, exec_time in schedules_by_prog[bench_name].items():
            try:
                schedule_encoding = build_action_history(bf, program_schedule)  # torch.Tensor
            except Exception as e:
                print(
                    f"[SKIP] Failed to parse schedule for {bench_name}: {program_schedule}: "
                    f"{type(e).__name__}: {e!r}",
                    file=sys.stderr,
                )
                continue

            # Pack exactly in the order LSTMEmbedding expects
            obs = torch.cat(
                [consumer_encoding, producer_encoding, schedule_encoding],
                dim=0
            ).unsqueeze(0)

            # TODO: store obs + labels

if __name__ == "__main__":
    main()