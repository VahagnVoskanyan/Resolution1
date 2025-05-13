import os, glob, json, argparse, time
from collections import defaultdict

import torch
from train_model_GNN import (
    EdgeClassifierGNN,
    build_graph_from_example,
    embed_literal, parse_literal,
)
from problem_solver import (
    load_clauses_from_jsonl,
    resolution_prover,
    UnificationResolution,
)

###############################################################################
# 1)  Helper: single-step GNN score for every candidate pair
###############################################################################
def rank_pairs(example, model, p2i, device, max_args=3):
    """Return a list [(score, pair_dict), …] sorted high→low."""
    data = build_graph_from_example(example, p2i, max_args=max_args).to(device)
    with torch.no_grad():
        logits = model(data.x, data.edge_index)[:, 1]
    scores = logits.cpu().tolist()[:-1]
    return sorted(
        zip(scores, example["resolvable_pairs"]), key=lambda x: x[0], reverse=True
    )

###############################################################################
# 2)  Guided prover with max clause size and timeout
###############################################################################
def resolution_prover_guided(
    clauses,
    model,
    p2i,
    device,
    max_args=3,
    max_clause_size=10,
    timeout_seconds=30
):
    """
    Guided resolution prover with:
      - GNN-ranked candidate ordering
      - max_clause_size limit
      - timeout mechanism
    Returns (proved: bool, steps: int).
    """
    ur = UnificationResolution()
    all_clauses = set(frozenset(c) for c in clauses)
    new_clauses = set()
    processed_keys = set()
    step_count = 0

    start_time = time.perf_counter()

    while True:
        # global timeout
        if time.perf_counter() - start_time > timeout_seconds:
            return False, step_count

        # build example
        example = {
            "clauses": [["u", "tmp", list(c)] for c in all_clauses],
            "resolvable_pairs": [],
        }
        for i, c1 in enumerate(sorted(all_clauses)):
            for j, c2 in enumerate(sorted(all_clauses)):
                if i >= j:
                    continue
                for a, litA in enumerate(c1):
                    for b, litB in enumerate(c2):
                        example["resolvable_pairs"].append({
                            "clauseA_index": i,
                            "literalA_index": a,
                            "clauseB_index": j,
                            "literalB_index": b,
                        })

        ranked = rank_pairs(example, model, p2i, device, max_args)

        for _score, pair in ranked:
            # periodic timeout
            if step_count % 100 == 0 and time.perf_counter() - start_time > timeout_seconds:
                return False, step_count

            i, a = pair["clauseA_index"], pair["literalA_index"]
            j, b = pair["clauseB_index"], pair["literalB_index"]
            c1 = sorted(all_clauses)[i]
            c2 = sorted(all_clauses)[j]

            key = (c1, c2, a, b)
            if key in processed_keys:
                continue
            processed_keys.add(key)

            litA = list(c1)[a]
            litB = list(c2)[b]
            unifier = ur.can_resolve(litA, litB)
            if unifier is None:
                continue

            resolvent = {
                ur.apply_subst_to_literal(l, unifier)
                for l in c1 if l != litA
            } | {
                ur.apply_subst_to_literal(l, unifier)
                for l in c2 if l != litB
            }

            step_count += 1

            # proof found
            if not resolvent:
                return True, step_count

            # apply size limit
            if len(resolvent) > max_clause_size:
                continue

            fs = frozenset(resolvent)
            if fs not in all_clauses and fs not in new_clauses:
                new_clauses.add(fs)

        if not new_clauses:
            return False, step_count

        all_clauses |= new_clauses
        new_clauses.clear()

###############################################################################
# 3)  CLI driver
###############################################################################
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", required=True,
                    help="Folder with *.jsonl problems")
    ap.add_argument("--checkpoint", default="Models/gnn_model1.pt")
    ap.add_argument("--predicates", nargs='+',
                    default=["pred1", "pred2", "pred3"])
    ap.add_argument("--max_args", type=int, default=3)
    ap.add_argument("--max_size", type=int, default=10,
                    help="Max resolvent size for both solvers")
    ap.add_argument("--timeout", type=int, default=30,
                    help="Time limit (seconds) per run")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EdgeClassifierGNN(in_dim=5, hidden_dim=64).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    p2i = {p: i+1 for i, p in enumerate(args.predicates)}

    guided_better = 0
    files = sorted(glob.glob(os.path.join(args.problems, "*.jsonl")))

    for fn in files:
        clauses = load_clauses_from_jsonl(fn)
        if not clauses:
            continue

        # brute-force with max_size limit and timeout
        proved_bf, steps_bf, timed_out_bf = resolution_prover(
            clauses,
            max_clause_size=args.max_size,
            timeout_seconds=args.timeout
        )

        # guided with same max_size and timeout
        proved_gn, steps_gn = resolution_prover_guided(
            clauses,
            model,
            p2i,
            device,
            max_args=args.max_args,
            max_clause_size=args.max_size,
            timeout_seconds=args.timeout
        )

        status_bf = "Proved" if proved_bf else "Not proved"
        status_gn = "✓" if proved_gn else "✗"
        if steps_gn < steps_bf:
            guided_better += 1

        print(f"{os.path.basename(fn):40s}  "
              f"brute: {steps_bf:4d} {status_bf:10s}   "
              f"guided: {steps_gn:4d} {status_gn}")

    if files:
        avg_bf = sum(
            resolution_prover(load_clauses_from_jsonl(f),
                              max_clause_size=args.max_size,
                              timeout_seconds=args.timeout)[1]
            for f in files
        ) / len(files)
        avg_gn = sum(
            resolution_prover_guided(load_clauses_from_jsonl(f),
                                     model,
                                     p2i,
                                     device,
                                     max_args=args.max_args,
                                     max_clause_size=args.max_size,
                                     timeout_seconds=args.timeout)[1]
            for f in files
        ) / len(files)
        gain = (avg_bf - avg_gn) / avg_bf * 100 if avg_bf else 0
        pct_better = guided_better / len(files) * 100

        print("\n──────────────── summary ────────────────")
        print(f"files tested: {len(files)}")
        print(f"avg steps – brute-force : {avg_bf:.1f}")
        print(f"avg steps – GNN-guided  : {avg_gn:.1f}")
        print(f"relative reduction      : {gain:.1f} %")
        print(f"guided faster on        : {guided_better}/{len(files)} files ({pct_better:.1f} %)" )

if __name__ == '__main__':
    main()
