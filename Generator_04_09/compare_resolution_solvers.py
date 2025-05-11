# benchmark_solver.py
#
# Compare two modes on a folder of *.jsonl problems
#   1. brute‑force resolution order  (your original solver)
#   2. GNN‑guided order              (same search, but pairs are tried
#      in descending score given by the checkpoint)

import os, glob, json, argparse, time
from collections import defaultdict

import torch
from train_model_GNN import (               # ← comes from your unchanged file
    EdgeClassifierGNN,
    build_graph_from_example,
    embed_literal, parse_literal,           # only needed if you tweak below
)
from solver_from_jsonl_1 import (           # ← unchanged baseline prover
    load_clauses_from_jsonl,
    resolution_prover,
    UnificationResolution,                  # imported transitively
)

###############################################################################
# 1)  Helper: single‑step GNN score for every candidate pair
###############################################################################
def rank_pairs(example, model, p2i, device, max_args=3):
    """Return a list [(score, pair_dict), …] sorted high→low."""
    data = build_graph_from_example(example, p2i, max_args=max_args).to(device)
    with torch.no_grad():
        logits = model(data.x, data.edge_index)[:, 1]      # probability “best”
    # ignore the final “reversed” edge that build_graph_from_example appends
    scores = logits.cpu().tolist()[:-1]
    ranked = sorted(
        zip(scores, example["resolvable_pairs"]), key=lambda x: x[0], reverse=True
    )
    return ranked


###############################################################################
# 2)  Guided prover (identical to brute force, pair‑order differs)
###############################################################################
def resolution_prover_guided(clauses, model, p2i, device, max_args=3):
    ur = UnificationResolution()
    all_clauses = set(frozenset(c) for c in clauses)
    new_clauses = set()
    processed_pairs = set()
    step_count = 0

    while True:
        # Build “example” on‑the‑fly for the current clause set
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
                        example["resolvable_pairs"].append(
                            {
                                "clauseA_index": i,
                                "literalA_index": a,
                                "clauseB_index": j,
                                "literalB_index": b,
                            }
                        )

        # Rank candidate pairs with the GNN
        ranked = rank_pairs(example, model, p2i, device, max_args)
        tried = 0
        for _score, pair in ranked:
            i, a = pair["clauseA_index"], pair["literalA_index"]
            j, b = pair["clauseB_index"], pair["literalB_index"]
            c1 = sorted(all_clauses)[i]
            c2 = sorted(all_clauses)[j]

            # Same bookkeeping as the baseline
            pair_key = (c1, c2, a, b)
            if pair_key in processed_pairs:
                continue
            processed_pairs.add(pair_key)

            litA = list(c1)[a]
            litB = list(c2)[b]
            unifier = ur.can_resolve(litA, litB)
            if unifier is None:
                continue

            # build resolvent
            resolvent = set()
            for l in c1:
                if l != litA:
                    resolvent.add(ur.apply_subst_to_literal(l, unifier))
            for l in c2:
                if l != litB:
                    resolvent.add(ur.apply_subst_to_literal(l, unifier))

            step_count += 1
            if not resolvent:
                return True, step_count
            res_fs = frozenset(resolvent)
            if res_fs not in all_clauses and res_fs not in new_clauses:
                new_clauses.add(res_fs)
            tried += 1

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
    ap.add_argument("--predicates", nargs="+",
                    default=["pred1", "pred2", "pred3"])
    ap.add_argument("--max_args", type=int, default=3)
    args = ap.parse_args()

    # ── load model ────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = EdgeClassifierGNN(in_dim=5, hidden_dim=64).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    p2i = {p: i + 1 for i, p in enumerate(args.predicates)}

    # ── run over all problems ─────────────────────────────────────────────────
    rows = []
    guided_better = 0
    for fn in sorted(glob.glob(os.path.join(args.problems, "*.jsonl"))):
        clauses = load_clauses_from_jsonl(fn)                      # baseline helper
        if not clauses:
            continue

        # brute force
        proved_bf, steps_bf = resolution_prover(clauses)           # from solver file

        # guided
        proved_gn, steps_gn = resolution_prover_guided(
            clauses, model, p2i, device, args.max_args
        )

        rows.append((os.path.basename(fn), proved_bf, steps_bf, proved_gn, steps_gn))
        if steps_gn < steps_bf:
            guided_better += 1 
        status_bf = "✓" if proved_bf else "✗"
        status_gn = "✓" if proved_gn else "✗"
        print(f"{fn:40s}  brute: {steps_bf:4d} {status_bf}   guided: {steps_gn:4d} {status_gn}")
        #print(f"{fn:40s}  brute: {steps_bf:4d}  guided: {steps_gn:4d}")

    # ── summary ───────────────────────────────────────────────────────────────
    if rows:
        avg_bf = sum(r[2] for r in rows) / len(rows)
        avg_gn = sum(r[4] for r in rows) / len(rows)
        print("\n──────────────── summary ────────────────")
        print(f"files tested: {len(rows)}")
        print(f"avg steps – brute‑force : {avg_bf:.1f}")
        print(f"avg steps – GNN‑guided  : {avg_gn:.1f}")
        gain = (avg_bf - avg_gn) / avg_bf * 100 if avg_bf else 0
        pct_better = guided_better / len(rows) * 100
        print(f"relative reduction      : {gain:.1f} %")
        print(f"guided faster on        : {guided_better}/{len(rows)} "
              f"files  ({pct_better:.1f} %)")


if __name__ == "__main__":
    main()

# python compare_resolution_solvers.py --problems Dataset/Test_Res_Pairs_3 --checkpoint Models/gnn_model2.pt --max_args 3
# --predicates Pred1 Pred2 Pred3 --max_args 3
