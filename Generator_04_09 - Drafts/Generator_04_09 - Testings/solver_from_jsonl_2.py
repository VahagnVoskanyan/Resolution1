import os
import json
import heapq
import itertools

from unification_resolution import UnificationResolution
from solver_helpers_2 import ResolutionDataGenerator

_hd = ResolutionDataGenerator()

def heuristic_score(clauses, i, j, idxA, idxB):
    # Pre‑compute frequencies once per call‑site if you like
    # For simplicity we recompute here (cheap for <100 clauses)
    freq = _hd.compute_predicate_frequencies(clauses)
    return _hd.score_resolution_pair(clauses, i, j, idxA, idxB, freq)

# Load clauses from a JSONL file, ignoring resolvable_pairs and best_pair
def load_clauses_from_jsonl(path):
    with open(path) as f:
        line = f.readline().strip()
        if not line:
            return []
        data = json.loads(line)
    raw = data.get('clauses', [])
    clauses = []
    for entry in raw:
        if isinstance(entry, list) and len(entry) >= 3 and isinstance(entry[2], list):
            lits = entry[2]
        elif isinstance(entry, list):
            lits = entry
        else:
            continue
        #clauses.append(set(lits))
        clauses.append(list(dict.fromkeys(lits)))   # list, duplicates removed
    return clauses


def resolution_prover(clauses, max_steps = 50_000):
    """
    Basic resolution prover: tries all clause pairs, applies unification,
    tracks steps until empty clause or exhaustion.
    Returns (proved: bool, steps: int).
    """
    ur = UnificationResolution()
    #all_clauses = list(map(set, clauses))          # keep orderable list
    all_clauses = [c[:] for c in clauses]                 # keep as lists
    clause_set  = {frozenset(c) for c in all_clauses}     # hashable copy
    
    # (support,SOS) split is optional; here we resolve *any* with *any*
    # -----------------------------------------------------------------
    # 1. Build *once* the heap of all candidate literal pairs
    heap = []
    for i, j in itertools.combinations(range(len(all_clauses)), 2):
        for idxA in range(len(all_clauses[i])):
            for idxB in range(len(all_clauses[j])):
                sc = heuristic_score(all_clauses, i, j, idxA, idxB)
                if sc < float("inf"):
                    heapq.heappush(heap, (sc, i, j, idxA, idxB))

    steps = 0
    while heap and steps < max_steps:
        score, i, j, idxA, idxB = heapq.heappop(heap)

        C1, C2 = all_clauses[i], all_clauses[j]
        litA, litB = list(C1)[idxA], list(C2)[idxB]
        subst = ur.can_resolve(litA, litB)
        if subst is None:
            continue

        # resolvent = {ur.apply_subst_to_literal(l, subst)
        #              for l in C1 if l != litA} | \
        #             {ur.apply_subst_to_literal(l, subst)
        #              for l in C2 if l != litB}
        
         # build as list so it stays indexable
        resolvent = (
            [ur.apply_subst_to_literal(l, subst) for l in C1 if l != litA] +
            [ur.apply_subst_to_literal(l, subst) for l in C2 if l != litB]
        )
        # drop duplicates but keep order
        resolvent = list(dict.fromkeys(resolvent))

        steps += 1
        #if not resolvent:
        if len(resolvent) == 0:
            return True, steps                # ⊥ found

        fs = frozenset(resolvent)
        if fs in clause_set:
            continue

        # 2. Add the new clause and generate its pairs lazily
        new_idx = len(all_clauses)
        all_clauses.append(resolvent)
        clause_set.add(fs)

        for k in range(new_idx):              # against every older clause
            for idxA2 in range(len(all_clauses[new_idx])):
                for idxB2 in range(len(all_clauses[k])):
                    sc = heuristic_score(all_clauses, new_idx, k, idxA2, idxB2)
                    if sc < float("inf"):
                        heapq.heappush(heap, (sc, new_idx, k, idxA2, idxB2))

    return False, steps

    # all_clauses = set(frozenset(c) for c in clauses)
    # new_clauses = set()
    # processed_pairs = set()
    # step_count = 0

    # while True:
    #     for c1 in list(all_clauses):
    #         for c2 in list(all_clauses):
    #             if c1 == c2:
    #                 continue
    #             if repr(c1) >= repr(c2):
    #                 continue
    #             pair_key = (c1, c2)
    #             if pair_key in processed_pairs:
    #                 continue
    #             processed_pairs.add(pair_key)

    #             for litA in c1:
    #                 for litB in c2:
    #                     unifier = ur.can_resolve(litA, litB)
    #                     if unifier is None:
    #                         continue
    #                     # build resolvent
    #                     resolvent = set()
    #                     for l in c1:
    #                         if l != litA:
    #                             resolvent.add(ur.apply_subst_to_literal(l, unifier))
    #                     for l in c2:
    #                         if l != litB:
    #                             resolvent.add(ur.apply_subst_to_literal(l, unifier))
    #                     step_count += 1
    #                     if not resolvent:
    #                         return True, step_count
    #                     res_fs = frozenset(resolvent)
    #                     if res_fs not in all_clauses and res_fs not in new_clauses:
    #                         new_clauses.add(res_fs)
    #     if not new_clauses:
    #         return False, step_count
    #     all_clauses |= new_clauses
    #     new_clauses.clear()


def main():
    jsonl_dir = 'JsonlDataset'
    if not os.path.isdir(jsonl_dir):
        print(f"Directory '{jsonl_dir}' not found.")
        return
    files = [f for f in sorted(os.listdir(jsonl_dir)) if f.endswith('.jsonl')]
    if not files:
        print(f"No .jsonl files in '{jsonl_dir}'.")
        return

    for fn in files:
        path = os.path.join(jsonl_dir, fn)
        clauses = load_clauses_from_jsonl(path)
        if not clauses:
            print(f"{fn}: NO CLAUSES PARSED")
            continue
        proved, steps = resolution_prover(clauses)
        status = 'PROVED' if proved else 'NOT PROVED'
        print(f"{fn}: {status} in {steps} resolution steps.")

if __name__ == '__main__':
    main()
