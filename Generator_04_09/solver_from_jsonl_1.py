import os
import json
from unification_resolution import UnificationResolution

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
        clauses.append(set(lits))
    return clauses


def resolution_prover(clauses):
    """
    Basic resolution prover: tries all clause pairs, applies unification,
    tracks steps until empty clause or exhaustion.
    Returns (proved: bool, steps: int).
    """
    ur = UnificationResolution()
    all_clauses = set(frozenset(c) for c in clauses)
    new_clauses = set()
    processed_pairs = set()
    step_count = 0

    while True:
        for c1 in list(all_clauses):
            for c2 in list(all_clauses):
                if c1 == c2:
                    continue
                if repr(c1) >= repr(c2):
                    continue
                pair_key = (c1, c2)
                if pair_key in processed_pairs:
                    continue
                processed_pairs.add(pair_key)

                for litA in c1:
                    for litB in c2:
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
        if not new_clauses:
            return False, step_count
        all_clauses |= new_clauses
        new_clauses.clear()


def main():
    jsonl_dir = 'Dataset/Test_Res_Pairs_2'
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
