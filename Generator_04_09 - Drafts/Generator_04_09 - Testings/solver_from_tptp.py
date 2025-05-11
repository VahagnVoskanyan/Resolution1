import os
import re
from collections import defaultdict
from unification_resolution import UnificationResolution

# --- Heuristic scoring (unit preference, shorter resolvent, frequency-based) ---
def score_resolvent(c1, c2, resolvent, litA, litB, freq_map):
    if not resolvent:
        return -1000.0
    old_total = len(c1) + len(c2)
    new_size = len(resolvent)
    score = new_size - old_total
    # Unit preference
    if len(c1) == 1 or len(c2) == 1:
        score -= 3.0
    # Frequency-based
    def pred(lit): return lit.split('(')[0].lstrip('~')
    score -= 0.5 * (freq_map.get(pred(litA), 0) + freq_map.get(pred(litB), 0))
    return score


def parse_clauses_from_file(file_path):
    pattern = re.compile(r"cnf\s*\(.*?,.*?,(.*?)\)\.", re.DOTALL)
    with open(file_path) as f:
        txt = f.read()
    txt = '\n'.join([ln for ln in txt.splitlines() if not ln.strip().startswith('%')])
    clauses = []
    for m in pattern.finditer(txt):
        form = m.group(1).strip()
        if form.startswith('(') and form.endswith(')'):
            form = form[1:-1]
        lits = [l.strip() for l in form.split('|') if l.strip()]
        clauses.append(set(lits))
    return clauses


def compute_freq_map(clauses):
    freq = defaultdict(int)
    for cl in clauses:
        for lit in cl:
            freq[lit.split('(')[0].lstrip('~')] += 1
    return freq


def generate_initial_resolvents(clauses, ur, freq_map):
    initial = []  # list of (score, resolvent, step_info)
    n = len(clauses)
    for i in range(n):
        for j in range(i+1, n):
            c1, c2 = clauses[i], clauses[j]
            for litA in c1:
                for litB in c2:
                    unifier = ur.can_resolve(litA, litB)
                    if unifier is None:
                        continue
                    resolvent = set(
                        ur.apply_subst_to_literal(l, unifier)
                        for l in (c1 - {litA}) | (c2 - {litB})
                    )
                    score = score_resolvent(c1, c2, resolvent, litA, litB, freq_map)
                    initial.append((score, resolvent))
    # sort ascending: best (lowest) first
    initial.sort(key=lambda x: x[0])
    return [res for _, res in initial]


def greedy_chain(clauses, seed_resolvent, max_steps, freq_map_init):
    ur = UnificationResolution()
    # start chain with original clauses + seed
    all_clauses = [set(c) for c in clauses]
    all_clauses.append(seed_resolvent)
    freq_map = freq_map_init.copy()
    for lit in seed_resolvent:
        freq_map[lit.split('(')[0].lstrip('~')] += 1
    steps = 1
    # if seed itself is empty
    if not seed_resolvent:
        return True, steps

    while steps < max_steps:
        best = None
        n = len(all_clauses)
        for i in range(n):
            for j in range(i+1, n):
                c1, c2 = all_clauses[i], all_clauses[j]
                for litA in c1:
                    for litB in c2:
                        unifier = ur.can_resolve(litA, litB)
                        if unifier is None:
                            continue
                        resolvent = set(
                            ur.apply_subst_to_literal(l, unifier)
                            for l in (c1 - {litA}) | (c2 - {litB})
                        )
                        sc = score_resolvent(c1, c2, resolvent, litA, litB, freq_map)
                        if best is None or sc < best[0]:
                            best = (sc, resolvent)
        if best is None:
            return False, steps
        _, resolvent = best
        steps += 1
        if not resolvent:
            return True, steps
        all_clauses.append(resolvent)
        for lit in resolvent:
            freq_map[lit.split('(')[0].lstrip('~')] += 1
    return False, steps


def solve_file(path, max_steps=20):
    clauses = parse_clauses_from_file(path)
    if not clauses:
        return False, 0
    ur = UnificationResolution()
    freq_map = compute_freq_map(clauses)
    # generate all initial seeds
    seeds = generate_initial_resolvents(clauses, ur, freq_map)
    # try each seed until proof
    for seed in seeds:
        proved, steps = greedy_chain(clauses, seed, max_steps, freq_map)
        if proved:
            return True, steps
    # none proved
    return False, max_steps


def main():
    dataset = 'Dataset'
    max_steps=50
    if not os.path.isdir(dataset):
        print(f"Directory '{dataset}' not found.")
        return
    for f in sorted(os.listdir(dataset)):
        if not f.endswith('.p'):
            continue
        path = os.path.join(dataset, f)
        proved, steps = solve_file(path, max_steps)
        status = 'PROVED' if proved else 'NOT PROVED'
        print(f"{f}: {status} in {steps} steps.")
        #print(f"{f}: {status} in {steps} steps (with fallback chains, max {max_steps}).")

if __name__ == '__main__':
    main()
