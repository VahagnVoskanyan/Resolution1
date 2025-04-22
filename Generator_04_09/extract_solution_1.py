import json, re, pathlib

# ---------------------------------------------------------------------
# 1.  ─── PARSING HELPERS ─────────────────────────────────────────────
# Match any inference whose name ends with “resolution”
resolution_line = re.compile(r'inference\(\s*[A-Za-z_]*resolution\s*,\s*\[\s*\]\s*,\s*\[([^\]]+)\]\)')
fof_clause_block  = re.compile(r'fof\((\w+),[^,]*,\(\s*(.*?)\s*\),', re.DOTALL)
quantified_clause = re.compile(r'!\s*\[.*?\]\s*:\s*\((.*)\)', re.DOTALL)

# def first_resolution_ids(proof_txt: str) -> tuple[str, str]:
#     """IDs of the two parent clauses in the *first* resolution step
#     (printed last in Vampire’s proof)."""
#     last = None
#     for line in proof_txt.splitlines():
#         m = resolution_line.search(line)
#         if m:
#             last = m
#     if not last:
#         raise ValueError("No resolution step found.")
#     return tuple(x.strip() for x in last.group(1).split(','))

def find_fof_blocks(proof: str):
    """
    Yield (clause_id, formula_text) for each 'fof(...)' in the proof.
    We scan for 'fof(' then parse balanced parentheses for the formula.
    """
    for m in re.finditer(r'fof\(\s*(\w+)\s*,', proof):
        cid = m.group(1)
        # skip to the '(' that opens the formula
        idx = proof.find('(', m.end())
        depth = 0
        start = None
        for i in range(idx, len(proof)):
            if proof[i] == '(':
                depth += 1
                if start is None:
                    start = i
            elif proof[i] == ')':
                depth -= 1
                if depth == 0 and start is not None:
                    body = proof[start+1:i].strip()
                    yield cid, body
                    break

def clause_literals_map(proof: str) -> dict[str, list[str]]:
    """
    Build cid → [literal1, literal2, …] by:
      • removing one layer of outer parens,
      • stripping a leading '![…]:(…)' quantifier if present,
      • splitting on top‑level '|'.
    """
    mp = {}
    quantifier = re.compile(r'!\s*\[.*?\]\s*:\s*\((.*)\)', re.DOTALL)
    for cid, body in find_fof_blocks(proof):
        # strip one layer of parens if they wrap the whole body
        if body.startswith('(') and body.endswith(')'):
            body = body[1:-1].strip()
        # strip leading ∀‑quantifier
        qm = quantifier.match(body)
        if qm:
            body = qm.group(1).strip()
        # now it's a disjunction of literals
        mp[cid] = [lit.strip() for lit in body.split('|')]
    return mp

# ─── 2. PICK THE RESOLVED PAIR ────────────────────────────────────────
def complementary_idx(L1: list[str], L2: list[str]) -> tuple[int,int]:
    """
    Return (i,j) where L1[i] and L2[j]:
     • share the same functor name and arity,
     • have opposite signs (~ vs no ~),
    falling back to (0,0) if nothing else matches.
    """
    functor = re.compile(r'^~?([A-Za-z]\w*)\(')
    def core(lit):
        return lit.lstrip('~').replace(' ', '')

    # first try exact or sign‑flip match (strong signal)
    for i, a in enumerate(L1):
        for j, b in enumerate(L2):
            if core(a) == core(b) and (a.startswith('~') ^ b.startswith('~')):
                return i, j

    # next try functor+arity
    def parse_fa(lit):
        m = functor.match(lit)
        if not m: return None, None
        name = m.group(1)
        args = lit[lit.find('(')+1 : lit.rfind(')')]
        arity = args.count(',') + 1 if args else 1
        return name, arity

    for i, a in enumerate(L1):
        name1, ar1 = parse_fa(a)
        if not name1: continue
        for j, b in enumerate(L2):
            name2, ar2 = parse_fa(b)
            if name1 == name2 and ar1 == ar2:
                return i, j

    # fallback: pick the first literal of each
    return 0, 0

# ─── 3. DATASET UPDATE ────────────────────────────────────────────────
def locate_literal_in_dataset(ds: dict, lit: str) -> tuple[int,int]:
    norm = lambda s: s.replace(' ','')
    for ci, (_, _, lits) in enumerate(ds['clauses']):
        for li, candidate in enumerate(lits):
            if norm(candidate) == norm(lit):
                return ci, li
    raise LookupError(f"Literal {lit!r} not found in dataset.")

def inject_best_pair(json_path: pathlib.Path, proof_path: pathlib.Path):
    ds    = json.loads(json_path.read_text())
    proof = proof_path.read_text()

    # find the resolution inference line
    # and extract its two parent IDs
    m = resolution_line.search(proof)
    if not m:
        raise ValueError("No plain resolution step in proof.")
    p1_id, p2_id = [x.strip() for x in m.group(1).split(',')]

    cmap = clause_literals_map(proof)
    L1, L2 = cmap[p1_id], cmap[p2_id]
    i1, i2  = complementary_idx(L1, L2)

    lit1, lit2   = L1[i1], L2[i2]
    c1, l1       = locate_literal_in_dataset(ds, lit1)
    c2, l2       = locate_literal_in_dataset(ds, lit2)

    ds['best_pair'] = {
      "clauseA_index":  c1,
      "literalA_index": l1,
      "clauseB_index":  c2,
      "literalB_index": l2
    }
    json_path.write_text(json.dumps(ds, ensure_ascii=False, indent=2))
    print(f"✓ {json_path.name}  ←  best_pair = ({c1},{l1}) × ({c2},{l2})")

# 4. Batch driver as before...
if __name__ == "__main__":
    res_pairs_dir = pathlib.Path("Res_Pairs")
    output_dir    = pathlib.Path("Output")

    for json_file in res_pairs_dir.glob("*.jsonl"):
        stem       = json_file.stem.replace("gen_problem_rs_", "gen_problem_")
        proof_file = output_dir / f"{stem}_solved.txt"
        if proof_file.exists():
            inject_best_pair(json_file, proof_file)
        else:
            print(f"⚠ No proof for {json_file.name}")