import os
import glob
import re
import json

# -----------------------------------------------------------------------------
#  extract_solution_v2.py  —  updated version of the user’s original script
# -----------------------------------------------------------------------------
#  Changes in this file (all original comments kept):
#   * new helper `vamp_id_to_index()`
#   * per‑clause loop inside `process_files()` rewritten to
#       – try fast numeric index lookup first, then
#       – fall back to slow text matching only if necessary.
#   * nothing else removed; the heavyweight helpers remain for robustness /
#     debugging.
# -----------------------------------------------------------------------------


def vamp_id_to_index(cid: str) -> int:
    """Return 0‑based guess for the position of an original clause.

    Vampire numbers all input clauses (both fof/ and their clausified cnf/u‑variants)
    with a *1‑based* integer suffix:
        f1, f2, … / u1, u2, …
    In most generated problems that integer equals the array index + 1 in the
    JSONL file.  So   f19  →  18.
    This is only a *hint*; we still verify the clause text later.
    """
    return int(re.search(r"\d+", cid)[0]) - 1


# -----------------------------------------------------------------------------
#  Helpers copied verbatim from the original script
# -----------------------------------------------------------------------------

def find_first_original_inference_by_role(solved_text):
    """
    Scan all inference(...) annotations bottom‑up and return the first one
    where BOTH clause‑IDs refer to a fof(..., axiom, ...) or
    fof(..., negated_conjecture, ...).
    """
    # 1) find every inference(...) line
    inf_pattern = (
        r"fof\(f\d+,\w+,\(\s*(.*?)\s*\)\s*,\s*inference\(([^,]+),\s*\[[^\]]*\],\s*\[([^\]]+)\]\)\)"
    )
    matches = list(re.finditer(inf_pattern, solved_text, re.DOTALL))

    # 2) helper to look up a clause's “role” (axiom / negated_conjecture / ...)
    def get_role(clause_id):
        m = re.search(rf"fof\({re.escape(clause_id)},\s*([^,]+)\s*,", solved_text)
        return m.group(1).strip() if m else None

    # 3) scan from the bottom (last in the file) upward
    for m in reversed(matches):
        formula = m.group(1).strip()
        inf_type = m.group(2).strip()
        clause_ids = [cid.strip() for cid in m.group(3).split(',')]
        roles = [get_role(cid) for cid in clause_ids]

        # accept only if both roles are in our original set
        if all(r in ("axiom", "negated_conjecture") for r in roles):
            return formula, inf_type, clause_ids

    # 4) fallback to the very last inference if none matched
    if matches:
        last = matches[-1]
        formula = last.group(1).strip()
        inf_type = last.group(2).strip()
        clause_ids = [cid.strip() for cid in last.group(3).split(',')]
        return formula, inf_type, clause_ids

    return None, None, []


def extract_clause_text(solved_text, clause_id):
    """
    Find the full clause text (with nested parens) for fof(clause_id, ...)
    Returns: (clause_role, clause_body)
    """
    header = f"fof({clause_id},"
    idx = solved_text.find(header)
    if idx == -1:
        return None, None

    # move to just past the clause_id comma
    idx += len(header)
    # find the comma after the role name
    comma_after_role = solved_text.find(',', idx)
    role = solved_text[idx:comma_after_role].strip()

    # now find the '(' that opens the clause‑term
    open_paren = solved_text.find('(', comma_after_role + 1)
    if open_paren == -1:
        return role, None

    # walk forward to find matching closing paren
    depth = 0
    for j in range(open_paren, len(solved_text)):
        if solved_text[j] == '(':  # opening paren
            depth += 1
        elif solved_text[j] == ')':
            depth -= 1
            if depth == 0:
                close_paren = j
                break
    else:
        return role, None

    # extract everything inside those outer parens
    clause_body = solved_text[open_paren + 1: close_paren].strip()
    return role, clause_body


def normalize_clause(clause_text):
    """
    Strip outer quantifiers/parens and return a *set* of literals.
    Example             →  ~p(X)|q(Y)
    (order & whitespace are irrelevant for equality testing)
    """
    txt = clause_text.strip()

    # strip universal quantifier if present
    m = re.match(r"^\(\s*!\s*\[[^\]]+\]\s*:\s*\(\s*(.*)\s*\)\s*\)\s*$", txt, re.DOTALL)
    if m:
        body = m.group(1)
    else:
        body = txt[1:-1] if txt.startswith('(') and txt.endswith(')') else txt

    body = re.sub(r"\s+", "", body)  # drop all whitespace
    return set(body.split('|'))


def find_matching_cnf_clause(problem_clauses, formula_text):
    """Slow O(n) scan that returns the first clause with identical literal set."""
    norm = normalize_clause(formula_text)

    # exact match
    for c in problem_clauses:
        if normalize_clause(c["text"]) == norm:
            return c
    return None


# -----------------------------------------------------------------------------
#  Main driver (problem JSONL ↔ proof log)
# -----------------------------------------------------------------------------

def process_files(problem_folder, output_folder):
    # only jsonl files ending with '_rs.jsonl'
    problem_files = glob.glob(os.path.join(problem_folder, "*_rs.jsonl"))
    if not problem_files:
        print(f"No problem files found in {problem_folder}")
        return

    for problem_file in problem_files:
        base = os.path.basename(problem_file)
        name_no_ext = os.path.splitext(base)[0]
        # strip trailing '_rs'
        clean = name_no_ext[:-3] if name_no_ext.endswith("_rs") else name_no_ext

        solved_file = os.path.join(output_folder, clean + "_solved.txt")
        if not os.path.exists(solved_file):
            print(f"Solution file not found for {base}: {solved_file}")
            continue

        print(f"\n==== Processing: {base} ====")
        print(f"Problem file: {problem_file}")
        print(f"Solution file: {solved_file}")

        # --- load clauses from the first JSON line ---
        with open(problem_file, 'r', encoding='utf-8') as f:
            try:
                data = json.loads(f.readline())
                raw_clauses = data.get("clauses", [])
                problem_clauses = [
                    {"id": entry[0], "type": entry[1], "text": "|".join(entry[2])}
                    for entry in raw_clauses
                ]
            except Exception as e:
                print(f"  ✗ Failed to parse JSON in {base}: {e}")
                continue

        print(f"Found {len(problem_clauses)} clauses in problem file")

        # --- load the Vampire proof log ---
        with open(solved_file, 'r', encoding='utf-8') as f:
            proof = f.read()

        formula, inf_type, clause_ids = find_first_original_inference_by_role(proof)
        if not inf_type:
            print(f"  ✗ No inference found in {solved_file}")
            continue

        print(f"\nLast inference ({inf_type}): {formula}")
        print(f"Using clauses: {', '.join(clause_ids)}")

        # ------------------------------------------------------------------
        #  NEW: robust mapping from Vampire clause IDs to JSONL entries
        # ------------------------------------------------------------------
        for cid in clause_ids:
            # 1) always extract the proof‑side clause text for verification
            _role, proof_txt = extract_clause_text(proof, cid)
            if proof_txt is None:
                print(f"\nClause {cid}: could not extract text from proof")
                continue

            guessed_idx = vamp_id_to_index(cid)
            match = None

            # fast path: numeric index & text agree
            if 0 <= guessed_idx < len(problem_clauses):
                stored = problem_clauses[guessed_idx]
                if normalize_clause(proof_txt) == normalize_clause(stored["text"]):
                    match = stored

            # slow path if necessary
            if match is None:
                match = find_matching_cnf_clause(problem_clauses, proof_txt)

            if match:
                idx = problem_clauses.index(match)
                print(
                    f"\nClause {cid}  →  JSONL index {idx}\n"
                    f"  id/type : {match['id']} ({match['type']})\n"
                    f"  text    : {match['text']}"
                )
            else:
                print(f"\nClause {cid}: no matching clause found in JSONL")


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Configure these paths for your system
    problem_folder = "Res_Pairs"    # Folder containing the .jsonl files
    output_folder = "Output"        # Folder containing the _solved.txt files

    process_files(problem_folder, output_folder)
