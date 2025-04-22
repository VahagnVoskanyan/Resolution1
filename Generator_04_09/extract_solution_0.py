import os
import glob
import re
import json

def find_first_original_inference_by_role(solved_text):
    """
    Scan all inference(...) annotations bottom‑up and return the first one
    where BOTH clause‑IDs refer to a fof(..., axiom, ...) or fof(..., negated_conjecture, ...).
    """
    # 1) find every inference(...) line
    inf_pattern = (
      r"fof\(f\d+,\w+,\(\s*(.*?)\s*\)\s*,\s*inference\(([^,]+),\s*\[[^\]]*\],\s*\[([^\]]+)\]\)\)"
    )
    matches = list(re.finditer(inf_pattern, solved_text, re.DOTALL))

    # 2) helper to look up a clause's “role” (axiom / negated_conjecture / plain / …)
    def get_role(clause_id):
        m = re.search(rf"fof\({re.escape(clause_id)},\s*([^,]+)\s*,", solved_text)
        return m.group(1).strip() if m else None

    # 3) scan from the bottom (last in the file) upward
    for m in reversed(matches):
        formula    = m.group(1).strip()
        inf_type   = m.group(2).strip()
        clause_ids = [cid.strip() for cid in m.group(3).split(',')]
        roles = [get_role(cid) for cid in clause_ids]

        # accept only if both roles are in our original set
        if all(r in ("axiom", "negated_conjecture") for r in roles):
            return formula, inf_type, clause_ids

    # 4) fallback to the very last inference if none matched
    if matches:
        last = matches[-1]
        formula    = last.group(1).strip()
        inf_type   = last.group(2).strip()
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
        if solved_text[j] == '(':
            depth += 1
        elif solved_text[j] == ')':
            depth -= 1
            if depth == 0:
                close_paren = j
                break
    else:
        return role, None

    # extract everything inside those outer parens
    clause_body = solved_text[open_paren+1:close_paren].strip()
    return role, clause_body

def normalize_clause(clause_text):
    """
    Given something like
       ( ! [X0,X1] : (~defined(X0,X1) | product(X0,X1,compose(X0,X1))) )
    strip the quantifier and outer () so you end up with
       ~defined(X0,X1) | product(X0,X1,compose(X0,X1))
    then split on '|' into a set of literal‐strings.
    """
    txt = clause_text.strip()
    # strip universal quantifier if present
    m = re.match(
        r"^\(\s*!\s*\[[^\]]+\]\s*:\s*\(\s*(.*)\s*\)\s*\)\s*$",
        txt,
        re.DOTALL
    )
    if m:
        body = m.group(1)
    else:
        # otherwise just remove one level of surrounding ()
        if txt.startswith('(') and txt.endswith(')'):
            body = txt[1:-1]
        else:
            body = txt

    # now split into literals
    body = re.sub(r'\s+', '', body)
    literals = set(body.split('|'))
    return literals

def find_matching_cnf_clause(problem_clauses, formula_text):
    norm = normalize_clause(formula_text)
    exact = [c for c in problem_clauses if normalize_clause(c['text']) == norm]
    if exact:
        return exact
    # otherwise score by overlap
    scored = []
    for c in problem_clauses:
        overlap = len(norm & normalize_clause(c['text']))
        if overlap:
            scored.append((overlap, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:3]]

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
        if name_no_ext.endswith("_rs"):
            clean = name_no_ext[:-3]
        else:
            clean = name_no_ext

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
                problem_clauses = []
                for entry in raw_clauses:
                    cid, ctype, lits = entry[0], entry[1], entry[2]
                    problem_clauses.append({
                        "id": cid,
                        "type": ctype,
                        "text": "|".join(lits)
                    })
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

        for cid in clause_ids:
            ctype, cf = extract_clause_text(proof, cid)
            if not cf:
                print(f"\nClause {cid}: could not extract text")
                continue

            print(f"\nClause {cid} ({ctype}):\n  {cf}")
            matches = find_matching_cnf_clause(problem_clauses, cf)
            if matches:
                print("  Matching clauses in problem file:")
                for i, m in enumerate(matches, 1):
                    print(f"   {i}. {m['id']} ({m['type']}): {m['text']}")
            else:
                print("  No matching clauses found in problem file")

if __name__ == "__main__":
    # Configure these paths for your system
    problem_folder = "Res_Pairs"    # Folder containing the .jsonl files
    output_folder = "Output"        # Folder containing the _solved.txt files
    
    process_files(problem_folder, output_folder)