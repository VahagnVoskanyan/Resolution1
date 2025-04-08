import os
import glob
import random
import re

# --- Read axioms from files in the Axioms folder ---

def read_axioms(axioms_folder):
    """
    Read all TPTP axiom files from the given folder.
    Supports files with extensions .p and .ax.
    """
    axiom_files = glob.glob(os.path.join(axioms_folder, "*.p")) + \
                  glob.glob(os.path.join(axioms_folder, "*.ax"))
    axioms = []
    for filename in axiom_files:
        with open(filename, 'r') as f:
            content = f.read().strip()
            if content:  # ignore empty files
                axioms.append(content)
    return axioms

def extract_predicates(axiom_text):
    """
    Extracts a set of predicate names from a TPTP axiom text.
    This simple regex looks for a lowercase word followed by an opening parenthesis.
    """
    pattern = r'\b([a-z][a-zA-Z0-9_]*)\s*\('
    matches = re.findall(pattern, axiom_text)
    return list(set(matches))

# --- Functions to generate random TPTP formulas ---

def random_term(depth=0, constants=["a", "b", "c"], variables=["X", "Y", "Z"]):
    """
    Recursively generates a random term.
    Terms can be a constant, variable, or a function applied to a subterm.
    """
    if depth > 2 or random.random() < 0.4:
        return random.choice(constants + variables)
    else:
        func = random.choice(["f", "g", "h"])  # function symbols
        subterm = random_term(depth+1, constants, variables)
        return f"{func}({subterm})"

def random_atom(predicates, depth=0):
    """
    Generate an atomic formula.
    If a list of predicates is available (from the axioms), choose one and assume binary arity.
    Otherwise, fall back to an equality atom.
    """
    if predicates and random.random() < 0.8:
        pred = random.choice(predicates)
        term1 = random_term(depth)
        term2 = random_term(depth)
        return f"{pred}({term1}, {term2})"
    else:
        # fallback: create an equality atom
        term1 = random_term(depth)
        term2 = random_term(depth)
        return f"{term1} = {term2}"

def random_formula(predicates, depth=0):
    """
    Recursively build a random formula.
    Uses negation, binary connectives, implication, and quantification.
    """
    if depth > 2:
        return random_atom(predicates, depth)
    
    choice = random.random()
    if choice < 0.3:
        # Negation
        return f"~({random_formula(predicates, depth+1)})"
    elif choice < 0.6:
        # Conjunction or Disjunction
        op = "&" if random.random() < 0.5 else "|"
        left = random_formula(predicates, depth+1)
        right = random_formula(predicates, depth+1)
        return f"({left} {op} {right})"
    elif choice < 0.8:
        # Implication
        left = random_formula(predicates, depth+1)
        right = random_formula(predicates, depth+1)
        return f"({left} => {right})"
    else:
        # Quantification (universal quantifier here)
        var = random.choice(["X", "Y", "Z", "U", "V"])
        subformula = random_formula(predicates, depth+1)
        # Force the quantifier to appear by a simple substitution
        if var not in subformula:
            subformula = subformula.replace("X", var, 1)
        return f"![{var}]: ({subformula})"

def generate_conjecture(predicates):
    """
    Generate a TPTP conjecture using the random formula generator.
    The resulting conjecture uses a formula that ideally references some predicate(s)
    extracted from the axioms.
    """
    formula = random_formula(predicates)
    return f"fof(conjecture, conjecture, {formula})."

# --- Assemble the full problem file ---

def generate_problem(axioms_folder):
    """
    Generates a complete TPTP problem file:
    - Reads all axioms from the provided folder.
    - Extracts predicate names from the axioms.
    - Generates a new conjecture referencing these predicates.
    - Combines the axioms and conjecture into a single problem file text.
    """
    axioms = read_axioms(axioms_folder)
    if not axioms:
        raise ValueError("No axiom files found in the specified folder.")
    
    all_axioms_text = "\n\n".join(axioms)
    # Extract predicate names from the combined axiom texts.
    predicates = extract_predicates(all_axioms_text)
    print("Extracted predicates:", predicates)
    conjecture = generate_conjecture(predicates)
    problem_text = all_axioms_text + "\n\n" + conjecture
    return problem_text

if __name__ == "__main__":
    axioms_folder = "Axioms"  # Ensure this folder exists and contains your TPTP axiom files.
    try:
        problem = generate_problem(axioms_folder)
        # Save the generated problem to a file.
        with open("generated_problem.p", "w") as f:
            f.write(problem)
        print("Generated TPTP problem:")
        print(problem)
    except Exception as e:
        print("Error:", e)
