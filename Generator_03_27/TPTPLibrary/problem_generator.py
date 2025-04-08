import random

# Define signature components
predicates = ["p", "q", "r"]      # predicate symbols (arity 2 for this example)
functions  = ["f", "g"]           # function symbols (arity 1 for simplicity)
constants  = ["a", "b", "c"]      # constant symbols
variables  = ["X", "Y", "Z", "U", "V"]  # variable symbols

def random_term(depth=0):
    """Generate a random term (constant, variable, or function term)."""
    if depth > 2 or random.random() < 0.4:  # base case: constant or variable
        if random.random() < 0.5:
            return random.choice(constants)
        else:
            return random.choice(variables)
    else:
        func = random.choice(functions)
        subterm = random_term(depth+1)
        return f"{func}({subterm})"

def random_atom():
    """Generate a random atomic formula (including possibly equality)."""
    if random.random() < 0.2:
        # Generate an equality atom
        term1 = random_term()
        term2 = random_term()
        return f"{term1} = {term2}"
    else:
        pred = random.choice(predicates)
        # Use two arguments for binary predicates
        term1 = random_term()
        term2 = random_term()
        return f"{pred}({term1}, {term2})"

def random_formula(depth=0):
    """Recursively build a random formula."""
    if depth > 2:  # limit depth for complexity
        return random_atom()
    # Decide to generate complex formula or atomic
    rand = random.random()
    if rand < 0.3:
        # Unary connective: negation
        return f"~({random_formula(depth+1)})"
    elif rand < 0.6:
        # Binary connective: conjunction or disjunction
        op = "&" if random.random() < 0.5 else "|"
        left = random_formula(depth+1)
        right = random_formula(depth+1)
        return f"({left} {op} {right})"
    elif rand < 0.8:
        # Implication
        left = random_formula(depth+1)
        right = random_formula(depth+1)
        return f"({left} => {right})"
    else:
        # Quantifier (for simplicity, quantify one variable in a subformula)
        var = random.choice(variables)
        subformula = random_formula(depth+1)
        # Ensure the variable appears in the subformula by simple replacement trick
        if var not in subformula:
            subformula = subformula.replace("X", var, 1)  # replace first occurrence of some variable
        return f"![{var}]: ({subformula})"

# Generate a random problem with a few axioms and a conjecture
axioms = []
for i in range(3):  # generate 3 random axioms
    formula = random_formula()
    axioms.append(f"fof(ax_{i+1}, axiom, {formula}).")
# Generate a random conjecture formula
conjecture = f"fof(conjecture, conjecture, {random_formula()})."

# Combine and output the problem
problem_content = "\n".join(axioms + [conjecture])
print(problem_content)
# (In practice, write problem_content to a .p file)
