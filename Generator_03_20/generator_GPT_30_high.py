import random

# Define basic term classes.
class Term:
    pass

class Variable(Term):
    def __init__(self, name):
        self.name = name
    def __str__(self):
        return self.name
    def __repr__(self):
        return self.name

class Function(Term):
    def __init__(self, fun, args):
        self.fun = fun
        self.args = args  # List of Term instances.
    def __str__(self):
        return f"{self.fun}({', '.join(str(arg) for arg in self.args)})"
    def __repr__(self):
        return str(self)

# Random term generator: can produce either a variable or a function term.
def random_term(variables, functions, max_depth=2, depth=0):
    if depth >= max_depth or random.random() < 0.5:
        return Variable(random.choice(variables))
    else:
        fun = random.choice(functions)
        # Choose a random number of arguments (1 or 2 for simplicity)
        num_args = random.choice([1, 2])
        args = [random_term(variables, functions, max_depth, depth+1) for _ in range(num_args)]
        return Function(fun, args)

# Literal class: holds a predicate with a list of term arguments and a sign.
class Literal:
    def __init__(self, predicate, terms, positive=True):
        self.predicate = predicate
        self.terms = terms  # List of Term objects.
        self.positive = positive
    def __str__(self):
        sign = "" if self.positive else "¬"
        return f"{sign}{self.predicate}({', '.join(str(t) for t in self.terms)})"
    def complement(self):
        return Literal(self.predicate, self.terms, not self.positive)

# Occurs check: ensure no cyclic substitutions.
def occurs(var, term):
    if isinstance(term, Variable):
        return term.name == var.name
    elif isinstance(term, Function):
        return any(occurs(var, arg) for arg in term.args)
    return False

# Unification functions.
def unify_terms(t1, t2, subst):
    if subst is None:
        return None
    if isinstance(t1, Variable):
        return unify_var(t1, t2, subst)
    elif isinstance(t2, Variable):
        return unify_var(t2, t1, subst)
    elif isinstance(t1, Function) and isinstance(t2, Function):
        if t1.fun != t2.fun or len(t1.args) != len(t2.args):
            return None
        for arg1, arg2 in zip(t1.args, t2.args):
            subst = unify_terms(arg1, arg2, subst)
            if subst is None:
                return None
        return subst
    else:
        return None

def unify_var(var, term, subst):
    if var.name in subst:
        return unify_terms(subst[var.name], term, subst)
    elif isinstance(term, Variable) and term.name in subst:
        return unify_terms(var, subst[term.name], subst)
    elif occurs(var, term):
        return None
    else:
        subst[var.name] = term
        return subst

def unify_literals(lit1, lit2):
    # Assumes literals have the same predicate and arity.
    if lit1.predicate != lit2.predicate or len(lit1.terms) != len(lit2.terms):
        return None
    subst = {}
    for t1, t2 in zip(lit1.terms, lit2.terms):
        subst = unify_terms(t1, t2, subst)
        if subst is None:
            return None
    return subst

# Generate a random literal given a predicate.
def random_literal(predicate, variables, functions, max_depth=2, positive=True):
    # For simplicity, we generate a literal with one randomly generated term.
    term = random_term(variables, functions, max_depth)
    return Literal(predicate, [term], positive)

# Generate a forced pair of complementary literals that are unifiable.
def generate_forced_literal_pair(predicates, variables, functions, max_depth=2):
    while True:
        predicate = random.choice(predicates)
        lit1 = random_literal(predicate, variables, functions, max_depth, positive=True)
        lit2 = random_literal(predicate, variables, functions, max_depth, positive=False)
        subst = unify_literals(lit1, lit2)
        if subst is not None:
            return lit1, lit2, subst

# Generate a pair of clauses that include the forced complementary literals.
def generate_clause_pair(predicates, variables, functions, extra_literals=1, max_depth=2):
    forced_lit1, forced_lit2, mgu = generate_forced_literal_pair(predicates, variables, functions, max_depth)
    clause1 = [forced_lit1]
    clause2 = [forced_lit2]
    # Optionally add extra random literals to each clause.
    for _ in range(extra_literals):
        clause1.append(random_literal(random.choice(predicates), variables, functions, max_depth, positive=random.choice([True, False])))
        clause2.append(random_literal(random.choice(predicates), variables, functions, max_depth, positive=random.choice([True, False])))
    return clause1, clause2, mgu

# Functions to linearize clauses and substitutions for sequence-to-sequence training.
def linearize_clause(clause):
    return " ∨ ".join(str(lit) for lit in clause)

def linearize_substitution(subst):
    if not subst:
        return "{}"
    return "{" + ", ".join(f"{var}/{str(term)}" for var, term in subst.items()) + "}"

# Generate one training example: (input string, target string)
def generate_training_example(predicates, variables, functions, extra_literals=1, max_depth=2):
    clause1, clause2, mgu = generate_clause_pair(predicates, variables, functions, extra_literals, max_depth)
    input_str = linearize_clause(clause1) + " ; " + linearize_clause(clause2)
    target_str = linearize_substitution(mgu)
    return input_str, target_str

# Generate a dataset of training examples.
def generate_dataset(num_examples, predicates, variables, functions, extra_literals=1, max_depth=2):
    dataset = []
    for _ in range(num_examples):
        example = generate_training_example(predicates, variables, functions, extra_literals, max_depth)
        dataset.append(example)
    return dataset

# Example usage: generate a few training examples.
if __name__ == "__main__":
    predicates = ["P", "Q", "R", "S"]
    variables = ["x", "y", "z", "a", "b", "c"]
    functions = ["f", "g", "h"]
    
    # Generate 10 examples; each example is a tuple (input string, target string).
    dataset = generate_dataset(10, predicates, variables, functions, extra_literals=1, max_depth=2)
    for i, (inp, tgt) in enumerate(dataset, 1):
        print(f"Example {i}:")
        print(f"Input (Clauses): {inp}")
        print(f"Target (MGU): {tgt}\n")

