import random
import itertools
from typing import List, Dict, Any, Optional

class RandomGenerators:
    def __init__(
        self,
        predicates: List[str] = None,
        variables: List[str] = None,
        constants: List[str] = None,
        functions: List[str] = None,
        max_clause_length: int = 3,
        max_term_arity: int = 3,
        max_function_arity: int = 1
    ):
        """Initialize the clause generator with customizable parameters."""
        self.predicates = predicates or ["Pred1", "Pred2", "Pred3", "Pred4", "Pred5"]
        self.variables = variables or ["X", "Y", "Z", "U", "V", "W"]
        self.constants = constants or ["const_a", "const_b", "const_c", "const_d", "const_e"]
        self.functions = functions or ["func_f", "func_g", "func_h"]
        self.max_clause_length = max_clause_length
        self.max_term_arity = max_term_arity
        self.max_function_arity = max_function_arity

    # -------------------------------------------------------------------------
    # 1) Core Random Generators
    # -------------------------------------------------------------------------
    def generate_random_term(self) -> str:
        """
        Generate a random term, which can be:
          - A constant, e.g. "const_a"
          - A variable, e.g. "X"
          - A function applied to subterms, e.g. "func_f(X)" or "func_f(const_a)"
        Respects self.max_function_arity.
        """
        choice_type = random.choice(["const", "var", "func"])

        # If generating a constant or variable, just pick from their lists:
        if choice_type == "const":
            return random.choice(self.constants)
        elif choice_type == "var":
            return random.choice(self.variables)

        # Otherwise, generate a function application:
        func_symbol = random.choice(self.functions)
        # We choose a random arity up to self.max_function_arity
        arity = random.randint(1, self.max_function_arity)
        # Generate sub-terms for the function
        sub_terms = [self._generate_subterm() for _ in range(arity)]
        inside = ", ".join(sub_terms)
        return f"{func_symbol}({inside})"

    def _generate_subterm(self) -> str:
        """
        Helper for generating a subterm (which can be just a variable or constant).
        This is separate so that we don't nest function calls too deeply (unless desired).
        If you do want deeper nesting, you can call `generate_random_term` directly.
        """
        # Here, let's decide to allow either a variable or a constant for sub-terms
        return random.choice(self.variables + self.constants)

    def generate_random_predicate_literal(self) -> str:
        """
        Generate a literal using one of the predicates, possibly with multiple arguments.
        The literal may be negated or positive.
        For example:  ¬Pred2(X, const_b)
        """
        # sign
        is_positive = random.choice([True, False])
        # pick a predicate
        predicate = random.choice(self.predicates)
        # pick how many arguments (1..max_term_arity)
        arity = random.randint(1, self.max_term_arity)
        # build arguments
        args = [self.generate_random_term() for _ in range(arity)]
        args_str = ", ".join(args)
        literal = f"{predicate}({args_str})"
        return literal if is_positive else f"¬{literal}"

    def generate_random_clause(self) -> List[str]:
        """
        Generate a clause (list of literal strings) of up to self.max_clause_length literals.
        For example: ["Pred1(const_a)", "¬Pred2(X, Y)"].
        """
        length = random.randint(1, self.max_clause_length)
        clause = [self.generate_random_predicate_literal() for _ in range(length)]
        # Remove duplicates
        clause = list(set(clause))
        return clause
    
if __name__ == "__main__":
    generator = RandomGenerators()
    a = generator.generate_random_clause()

    print(a)
    
    # dataset = generator.create_dataset(n_examples=5, min_clauses=3, max_clauses=5)
    # # Write to a JSONL file
    # with open("toy_gnn_dataset.jsonl", "w") as f:
    #     for item in dataset:
    #         f.write(json.dumps(item) + "\n")

    # # Print out the first sample
    # import pprint
    # pprint.pprint(dataset[0])