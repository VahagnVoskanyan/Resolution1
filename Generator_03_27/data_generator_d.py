from typing import List, Tuple, Union
import random
import random
from typing import List, Tuple, Dict, Set

class ResolutionDataGenerator:
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
        """Initialize the clause generator with customizable parameters.
        
        Args:
            predicates: List of predicate names to use
            variables: List of variable names to use
            constants: List of constant names to use
            functions: List of function symbols to use.
            max_clause_length: Maximum number of literals in a clause
            max_term_arity: Maximum number of terms in a predicate
            max_function_arity: Maximum number of arguments in a function.
        """
        self.predicates = predicates or ["P", "Q", "R", "S", "T"]
        self.variables = variables or ["x", "y", "z", "u", "v", "w"]
        self.constants = constants or ["a", "b", "c", "d", "e"]
        self.functions = functions or ["f", "g", "h"]
        self.max_clause_length = max_clause_length
        self.max_term_arity = max_term_arity
        self.max_function_arity = max_function_arity
    
    # Constants: ("a",), Variables: ("x",), Functions: ("f", ("x",))
    def generate_term(self, depth: int = 0) -> Tuple:
        """
        Generate a random term that can be a constant, variable, or function.
        
        Args:
            depth (int): Track recursion depth to prevent excessive nesting
        """
        # Prevent deep recursion and excessive function nesting
        if depth >= self.max_function_arity:
            return random.choice(self.constants + self.variables)
        
        # Randomly choose term type
        term_type = random.choice(['constant', 'variable', 'function'])
        
        if term_type == 'constant':
            return (random.choice(self.constants),)
        
        if term_type == 'variable':
            return (random.choice(self.variables),)
        
        # Function generation
        func = random.choice(self.functions)
        
        # Generate function arguments respecting max_function_arity
        args = tuple(self.generate_term(depth + 1) for _ in range(random.randint(1, self.max_function_arity)))
        
        return (func,) + args

    def generate_literal(self, allow_negation: bool = True) -> Tuple[str, Tuple, bool]:
        """
        Generate a random literal with support for functions.
        
        Args:
            allow_negation (bool): Whether to allow negated literals
        
        Returns:
            A tuple containing:
            - Predicate name (str)
            - Arguments (tuple of terms)
            - Negation flag (bool)
        """
        predicate = random.choice(self.predicates)
        
        # Determine arity of the predicate
        arity = random.randint(1, min(self.max_term_arity, 3))
        
        # Generate arguments with support for functions
        args = tuple(self.generate_term() for _ in range(arity))
        
        # Determine if the literal is negated
        is_negated = allow_negation and random.random() < 0.5
        
        return (predicate, args, is_negated)
    
    def generate_clause(self) -> List[Tuple[str, Tuple, bool]]:
        """
        Generates a clause (list of literals).
        Returns:
            List of literals, where each literal is `(predicate, args, is_negated)`.
        """
        num_literals = random.randint(1, self.max_clause_length)

        return [self.generate_literal() for _ in range(num_literals)]
    
    def clause_to_str(self, clause):
        literals = []
        for pred, args, neg in clause:
            arg_str = ",".join("".join(a) if isinstance(a, tuple) else a for a in args)
            lit_str = f"{'¬' if neg else ''}{pred}({arg_str})"
            literals.append(lit_str)
        return " ∨ ".join(literals)
    
    def is_valid_clause(self, clause):
    # Check for duplicate literals
        unique_lits = set()
        for pred, args, neg in clause:
            lit = (pred, args, neg)
            if lit in unique_lits:
                return False
            unique_lits.add(lit)
        # Check for tautologies
        for pred, args, neg in clause:
            if (pred, args, not neg) in unique_lits:
                return False
        return True

    def generate_valid_clause(self, max_attempts=10):
        for _ in range(max_attempts):
            clause = self.generate_clause()
            if self.is_valid_clause(clause):
                return clause
        return None  # Fallback if no valid clause found
    
    def generate_complementary_pair(self) -> List[Tuple[str, Tuple, bool]]:
        """Returns a clause with a literal and its negation."""
        pred = random.choice(self.predicates)
        args = tuple(self.generate_term() for _ in range(random.randint(1, self.max_term_arity)))
        return [(pred, args, False), (pred, args, True)]  # Single clause with 2 literals
    
    def generate_clause_set(self, size=5, ensure_contradiction=True):
        """
        Generate a set of clauses that may contain complementary pairs.
        If ensure_contradiction=True, force at least one complementary pair.
        """
        clauses = []
        if ensure_contradiction:
            clauses.append(self.generate_complementary_pair())  # Add as a single clause
        for _ in range(size - len(clauses)):
            clause = self.generate_valid_clause()
            if clause:
                clauses.append(clause)
        return clauses
    
if __name__ == "__main__":
    generator = ResolutionDataGenerator()
    clause = generator.generate_clause()
    print(clause)
    # Might output something like:
    # [('P', ('x',), False), ('Q', ('a', 'b'), True)]
    
    # generator = ResolutionDataGenerator()
    # clause_set = generator.generate_clause_set(size=5, ensure_contradiction=True)
    # print([generator.clause_to_str(c) for c in clause_set])
    # Output: ["P(x) ∨ ¬Q(y)", "¬P(a)", "Q(y) ∨ R(z)", ...]