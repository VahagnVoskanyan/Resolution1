import random
import itertools
from typing import List, Optional, Tuple

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
        """Initialize the clause generator with customizable parameters."""
        self.predicates = predicates or ["pred1", "pred2", "pred3"]#, "Pred4", "Pred5"]
        self.variables = variables or ["X1", "X2", "X3"]#, "U", "V", "W"]
        self.constants = constants or ["const_a", "const_b", "const_c", "const_d", "const_e"]
        self.functions = functions or ["func_f", "func_g", "func_h"]
        self.max_clause_length = max_clause_length
        self.max_term_arity = max_term_arity
        self.max_function_arity = max_function_arity
    
    def generate_term(self, allow_function: bool = True, is_top_level: bool = False) -> str:
        """Generate a random term with careful syntax considerations."""
        if is_top_level:
            # Force variable at top level to avoid type quantification issues
            return random.choice(self.variables)
        
        term_type = random.choice(['variable', 'constant', 'function'] if allow_function else ['variable', 'constant'])
        
        if term_type == 'variable':
            return random.choice(self.variables)
        
        if term_type == 'constant':
            return random.choice(self.constants)
        
        # Function generation
        func = random.choice(self.functions)
        arity = random.randint(1, min(self.max_function_arity, len(self.variables)))
        args = [self.generate_term(allow_function=False) for _ in range(arity)]
        return f"{func}({','.join(args)})"
    
    def generate_literal(self, polarity: Optional[bool] = None) -> str:
        """Generate a random literal with syntax considerations."""
        predicate = random.choice(self.predicates)
        arity = random.randint(1, self.max_term_arity)
        
        # Generate terms, ensuring top-level terms are variables
        terms = [self.generate_term(is_top_level=True) for _ in range(arity)]
        
        # Randomly decide polarity if not specified
        if polarity is None:
            polarity = random.choice([True, False])
        
        literal = f"{predicate}({','.join(terms)})"
        return literal if polarity else f"~{literal}"
    
    def is_tautological_clause(self, clause: List[str]) -> bool:
        """Check if a clause is tautological."""
        for literal in clause:
            negated = literal[1:] if literal.startswith('~') else f"~{literal}"
            if negated in clause:
                return True
        return False
    
    def is_trivial_clause(self, clause: List[str]) -> bool:
        """Check if a clause is trivial."""
        # If all literals are the same, it's trivial
        if len(set(clause)) == 1:
            return True
        
        # A naive check for multiple constants in the same clause
        constants_in_clause = set()
        for literal in clause:
            for c in self.constants:
                if c in literal:
                    constants_in_clause.add(c)
        if len(constants_in_clause) > 1:
            return True
        
        return False
    
    def generate_non_trivial_clause(self, max_attempts: int = 100) -> List[str]:
        """Generate a clause that is neither tautological nor trivial."""
        for _ in range(max_attempts):
            num_literals = random.randint(1, self.max_clause_length)
            clause = [self.generate_literal() for _ in range(num_literals)]
            
            if not (self.is_tautological_clause(clause) or self.is_trivial_clause(clause)):
                return clause
        
        raise RuntimeError("Unable to generate a non-trivial clause after maximum attempts")
    
    def generate_problem_set(
        self, 
        num_clauses: int = 10, 
        output_file: str = 'vampire_problem.p'
    ) -> List[List[str]]:
        """Generate a set of non-trivial clauses and save to TPTP format."""
        # Generate non-trivial clauses
        clauses = []
        while len(clauses) < num_clauses:
            try:
                clause = self.generate_non_trivial_clause()
                clauses.append(clause)
            except RuntimeError:
                # If generation fails repeatedly, skip
                continue
        
        # Prepare TPTP output lines
        tptp_content = []
        
        problem_name = f"resolution_problem_{random.randint(1000, 9999)}"
        tptp_content.append(f"% Problem: {problem_name}")
        
        # ----------------------------------------------------------------------
        # 1) Sort/Type Declaration
        # Note: $i is the built-in type for individuals, so we do NOT re-declare it
        # If you wanted a custom type, you'd do e.g.:
        # tptp_content.append("tff(my_sort, type, myType: $tType).")
        #
        # But for pure TFF with built-in $i, simply skip re-declaration:
        # ----------------------------------------------------------------------
        
        # 2) Predicate type declarations
        # Example: Pred1: $i * $i > $o
        predicate_declarations = []
        for pred in self.predicates:
            arity = random.randint(1, self.max_term_arity)
            # If the predicate is arity n, it's typed $i * $i * ... * $i > $o
            pred_type = f"{pred}: {' * '.join(['$i'] * arity)} > $o"
            predicate_declarations.append(pred_type)
        
        tptp_content.append(
            "tff(predicates, type, " +
            " & ".join(predicate_declarations) +
            ")."
        )
        
        # 3) Function type declarations
        # Example: func_f: $i * $i > $i
        if self.functions:
            function_declarations = []
            for func in self.functions:
                arity = random.randint(1, self.max_function_arity)
                func_type = f"{func}: {' * '.join(['$i'] * arity)} > $i"
                function_declarations.append(func_type)
            
            tptp_content.append(
                "tff(functions, type, " +
                " & ".join(function_declarations) +
                ")."
            )
        
        # 4) Axioms (clauses)
        for i, clause in enumerate(clauses, 1):
            clause_str = " | ".join(clause)
            # In TFF, we can simply write:
            # tff(axiom_i, axiom, ( literal1 | literal2 )).
            tptp_content.append(f"tff(axiom_{i}, axiom, ({clause_str})).")
        
        # 5) Goal (conjecture)
        goal_clause = self.generate_non_trivial_clause()
        goal_str = " | ".join(goal_clause)
        tptp_content.append(f"tff(goal, conjecture, ({goal_str})).")
        
        # Write out with a final newline
        with open(output_file, 'w') as f:
            f.write("\n".join(tptp_content))
            f.write("\n")  # final newline
        
        print(f"Generated problem set saved to {output_file}")
        return clauses

# Example usage
if __name__ == "__main__":
    generator = ResolutionDataGenerator(
        predicates=["p1", "p2", "p3"],
        variables=["X1", "X2", "X3"],
        constants=["const_a", "const_b", "const_c"],
        max_clause_length=3,
        max_term_arity=2
    )
    
    generator.generate_problem_set(num_clauses=15, output_file='vampire_problem.p')
