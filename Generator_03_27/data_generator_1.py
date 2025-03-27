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
    
    def generate_term(self, depth: int = 0) -> Union[str, Tuple[str, Tuple]]:
        """
        Generate a random term that can be a constant, variable, or function.
        
        Args:
            depth (int): Track recursion depth to prevent excessive nesting
        
        Returns:
            A term which can be:
            - A constant (str)
            - A variable (str)
            - A function with arguments respecting max_function_arity
        """
        # Prevent deep recursion and excessive function nesting
        if depth >= self.max_function_arity:
            return random.choice(self.constants + self.variables)
        
        # Randomly choose term type
        term_type = random.choice(['constant', 'variable', 'function'])
        
        if term_type == 'constant':
            return random.choice(self.constants)
        
        if term_type == 'variable':
            return random.choice(self.variables)
        
        # Function generation
        func = random.choice(self.functions)
        
        # Generate function arguments respecting max_function_arity
        args = tuple(
            self.generate_term(depth + 1) 
            for _ in range(random.randint(1, min(self.max_function_arity, 3)))
        )
        
        return (func, args)

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
        Generate a random clause (a set of literals).
        
        Returns:
            A list of literals
        """
        # Determine number of literals in the clause
        num_literals = random.randint(1, self.max_clause_length)
        
        return [self.generate_literal() for _ in range(num_literals)]

class ResolutionProblemGenerator:
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
        """
        Initialize the resolution problem generator.
        
        Args:
            See ResolutionDataGenerator for parameter descriptions
        """
        self.data_generator = ResolutionDataGenerator(
            predicates, variables, constants, functions,
            max_clause_length, max_term_arity, max_function_arity
        )

    def generate_resolution_problem(
        self, 
        num_clauses: int = 5, 
        resolution_probability: float = 0.5
    ) -> Dict:
        """
        Generate a resolution problem with resolvable clauses.
        
        Args:
            num_clauses (int): Number of clauses to generate
            resolution_probability (float): Probability of generating resolvable clauses
        
        Returns:
            Dict containing:
            - clauses: List of generated clauses
            - resolvable_pairs: List of resolvable literal pairs
            - substitutions: Potential variable substitutions
        """
        # Generate initial clauses
        clauses = [self.data_generator.generate_clause() for _ in range(num_clauses)]
        
        # Find resolvable literals
        resolvable_pairs = self._find_resolvable_literals(clauses)
        
        # Optional: Adjust clauses to increase resolvability
        if random.random() < resolution_probability:
            self._enhance_resolvability(clauses, resolvable_pairs)
        
        return {
            'clauses': clauses,
            'resolvable_pairs': resolvable_pairs,
            'substitutions': self._generate_substitutions(clauses)
        }

    def _find_resolvable_literals(self, clauses: List[List[Tuple]]) -> List[Tuple[int, int, int, int]]:
        """
        Find resolvable literal pairs across clauses.
        
        Args:
            clauses (List[List[Tuple]]): List of clauses
        
        Returns:
            List of tuples (clause_idx1, literal_idx1, clause_idx2, literal_idx2)
            representing resolvable literal pairs
        """
        resolvable_pairs = []
        
        for i in range(len(clauses)):
            for j in range(i + 1, len(clauses)):
                for k, lit1 in enumerate(clauses[i]):
                    for l, lit2 in enumerate(clauses[j]):
                        if self._can_resolve(lit1, lit2):
                            resolvable_pairs.append((i, k, j, l))
        
        return resolvable_pairs

    def _can_resolve(self, lit1: Tuple, lit2: Tuple) -> bool:
        """
        Check if two literals can be resolved.
        
        Args:
            lit1 (Tuple): First literal (predicate, args, is_negated)
            lit2 (Tuple): Second literal (predicate, args, is_negated)
        
        Returns:
            bool: Whether literals can be resolved
        """
        # Check if predicates are the same but negation is opposite
        if lit1[0] == lit2[0] and lit1[2] != lit2[2]:
            # Check if variable substitution is possible
            return self._check_substitution_possibility(lit1[1], lit2[1])
        return False

    def _check_substitution_possibility(self, args1: Tuple, args2: Tuple) -> bool:
        """
        Check if variable substitution is possible between two sets of arguments.
        
        Args:
            args1 (Tuple): Arguments of first literal
            args2 (Tuple): Arguments of second literal
        
        Returns:
            bool: Whether substitution is possible
        """
        # Simple check: same number of arguments and potential for variable mapping
        if len(args1) != len(args2):
            return False
        
        # Track potential substitutions
        substitutions = {}
        
        for a1, a2 in zip(args1, args2):
            # If either argument is a function, do a deep comparison
            if isinstance(a1, tuple) or isinstance(a2, tuple):
                if not self._deep_compare_terms(a1, a2):
                    return False
            
            # Check variable substitution possibilities
            if isinstance(a1, str) and isinstance(a2, str):
                # If both are variables or constants
                if a1 in self.data_generator.variables and a2 in self.data_generator.variables:
                    continue
                if a1 in self.data_generator.constants and a2 in self.data_generator.constants:
                    if a1 != a2:
                        return False
        
        return True

    def _deep_compare_terms(self, term1, term2) -> bool:
        """
        Recursively compare complex terms (functions).
        
        Args:
            term1: First term to compare
            term2: Second term to compare
        
        Returns:
            bool: Whether terms can be unified
        """
        # If one is a function and other is not, they can't be the same
        if isinstance(term1, tuple) != isinstance(term2, tuple):
            return False
        
        # If both are simple terms (variables/constants)
        if not isinstance(term1, tuple):
            return term1 in self.data_generator.variables or term1 == term2
        
        # Compare functions
        if term1[0] != term2[0]:  # Different function names
            return False
        
        # Recursively compare arguments
        return all(
            self._deep_compare_terms(t1, t2) 
            for t1, t2 in zip(term1[1], term2[1])
        )

    def _generate_substitutions(self, clauses: List[List[Tuple]]) -> List[Dict]:
        """
        Generate potential variable substitutions.
        
        Args:
            clauses (List[List[Tuple]]): List of clauses
        
        Returns:
            List of possible substitution dictionaries
        """
        substitutions = []
        variables = self.data_generator.variables
        constants = self.data_generator.constants
        
        # Generate some random substitutions
        for _ in range(min(5, len(variables))):
            # Create a substitution mapping
            sub = {
                var: random.choice(constants + variables)
                for var in random.sample(variables, random.randint(1, 3))
            }
            substitutions.append(sub)
        
        return substitutions

    def _enhance_resolvability(self, clauses: List[List[Tuple]], resolvable_pairs: List[Tuple]):
        """
        Attempt to enhance resolvability of clauses.
        
        Args:
            clauses (List[List[Tuple]]): Clauses to modify
            resolvable_pairs (List[Tuple]): Current resolvable pairs
        """
        # If no resolvable pairs, try to create some
        if not resolvable_pairs:
            # Modify some literals to create resolvability
            for clause in clauses:
                if random.random() < 0.3:  # 30% chance of modification
                    self._modify_clause_for_resolution(clause)

    def _modify_clause_for_resolution(self, clause: List[Tuple]):
        """
        Modify a clause to increase its resolvability.
        
        Args:
            clause (List[Tuple]): Clause to modify
        """
        # If clause is empty or too short, do nothing
        if len(clause) < 2:
            return
        
        # Pick a random literal to modify
        lit_idx = random.randint(0, len(clause) - 1)
        current_lit = clause[lit_idx]
        
        # Flip negation or modify arguments to create resolution potential
        modified_lit = (
            current_lit[0],  # Same predicate
            current_lit[1],  # Same arguments
            not current_lit[2]  # Flip negation
        )
        
        # Replace the literal
        clause[lit_idx] = modified_lit

# Main execution for demonstration
if __name__ == "__main__":
    # Create problem generator
    generator = ResolutionProblemGenerator()
    
    # Generate a resolution problem
    problem = generator.generate_resolution_problem(num_clauses=3)
    
    print("Generated Resolution Problem:")
    print("\nClauses:")
    for i, clause in enumerate(problem['clauses']):
        print(f"Clause {i}: {clause}")
    
    print("\nResolvable Pairs:")
    for pair in problem['resolvable_pairs']:
        print(f"Clauses {pair[0]}, {pair[2]} - Literals {pair[1]}, {pair[3]}")
    
    print("\nPotential Substitutions:")
    for sub in problem['substitutions']:
        print(sub)