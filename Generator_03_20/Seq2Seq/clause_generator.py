import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Union


@dataclass
class Term:
    """Represents a term in first-order logic (either a variable or a constant)."""
    name: str
    is_variable: bool

    def __str__(self):
        return self.name


@dataclass
class Literal:
    """Represents a literal (predicate applied to terms)."""
    predicate: str
    terms: List[Term]
    negated: bool = False
    
    def __str__(self):
        terms_str = ", ".join(str(term) for term in self.terms)
        if self.negated:
            return f"¬{self.predicate}({terms_str})"
        return f"{self.predicate}({terms_str})"
    
    def complement(self):
        """Returns a copy of this literal with negation flipped."""
        return Literal(
            predicate=self.predicate,
            terms=self.terms.copy(),
            negated=not self.negated
        )


@dataclass
class Clause:
    """Represents a disjunction of literals."""
    literals: List[Literal]
    
    def __str__(self):
        return " ∨ ".join(str(lit) for lit in self.literals)


@dataclass
class ClausePair:
    """A pair of clauses with a complementary literal and resulting MGU."""
    clause1: Clause
    clause2: Clause
    complementary_lit_idx1: int
    complementary_lit_idx2: int
    mgu: Dict[str, Term]
    resolved_clause: Clause
    
    def __str__(self):
        return (f"Clause 1: {self.clause1}\n"
                f"Clause 2: {self.clause2}\n"
                f"Complementary literals: {self.clause1.literals[self.complementary_lit_idx1]} and "
                f"{self.clause2.literals[self.complementary_lit_idx2]}\n"
                f"MGU: {{{', '.join(f'{k}←{v}' for k, v in self.mgu.items())}}}\n"
                f"Resolved: {self.resolved_clause}")


class ClauseGenerator:
    def __init__(
        self,
        predicates: List[str] = None,
        variables: List[str] = None,
        constants: List[str] = None,
        max_clause_length: int = 3,
        max_term_arity: int = 3
    ):
        """Initialize the clause generator with customizable parameters.
        
        Args:
            predicates: List of predicate names to use
            variables: List of variable names to use
            constants: List of constant names to use
            max_clause_length: Maximum number of literals in a clause
            max_term_arity: Maximum number of terms in a predicate
        """
        self.predicates = predicates or ["P", "Q", "R", "S", "T"]
        self.variables = variables or ["x", "y", "z", "u", "v", "w"]
        self.constants = constants or ["a", "b", "c", "d", "e"]
        self.max_clause_length = max_clause_length
        self.max_term_arity = max_term_arity
    
    def generate_term(self, force_variable: bool = False, force_constant: bool = False) -> Term:
        """Generate a random term (variable or constant)."""
        if force_variable and force_constant:
            raise ValueError("Cannot force both variable and constant")
        
        if force_variable:
            is_variable = True
        elif force_constant:
            is_variable = False
        else:
            is_variable = random.random() < 0.7  # 70% chance of variable
        
        if is_variable:
            name = random.choice(self.variables)
        else:
            name = random.choice(self.constants)
            
        return Term(name=name, is_variable=is_variable)
    
    def generate_literal(self, predicate: str = None, terms: List[Term] = None, negated: bool = None) -> Literal:
        """Generate a random literal or one with specific properties."""
        if predicate is None:
            predicate = random.choice(self.predicates)
        
        if terms is None:
            arity = random.randint(1, self.max_term_arity)
            terms = [self.generate_term() for _ in range(arity)]
        
        if negated is None:
            negated = random.random() < 0.5
            
        return Literal(predicate=predicate, terms=terms, negated=negated)
    
    def generate_complementary_literal_pair(self) -> Tuple[Literal, Literal]:
        """Generate a pair of complementary literals that can be resolved."""
        # Generate the first literal
        predicate = random.choice(self.predicates)
        arity = random.randint(1, self.max_term_arity)
        
        # Create terms for the first literal with a mix of variables and constants
        terms1 = []
        for _ in range(arity):
            # Force at least one variable for unification
            force_var = random.random() < 0.7
            terms1.append(self.generate_term(force_variable=force_var))
        
        lit1 = Literal(predicate=predicate, terms=terms1, negated=random.choice([True, False]))
        
        # For the second literal, use the same predicate but potentially different terms
        # that can be unified with the first literal's terms
        terms2 = []
        for term in terms1:
            if term.is_variable:
                # For variables in first literal, we can use either:
                # 1. Same variable
                # 2. Different variable
                # 3. Constant
                choice = random.randint(1, 3)
                if choice == 1:
                    # Same variable
                    terms2.append(Term(name=term.name, is_variable=True))
                elif choice == 2:
                    # Different variable
                    var_name = random.choice([v for v in self.variables if v != term.name])
                    terms2.append(Term(name=var_name, is_variable=True))
                else:
                    # Constant
                    const_name = random.choice(self.constants)
                    terms2.append(Term(name=const_name, is_variable=False))
            else:
                # For constants in first literal, we can use either:
                # 1. Same constant (most likely to unify)
                # 2. Variable (will unify)
                # 3. Different constant (won't unify, but we allow this occasionally)
                choice = random.choices([1, 2, 3], weights=[0.5, 0.4, 0.1])[0]
                if choice == 1:
                    # Same constant
                    terms2.append(Term(name=term.name, is_variable=False))
                elif choice == 2:
                    # Variable
                    var_name = random.choice(self.variables)
                    terms2.append(Term(name=var_name, is_variable=True))
                else:
                    # Different constant (will make unification fail)
                    const_options = [c for c in self.constants if c != term.name]
                    if const_options:
                        const_name = random.choice(const_options)
                        terms2.append(Term(name=const_name, is_variable=False))
                    else:
                        # Fallback if no other constants available
                        terms2.append(Term(name=term.name, is_variable=False))
        
        # Make sure the literals have opposite negation
        lit2 = Literal(predicate=predicate, terms=terms2, negated=not lit1.negated)
        
        return lit1, lit2
    
    def can_unify(self, term1: Term, term2: Term) -> bool:
        """Check if two terms can be unified."""
        # Two constants unify only if they are identical
        if not term1.is_variable and not term2.is_variable:
            return term1.name == term2.name
        
        # Variables can unify with anything
        return True
    
    def compute_mgu(self, lit1: Literal, lit2: Literal) -> Optional[Dict[str, Term]]:
        """Compute the Most General Unifier for two literals if they can be resolved."""
        # Check if predicates match and negations are complementary
        if lit1.predicate != lit2.predicate or lit1.negated == lit2.negated:
            return None
        
        # Check if arity matches
        if len(lit1.terms) != len(lit2.terms):
            return None
        
        # Initialize substitution dictionary
        substitution = {}
        
        # Try to unify each pair of terms
        for t1, t2 in zip(lit1.terms, lit2.terms):
            if not self.can_unify(t1, t2):
                return None
            
            if t1.is_variable:
                # If t1 is already bound to something else
                if t1.name in substitution and substitution[t1.name] != t2:
                    return None
                substitution[t1.name] = t2
            elif t2.is_variable:
                # If t2 is already bound to something else
                if t2.name in substitution and substitution[t2.name] != t1:
                    return None
                substitution[t2.name] = t1
            elif t1.name != t2.name:
                # Two different constants can't unify
                return None
        
        return substitution
    
    def apply_substitution(self, literal: Literal, substitution: Dict[str, Term]) -> Literal:
        """Apply a substitution to a literal."""
        new_terms = []
        for term in literal.terms:
            if term.is_variable and term.name in substitution:
                new_terms.append(substitution[term.name])
            else:
                new_terms.append(term)
        
        return Literal(predicate=literal.predicate, terms=new_terms, negated=literal.negated)
    
    def resolve_clauses(self, clause1: Clause, clause2: Clause, lit_idx1: int, lit_idx2: int, mgu: Dict[str, Term]) -> Clause:
        """Resolve two clauses using the given MGU and complementary literals."""
        resolved_literals = []
        
        # Add literals from first clause (except the one being resolved)
        for i, lit in enumerate(clause1.literals):
            if i != lit_idx1:
                resolved_literals.append(self.apply_substitution(lit, mgu))
        
        # Add literals from second clause (except the one being resolved)
        for i, lit in enumerate(clause2.literals):
            if i != lit_idx2:
                resolved_literals.append(self.apply_substitution(lit, mgu))
        
        return Clause(literals=resolved_literals)
    
    def generate_clause_pair(self) -> ClausePair:
        """Generate a pair of clauses with complementary literals that can be resolved."""
        # Generate the complementary literals
        lit1, lit2 = self.generate_complementary_literal_pair()
        
        # Compute the MGU
        mgu_dict = self.compute_mgu(lit1, lit2)
        
        # If unification fails, try again
        while mgu_dict is None:
            lit1, lit2 = self.generate_complementary_literal_pair()
            mgu_dict = self.compute_mgu(lit1, lit2)
        
        # Generate the rest of the clauses
        clause1_length = random.randint(1, self.max_clause_length)
        clause2_length = random.randint(1, self.max_clause_length)
        
        # Ensure at least one literal in each clause
        clause1_length = max(1, clause1_length)
        clause2_length = max(1, clause2_length)
        
        # Generate other literals for clause1
        clause1_literals = [lit1]
        for _ in range(clause1_length - 1):
            clause1_literals.append(self.generate_literal())
        
        # Generate other literals for clause2
        clause2_literals = [lit2]
        for _ in range(clause2_length - 1):
            clause2_literals.append(self.generate_literal())
        
        # Create the clauses
        clause1 = Clause(literals=clause1_literals)
        clause2 = Clause(literals=clause2_literals)
        
        # Random position for complementary literals
        lit1_idx = 0  # We already put lit1 at position 0
        lit2_idx = 0  # We already put lit2 at position 0
        
        # Convert MGU dict to the expected format
        mgu = {k: v for k, v in mgu_dict.items()}
        
        # Resolve the clauses
        resolved = self.resolve_clauses(clause1, clause2, lit1_idx, lit2_idx, mgu_dict)
        
        return ClausePair(
            clause1=clause1,
            clause2=clause2,
            complementary_lit_idx1=lit1_idx,
            complementary_lit_idx2=lit2_idx,
            mgu=mgu,
            resolved_clause=resolved
        )
    
    def generate_dataset(self, size: int) -> List[ClausePair]:
        """Generate a dataset of clause pairs for MGU training."""
        dataset = []
        for _ in range(size):
            pair = self.generate_clause_pair()
            dataset.append(pair)
        return dataset


# Example usage
def main():
    generator = ClauseGenerator()
    
    # Generate a single example
    pair = generator.generate_clause_pair()
    print("Generated Clause Pair:")
    print(pair)
    
    # Generate a small dataset
    print("\nGenerating dataset of 5 examples:")
    dataset = generator.generate_dataset(5)
    for i, pair in enumerate(dataset):
        print(f"\nExample {i+1}:")
        print(pair)


if __name__ == "__main__":
    main()
