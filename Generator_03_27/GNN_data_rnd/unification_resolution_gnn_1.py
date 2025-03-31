from typing import List, Dict, Optional

class UnificationResolution:
    def __init__(
        self,
        variables: List[str] = None,
    ):
        self.variables = variables or ["X", "Y", "Z"]#, "U", "V", "W"]

    # -------------------------------------------------------------------------
    # 2) Resolution / Unification Logic
    # -------------------------------------------------------------------------
    def parse_literal(self, lit_str: str):
        """
        Returns (sign, predicate, [arguments]).
          sign: +1 or -1
          predicate: e.g., "P"
          arguments: list of strings for each argument term
        """
        sign = +1
        core = lit_str.strip()
        if core.startswith("¬"):
            sign = -1
            core = core[1:].strip()  # remove negation
        # Now core should be like "Pred1(...)" or "Pred2(...)"
        pred = core.split("(", 1)[0]
        inside = core[len(pred) + 1 : -1]  # everything inside parentheses
        # split arguments by comma
        if inside.strip() == "":
            args = []
        else:
            args = [arg.strip() for arg in inside.split(",")]
        return sign, pred, args
    
    def occurs_check(self, var: str, term: str, subst: Dict[str, str]) -> bool:
        """
        Returns True if 'var' occurs anywhere in 'term' under the current substitution,
        i.e. if binding var to term would create a circular reference.
        """
        # First apply existing substitution to 'term'
        term = self.apply_subst_to_term(term, subst)
    
        if var == term:
            return True
    
        # If it's a function, check each sub-argument
        if "(" in term and term.endswith(")"):
            func_symbol, arg_strs = self._parse_function(term)
            for arg in arg_strs:
                if self.occurs_check(var, arg, subst):
                    return True
        return False

    def unify(self, termA: str, termB: str, subst: Dict[str, str] = None) -> Optional[Dict[str, str]]:
        """
        Perform first-order unification with occurs-check.
    
        If successful, return a unifier dict that can be used to make termA and termB identical.
        If unification is impossible, return None.
        """
        if subst is None:
            subst = {}
    
        # Step 1: Apply existing substitutions
        termA = self.apply_subst_to_term(termA, subst)
        termB = self.apply_subst_to_term(termB, subst)
    
        # Step 2: Check if they're now identical
        if termA == termB:
            return subst
    
        # Step 3: If termA is a variable
        if termA in self.variables:
            # Occurs-check: if termA appears in termB, fail
            if self.occurs_check(termA, termB, subst):
                return None
            subst[termA] = termB
            return subst
    
        # Step 4: If termB is a variable
        if termB in self.variables:
            if self.occurs_check(termB, termA, subst):
                return None
            subst[termB] = termA
            return subst
    
        # Step 5: If both are function calls, parse them
        if "(" in termA and termA.endswith(")") and "(" in termB and termB.endswith(")"):
            funcA, argsA = self._parse_function(termA)
            funcB, argsB = self._parse_function(termB)
            # must have same function symbol and same number of arguments
            if funcA != funcB or len(argsA) != len(argsB):
                return None
            for a_i, b_i in zip(argsA, argsB):
                subst = self.unify(a_i, b_i, subst)
                if subst is None:
                    return None
            return subst
    
        # Step 6: If they differ and are both constants (or function vs constant mismatch), fail
        if termA != termB:
            return None
    
        # If we get here, we can unify
        return subst

    def _parse_function(self, func_str: str):
        """
        Parse a function string like func_f(X, const_a) => ("func_f", ["X", "const_a"]).
        """
        func_symbol = func_str.split("(", 1)[0]
        inside = func_str[len(func_symbol) + 1 : -1]  # contents inside parentheses
        arg_strs = [arg.strip() for arg in inside.split(",")]
        return func_symbol, arg_strs

    def can_resolve(self, litA: str, litB: str) -> Optional[Dict[str, str]]:
        """
        Returns a unifier (dict) if litA and litB can be resolved, else None.
        i.e., sign(litA) = - sign(litB), same predicate, and arguments unifiable.
        """
        sA, pA, argsA = self.parse_literal(litA)
        sB, pB, argsB = self.parse_literal(litB)
        # Opposite signs, same predicate, same number of arguments
        if pA != pB:
            return None
        if sA + sB != 0:
            return None
        if len(argsA) != len(argsB):
            return None

        # unify each argument in turn
        unifier = {}
        for a, b in zip(argsA, argsB):
            unifier = self.unify(a, b, unifier)
            if unifier is None:
                return None
        return unifier

    def apply_subst_to_term(self, term: str, subst: Dict[str, str]) -> str:
        """
        Recursively apply substitution dict to a single term.
        For example, if subst={"X":"const_a"}, apply_subst_to_term("X") => "const_a".
        If we see a function, we apply to each argument.
        """
        # Keep substituting if the term itself is in subst
        while term in subst:
            term = subst[term]

        # If it's a function call, parse & apply recursively to arguments
        if "(" in term and term.endswith(")"):
            func_symbol, arg_strs = self._parse_function(term)
            new_args = [self.apply_subst_to_term(arg, subst) for arg in arg_strs]
            inside = ", ".join(new_args)
            return f"{func_symbol}({inside})"

        # otherwise, it's either a constant or variable not bound in subst
        return term

    def apply_subst_to_literal(self, literal: str, subst: Dict[str, str]) -> str:
        """
        Apply a substitution to an entire literal string (including sign, predicate, and arguments).
        """
        sign, pred, args = self.parse_literal(literal)
        new_args = [self.apply_subst_to_term(arg, subst) for arg in args]
        args_str = ", ".join(new_args)
        if sign > 0:
            return f"{pred}({args_str})"
        else:
            return f"¬{pred}({args_str})"

    def resolve_clauses(self, clauseA: List[str], clauseB: List[str], idxA: int, idxB: int) -> Optional[List[str]]:
        """
        Resolve clauseA[idxA] and clauseB[idxB].
        Returns the new resolvent clause, or None if not resolvable.
        """
        litA = clauseA[idxA]
        litB = clauseB[idxB]
        unifier = self.can_resolve(litA, litB)
        if unifier is None:
            return None
        # build the resolvent
        new_clause = []
        for i, la in enumerate(clauseA):
            if i == idxA:
                continue
            new_clause.append(self.apply_subst_to_literal(la, unifier))
        for j, lb in enumerate(clauseB):
            if j == idxB:
                continue
            new_clause.append(self.apply_subst_to_literal(lb, unifier))
        # Remove duplicates
        new_clause = list(set(new_clause))
        return new_clause