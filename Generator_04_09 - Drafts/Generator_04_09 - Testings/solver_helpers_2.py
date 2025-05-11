from typing import List, Dict, Any, Optional


from unification_resolution import UnificationResolution
class ResolutionDataGenerator:
    def __init__(self):
        # We create one instance of the resolution/unification logic
        self.resolver = UnificationResolution(
            #variables=(variables or ["x", "y", "z"])
        )

    def score_resolution_pair(
        self,
        clauses: List[List[str]],
        i: int,
        j: int,
        idxA: int,
        idxB: int,
        freq_map: Dict[str, int]
    ) -> float:
        """
        Returns a numeric score for resolving clauses[i] and clauses[j] at literal idxA, idxB.
        
        - If the resolvent is empty, we return -1000 (lowest/best).
        - Otherwise, we measure how much the total literal count changes after resolution.
          Lower is better, so a negative difference (new_total < old_total) indicates a bigger reduction.
        - If either clause is a unit clause, we subtract a bonus from the score
          so that "unit-preference" pairs rank more favorably.
        - a frequency-based term to either reward or penalize certain predicates.
        """
        resolvent = self.resolver.resolve_clauses(clauses[i], clauses[j], idxA, idxB)
        if resolvent is None:
            return float("inf")  # not resolvable => can't use

        # Empty resolvent => contradiction => best possible
        if len(resolvent) == 0:
            return -1000.0

        # As a simple heuristic, let's measure the sum of lengths if we replaced
        # these two clauses by the resolvent (like a single resolution step).
        new_clauses = list(clauses)
        # remove old ones, add resolvent
        # (in real resolution we might keep old clauses around, but for scoring let's do this simplification)
        c_i = new_clauses[i]
        c_j = new_clauses[j]
        new_clauses.remove(c_i)
        # watch out if i != j
        if i != j:
            new_clauses.remove(c_j)
        new_clauses.append(resolvent)

        # sum up all literal counts
        #total_lits = sum(len(cl) for cl in new_clauses)

        # Count total literals before and after the resolution
        old_total = sum(len(c) for c in clauses)
        new_total = sum(len(c) for c in new_clauses)
        # Our baseline score is the difference: new_total - old_total (improvement (3))
        score = (new_total - old_total)

        # ---------- UNIT-PREFERENCE BONUS ---------- (improvement (1))
        # If either original clause is a unit clause, prefer it by lowering the score
        if len(clauses[i]) == 1 or len(clauses[j]) == 1:
            score -= 3.0

        # --- FREQUENCY-BASED BONUS ---  (improvement (4))
        # Parse the two literals we are resolving and sum up their frequencies
        litA = clauses[i][idxA]
        litB = clauses[j][idxB]
        sA, pA, _ = self.resolver.parse_literal(litA)
        sB, pB, _ = self.resolver.parse_literal(litB)
        freqA = freq_map.get(pA, 0)
        freqB = freq_map.get(pB, 0)

        # If we want to *reward* unifying very frequent predicates:
        freq_sum = freqA + freqB
        score -= 0.5 * freq_sum  # tweak 0.5 or any alpha you want

        return score
    
    # Frequency-Based Literal Selection Heuristic (improvement (4))
    def compute_predicate_frequencies(self, clauses: List[List[str]]) -> Dict[str, int]:
        """
        Returns a dictionary mapping predicate -> how many times it appears across all clauses.
        """
        freq = {}
        for clause in clauses:
            for lit in clause:
                sign, pred, args = self.resolver.parse_literal(lit)
                freq[pred] = freq.get(pred, 0) + 1
        return freq