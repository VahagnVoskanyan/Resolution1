import random
import itertools
import json
from typing import List, Dict, Any, Optional

from random_generators_gnn_1 import RandomGenerators
from unification_resolution_gnn_1 import UnificationResolution

# Returns Dataset with best pairs
class ResolutionDataGenerator:
    def __init__(
        self,
        predicates: List[str] = None,
        variables: List[str] = None,
        constants: List[str] = None,
        functions: List[str] = None,
        max_clause_length: int = 3,
        max_term_arity: int = 3,
        max_function_arity: int = 1,
        seed=42
    ):
        
        """
        Combine random generation with resolution logic to create full examples.
        """
        # We create one instance of the random generator
        self.rgen = RandomGenerators(
            predicates=predicates,
            variables=variables,
            constants=constants,
            functions=functions,
            max_clause_length=max_clause_length,
            max_term_arity=max_term_arity,
            max_function_arity=max_function_arity,
            seed=seed
        )

        # We create one instance of the resolution/unification logic
        self.resolver = UnificationResolution(
            #variables=(variables or ["x", "y", "z"])
        )

        random.seed(seed)

    # -------------------------------------------------------------------------
    # 3) Scoring & Dataset Generation
    # -------------------------------------------------------------------------
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

    def generate_dataset_entry(self, min_clauses=3, max_clauses=10) -> Dict[str, Any]:
        """
        Generate one dataset example:
         1) Create random clauses
         2) Find all resolvable pairs
         3) Score them
         4) Return the single best pair
        """
        n_clauses = random.randint(min_clauses, max_clauses)
        clauses = [self.rgen.generate_random_clause_preemptive() for _ in range(n_clauses)]

        # 1) Compute frequencies for these clauses (improvement (4))
        freq_map = self.compute_predicate_frequencies(clauses)

        all_pairs = []
        for i, j in itertools.combinations(range(n_clauses), 2):
            for idxA in range(len(clauses[i])):
                for idxB in range(len(clauses[j])):
                    # 2) Pass freq_map into the scoring function
                    sc = self.score_resolution_pair(clauses, i, j, idxA, idxB, freq_map)
                    if sc < float("inf"):
                        all_pairs.append(((i, idxA), (j, idxB), sc))

        # (Optional) If you also want to consider resolution within the same clause:
        # for i in range(n_clauses):
        #     for idxA in range(len(clauses[i])):
        #         for idxB in range(idxA + 1, len(clauses[i])):
        #             sc = self.score_resolution_pair(clauses, i, i, idxA, idxB)
        #             if sc < float("inf"):
        #                 all_pairs.append(((i, idxA), (i, idxB), sc))

        # If no resolvable pairs, best_pair is None
        if not all_pairs:
            return {
                "clauses": clauses,
                "resolvable_pairs": [],
                "best_pair": None
            }

        # Choose pair with minimal score
        best = min(all_pairs, key=lambda x: x[2])
        best_pair = {
            "clauseA_index": best[0][0],
            "literalA_index": best[0][1],
            "clauseB_index": best[1][0],
            "literalB_index": best[1][1],
            "score": best[2]
        }

        # Package all pairs
        resolvable_pairs = [
            {
                "clauseA_index": p[0][0],
                "literalA_index": p[0][1],
                "clauseB_index": p[1][0],
                "literalB_index": p[1][1],
                "score": p[2]
            }
            for p in all_pairs
        ]

        return {
            "clauses": clauses,
            "resolvable_pairs": resolvable_pairs,
            "best_pair": best_pair
        }

    def create_dataset(
        self,
        n_examples: int = 20,
        min_clauses: int = 3,
        max_clauses: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Create a dataset of n_examples, each with random clauses and a best-pair label.
        """
        dataset = []
        for _ in range(n_examples):
            entry = self.generate_dataset_entry(min_clauses=min_clauses, max_clauses=max_clauses)
            dataset.append(entry)
        return dataset
    
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
    
if __name__ == "__main__":
    generator = ResolutionDataGenerator(seed=2023)
    data = generator.create_dataset(n_examples=800, min_clauses=3, max_clauses=10)

    # Count how many entries have a valid best_pair (i.e., not None)
    valid_best_pair_count = sum(1 for entry in data if entry["best_pair"] is not None)
    print(f"Found {valid_best_pair_count} best pairs out of {len(data)} examples.")

    # Count and print details for empty clause resolvents
    empty_clause_count = 0
    for idx, entry in enumerate(data):
        for pair in entry["resolvable_pairs"]:
            if pair["score"] == -1000.0:
                empty_clause_count += 1
                #print(f"Empty clause in example {idx}: "
                #      f"clauseA_index={pair['clauseA_index']} (literal index {pair['literalA_index']}), "
                #      f"clauseB_index={pair['clauseB_index']} (literal index {pair['literalB_index']}).")
    print(f"Found {empty_clause_count} empty clause resolvents in total.")

    # Print or save
    # for d in data:
        # print(d)


    # dataset = generator.create_dataset(n_examples=5, min_clauses=3, max_clauses=5)
    # Write to a JSONL file
    # with open("toy_gnn_dataset.jsonl", "w") as f:
    #     for item in data:
    #         f.write(json.dumps(item) + "\n")

    # # Print out the first sample
    # import pprint
    # pprint.pprint(dataset[0])