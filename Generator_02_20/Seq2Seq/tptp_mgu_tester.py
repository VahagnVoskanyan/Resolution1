import os
import sys
import argparse
import json
import torch
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Tuple, Set, Optional

# Import your existing modules
# Adjust these imports based on your actual module structure
from mgu_data_processor import MGUDataProcessor
from clause_generator import ClauseGenerator, Term, Literal, Clause, ClausePair
# Import your trained model - adjust import based on your module structure
from mgu_seq2seq_model import Seq2Seq, load_model

# Try to import a TPTP parser - we'll use pytptp if available
# try:
#     from pytptp import TPTP, TptpProof
#     PYTPTP_AVAILABLE = True
# except ImportError:
#     PYTPTP_AVAILABLE = False
#     print("Warning: pytptp not found. We'll use a simplified TPTP parser.")
PYTPTP_AVAILABLE = False
class TPTPParser:
    """Simplified TPTP parser if pytptp is not available."""
    
    def __init__(self):
        self.clauses = []
    
    def parse_file(self, file_path):
        """Parse a TPTP file and extract clauses."""
        self.clauses = []
        
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            # Skip comments and empty lines
            if not line or line.startswith('%'):
                continue
            
            # Handle cnf formulas (clauses)
            if line.startswith('cnf') or 'cnf(' in line:
                try:
                    # Extract the clause content
                    clause_content = line.split('(', 1)[1]
                    clause_content = clause_content.rsplit(',', 1)[0]
                    # Remove trailing parentheses and period
                    while clause_content.endswith(').') or clause_content.endswith(')'):
                        clause_content = clause_content[:-1]
                    
                    # Further parsing will be done by the clause converter
                    self.clauses.append(clause_content)
                except Exception as e:
                    print(f"Error parsing clause: {line}")
                    print(f"Error details: {e}")
        
        return self.clauses


class TPTPToClauseConverter:
    """Convert TPTP clauses to our internal Clause representation."""
    
    def __init__(self):
        self.variables = set()
        self.constants = set()
        self.predicates = set()
    
    def parse_term(self, term_str: str) -> Term:
        """Parse a term string into a Term object."""
        term_str = term_str.strip()
        
        # In TPTP, variables start with uppercase letters
        is_variable = term_str[0].isupper() if term_str else False
        
        if is_variable:
            self.variables.add(term_str)
        else:
            self.constants.add(term_str)
        
        return Term(name=term_str, is_variable=is_variable)
    
    def parse_literal(self, literal_str: str) -> Optional[Literal]:
        """Parse a literal string into a Literal object."""
        literal_str = literal_str.strip()
        
        # Check for negation
        negated = False
        if literal_str.startswith('~'):
            negated = True
            literal_str = literal_str[1:].strip()
        
        # Extract predicate and terms
        if '(' in literal_str and ')' in literal_str:
            pred_part, terms_part = literal_str.split('(', 1)
            predicate = pred_part.strip()
            terms_str = terms_part.rsplit(')', 1)[0]
            
            # Split terms by comma
            term_strs = [t.strip() for t in terms_str.split(',')]
            terms = [self.parse_term(t) for t in term_strs]
            
            self.predicates.add(predicate)
            
            return Literal(predicate=predicate, terms=terms, negated=negated)
        elif '=' in literal_str:
            # Handle equality literals
            parts = literal_str.split('=')
            if len(parts) == 2:
                left_term = self.parse_term(parts[0].strip())
                right_term = self.parse_term(parts[1].strip())
                
                predicate = "equals"
                self.predicates.add(predicate)
                
                return Literal(predicate=predicate, terms=[left_term, right_term], negated=negated)
        
        # If we couldn't parse it, return None
        print(f"Warning: Could not parse literal: {literal_str}")
        return None
    
    def parse_clause(self, clause_str: str) -> Optional[Clause]:
        """Parse a clause string into a Clause object."""
        clause_str = clause_str.strip()
        
        # Split by disjunction
        literal_strs = clause_str.split('|')
        
        literals = []
        for lit_str in literal_strs:
            lit = self.parse_literal(lit_str.strip())
            if lit:
                literals.append(lit)
        
        if literals:
            return Clause(literals=literals)
        else:
            return None
    
    def convert_tptp_clauses(self, tptp_clauses: List[str]) -> List[Clause]:
        """Convert a list of TPTP clause strings to Clause objects."""
        clauses = []
        
        for clause_str in tptp_clauses:
            clause = self.parse_clause(clause_str)
            if clause:
                clauses.append(clause)
        
        return clauses


class MGUFinder:
    """Find potential MGUs between clauses."""
    
    def __init__(self, converter: TPTPToClauseConverter, generator: 'ClauseGenerator'):
        self.converter = converter
        self.generator = generator
    
    def can_have_complementary_literals(self, lit1: Literal, lit2: Literal) -> bool:
        """Check if two literals can potentially be complementary."""
        return (lit1.predicate == lit2.predicate and 
                lit1.negated != lit2.negated and 
                len(lit1.terms) == len(lit2.terms))
    
    def find_potential_resolvable_pairs(self, clauses: List[Clause]) -> List[ClausePair]:
        """Find pairs of clauses that potentially contain complementary literals."""
        pairs = []
        
        for i, clause1 in enumerate(clauses):
            for j, clause2 in enumerate(clauses):
                if i >= j:  # Avoid duplicate pairs and self-pairs
                    continue
                
                # Find potential complementary literals
                for lit1_idx, lit1 in enumerate(clause1.literals):
                    for lit2_idx, lit2 in enumerate(clause2.literals):
                        if self.can_have_complementary_literals(lit1, lit2):
                            # Try to compute MGU
                            mgu_dict = self.generator.compute_mgu(lit1, lit2)
                            
                            if mgu_dict is not None:
                                # Create a resolved clause
                                resolved = self.generator.resolve_clauses(
                                    clause1, clause2, lit1_idx, lit2_idx, mgu_dict
                                )
                                
                                # Create a ClausePair
                                pair = ClausePair(
                                    clause1=clause1,
                                    clause2=clause2,
                                    complementary_lit_idx1=lit1_idx,
                                    complementary_lit_idx2=lit2_idx,
                                    mgu=mgu_dict,
                                    resolved_clause=resolved
                                )
                                
                                pairs.append(pair)
        
        return pairs


class MGUTester:
    """Test MGU model on TPTP problems."""
    
    def __init__(self, model_path: str, vocab_path: str, device: str = 'cpu'):
        """
        Initialize the MGU tester.
        
        Args:
            model_path: Path to the trained model checkpoint
            vocab_path: Path to the vocabulary file
            device: Device to run the model on ('cpu' or 'cuda')
        """
        self.device = device
        
        # Load vocabulary
        with open(vocab_path, 'r') as f:
            self.token2idx = json.load(f)
        
        self.idx2token = {idx: token for token, idx in self.token2idx.items()}
        
        # Initialize data processor
        self.processor = MGUDataProcessor()
        self.processor.token2idx = self.token2idx
        self.processor.idx2token = self.idx2token
        self.processor.vocab_built = True
        
        # Load model
        self.model = load_model(model_path, self.token2idx, device=device)
        self.model.eval()
        
        # Initialize TPTP parser
        # if PYTPTP_AVAILABLE:
        #     self.tptp = TPTP()
        # else:
        #     self.tptp = TPTPParser() ### --- ###

        self.tptp = TPTPParser()
        # Initialize converter
        self.converter = TPTPToClauseConverter()
        
        # Initialize clause generator for MGU computation
        self.generator = None  # Will be initialized after parsing TPTP files
    
    def format_clause_pair(self, pair: ClausePair) -> dict:
        """Format a clause pair for model input."""
        # Format the input: both clauses as strings
        input_str = f"Clause1: {str(pair.clause1)}\nClause2: {str(pair.clause2)}"
        
        # Format the complementary literals for additional context
        comp_lit1 = str(pair.clause1.literals[pair.complementary_lit_idx1])
        comp_lit2 = str(pair.clause2.literals[pair.complementary_lit_idx2])
        complementary_literals = f"{comp_lit1} and {comp_lit2}"
        
        # Format the output: MGU as a string
        mgu_str = "{" + ", ".join(f"{k}←{v}" for k, v in pair.mgu.items()) + "}"
        
        return {
            "input": input_str,
            "complementary_literals": complementary_literals,
            "mgu": mgu_str,
            "clause1": str(pair.clause1),
            "clause2": str(pair.clause2),
            "mgu_raw": {k: str(v) for k, v in pair.mgu.items()}
        }
    
    def tokenize_input(self, input_text: str, max_length: int = 200):
        """Tokenize input text for the model."""
        tokens = self.processor._tokenize(input_text)
        indices = self.processor._encode(tokens)
        
        # Truncate or pad
        if len(indices) > max_length:
            indices = indices[:max_length]
        else:
            indices = indices + [0] * (max_length - len(indices))
        
        return torch.tensor([indices], dtype=torch.long).to(self.device)
    
    def predict_mgu(self, input_text: str, max_length: int = 200):
        """Predict MGU for input text."""
        input_tensor = self.tokenize_input(input_text, max_length)
        
        with torch.no_grad():
            # Generate output (implementation depends on your model's interface)
            # You might need to adjust this based on your model's implementation
            output_indices = self.model.generate(
                input_tensor, 
                max_length=max_length, 
                sos_token_id=self.token2idx["< SOS >"],
                eos_token_id=self.token2idx["<EOS>"]
            )
            
            # Convert indices back to tokens
            output_tokens = []
            for idx in output_indices[0].cpu().numpy():
                if idx == self.token2idx["<EOS>"]:
                    break
                if idx != self.token2idx["< SOS >"]:
                    token = self.idx2token.get(idx, "<UNK>")
                    output_tokens.append(token)
            
            output_text = " ".join(output_tokens)
            return output_text
    
    def process_tptp_file(self, file_path: str):
        """Process a TPTP file and test the model on it."""
        print(f"Processing TPTP file: {file_path}")
        
        # Parse TPTP file
        if PYTPTP_AVAILABLE:
            problem = self.tptp.parse_file(file_path)
            tptp_clauses = [str(clause) for clause in problem.clauses]
        else:
            tptp_clauses = self.tptp.parse_file(file_path)
        
        print(f"Extracted {len(tptp_clauses)} clauses from TPTP file")
        
        # Convert TPTP clauses to our internal format
        clauses = self.converter.convert_tptp_clauses(tptp_clauses)
        print(f"Converted {len(clauses)} clauses to internal format")
        
        # Initialize clause generator with extracted vocabulary
        self.generator = ClauseGenerator(
            predicates=list(self.converter.predicates),
            variables=list(self.converter.variables),
            constants=list(self.converter.constants)
        )
        
        # Find potential resolvable pairs
        mgu_finder = MGUFinder(self.converter, self.generator)
        pairs = mgu_finder.find_potential_resolvable_pairs(clauses)
        print(f"Found {len(pairs)} potential resolvable clause pairs")
        
        if not pairs:
            print("No resolvable pairs found in this problem.")
            return []
        
        # Test model on each pair
        results = []
        for pair in tqdm(pairs, desc="Testing model on clause pairs"):
            # Format pair for model input
            formatted_pair = self.format_clause_pair(pair)
            
            # Predict MGU
            input_text = formatted_pair["input"]
            predicted_mgu = self.predict_mgu(input_text)
            
            # Evaluate prediction
            actual_mgu = formatted_pair["mgu"]
            
            results.append({
                "input": input_text,
                "predicted_mgu": predicted_mgu,
                "actual_mgu": actual_mgu,
                "correct": self.evaluate_mgu_prediction(predicted_mgu, actual_mgu)
            })
        
        return results
    
    def evaluate_mgu_prediction(self, predicted_mgu: str, actual_mgu: str) -> bool:
        """
        Evaluate if the predicted MGU is equivalent to the actual MGU.
        
        Note: This is a simplified evaluation that checks for string equality.
        A more sophisticated evaluation would check for logical equivalence of substitutions.
        """
        # Clean up strings
        pred_clean = predicted_mgu.replace(" ", "").strip("{}")
        actual_clean = actual_mgu.replace(" ", "").strip("{}")
        
        # Simple case: exact match
        if pred_clean == actual_clean:
            return True
        
        # TODO: Implement more sophisticated evaluation that checks for
        # logical equivalence of substitutions
        
        return False
    
    def process_tptp_directory(self, directory_path: str, max_files: int = None):
        """Process all TPTP files in a directory."""
        all_results = []
        
        # Find all TPTP files
        tptp_files = []
        for root, _, files in os.walk(directory_path):
            for file in files:
                if file.endswith('.p') or file.endswith('.tptp'):
                    tptp_files.append(os.path.join(root, file))
        
        if max_files:
            tptp_files = tptp_files[:max_files]
        
        print(f"Found {len(tptp_files)} TPTP files")
        
        # Process each file
        for file_path in tptp_files:
            try:
                results = self.process_tptp_file(file_path)
                if results:
                    all_results.extend(results)
                    print(f"Results for {file_path}:")
                    print(f"  - Pairs tested: {len(results)}")
                    correct = sum(1 for r in results if r["correct"])
                    print(f"  - Correct predictions: {correct}/{len(results)} ({correct/len(results)*100:.2f}%)")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        
        # Summarize results
        if all_results:
            correct = sum(1 for r in all_results if r["correct"])
            print(f"\nOverall results:")
            print(f"  - Total pairs tested: {len(all_results)}")
            print(f"  - Total correct predictions: {correct}/{len(all_results)} ({correct/len(all_results)*100:.2f}%)")
        else:
            print("\nNo results to summarize.")
        
        return all_results
    
    def save_results(self, results, output_file: str):
        """Save results to a JSON file."""
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Test MGU model on TPTP problems')
    parser.add_argument('--model-path', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--vocab-path', type=str, required=True, help='Path to vocabulary file')
    parser.add_argument('--tptp-path', type=str, required=True, help='Path to TPTP problem or directory')
    parser.add_argument('--output-file', type=str, default='mgu_test_results.json', help='Output file for results')
    parser.add_argument('--max-files', type=int, default=None, help='Maximum number of TPTP files to process')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run model on (cpu or cuda)')
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = MGUTester(args.model_path, args.vocab_path, args.device)
    
    # Process TPTP problems
    if os.path.isdir(args.tptp_path):
        results = tester.process_tptp_directory(args.tptp_path, args.max_files)
    else:
        results = tester.process_tptp_file(args.tptp_path)
    
    # Save results
    if results:
        tester.save_results(results, args.output_file)


if __name__ == "__main__":
    main()
