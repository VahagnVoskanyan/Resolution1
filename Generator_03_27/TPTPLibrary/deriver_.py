import re
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Set

@dataclass
class Clause:
    id: str
    content: str
    type: str  # 'plain', 'axiom', etc.
    literals: List[str]
    source: Optional[List[str]] = None

@dataclass
class ResolutionStep:
    result_id: str
    result_content: str
    parent1_id: str
    parent1_content: str
    parent2_id: str
    parent2_content: str
    likely_unified_literals: List[Tuple[str, str]]

def clean_content(content: str) -> str:
    """Clean up clause content for easier parsing"""
    content = content.strip()
    if content.startswith('(') and content.endswith(')'):
        content = content[1:-1].strip()
    return content

def extract_literals(content: str) -> List[str]:
    """Extract individual literals from a clause content"""
    # Remove universal quantifiers
    content = re.sub(r'!\s*\[[^\]]+\]\s*:', '', content)
    content = clean_content(content)
    
    # Split by disjunction | and clean up
    literals = []
    # Handle parentheses correctly when splitting by |
    paren_level = 0
    current_lit = ""
    
    for char in content:
        if char == '(' or char == '[':
            paren_level += 1
            current_lit += char
        elif char == ')' or char == ']':
            paren_level -= 1
            current_lit += char
        elif char == '|' and paren_level == 0:
            literals.append(current_lit.strip())
            current_lit = ""
        else:
            current_lit += char
    
    if current_lit:
        literals.append(current_lit.strip())
    
    return [lit for lit in literals if lit]

def parse_fof_statement(text: str) -> List[Tuple[str, str, str, List[str]]]:
    """Parse FOF statements from a block of text"""
    results = []
    
    # Pattern to match FOF statements more robustly
    pattern = r'fof\(([^,]+),\s*([^,]+),\s*\(\s*(.*?)\s*\)\s*,\s*([^)]+)\)'
    
    # Find all matches in the text
    matches = re.finditer(pattern, text, re.DOTALL)
    
    for match in matches:
        clause_id = match.group(1).strip()
        clause_type = match.group(2).strip()
        content = match.group(3).strip()
        source_info = match.group(4).strip()
        
        source_clauses = []
        if "inference" in source_info:
            # Extract source clauses from inference info
            src_match = re.search(r'inference\([^,]+,\s*\[\],\s*\[([^]]+)\]\)', source_info)
            if src_match:
                source_clauses = [s.strip() for s in src_match.group(1).split(',')]
        
        results.append((clause_id, clause_type, content, source_clauses))
    
    return results

def analyze_literals(literals1: List[str], literals2: List[str]) -> List[Tuple[str, str]]:
    """
    Analyze two sets of literals to find potential resolution pairs
    (one positive, one negative with matching predicate)
    """
    potential_pairs = []
    
    for lit1 in literals1:
        is_neg1 = lit1.strip().startswith('~')
        base_lit1 = lit1.strip()[1:] if is_neg1 else lit1.strip()
        pred1 = base_lit1.split('(')[0] if '(' in base_lit1 else base_lit1
        
        for lit2 in literals2:
            is_neg2 = lit2.strip().startswith('~')
            
            # Only consider literals with opposite polarity
            if is_neg1 == is_neg2:
                continue
                
            base_lit2 = lit2.strip()[1:] if is_neg2 else lit2.strip()
            pred2 = base_lit2.split('(')[0] if '(' in base_lit2 else base_lit2
            
            # Check if predicates match
            if pred1 == pred2:
                potential_pairs.append((lit1, lit2))
    
    return potential_pairs

def clean_vampire_output(text: str) -> str:
    """Clean up Vampire output for easier parsing"""
    # Remove timestamps and other non-essential information
    lines = []
    for line in text.split('\n'):
        # Remove timestamps and other prefixes
        cleaned_line = re.sub(r'^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\s+', '', line)
        lines.append(cleaned_line)
    
    return '\n'.join(lines)

def parse_vampire_output(output_text: str) -> Tuple[Dict[str, Clause], List[ResolutionStep]]:
    """Parse Vampire output to identify clauses and resolution steps"""
    # Clean the output text first
    cleaned_text = clean_vampire_output(output_text)
    
    # Parse all FOF statements
    fof_statements = parse_fof_statement(cleaned_text)
    
    # Build clause dictionary
    clauses = {}
    for clause_id, clause_type, content, source_clauses in fof_statements:
        literals = extract_literals(content)
        clauses[clause_id] = Clause(
            id=clause_id,
            content=content,
            type=clause_type,
            literals=literals,
            source=source_clauses
        )
    
    # Identify resolution steps
    resolution_steps = []
    for clause_id, clause in clauses.items():
        if clause.source and len(clause.source) == 2:  # Standard binary resolution
            parent1_id, parent2_id = clause.source
            
            if parent1_id in clauses and parent2_id in clauses:
                parent1 = clauses[parent1_id]
                parent2 = clauses[parent2_id]
                
                # Find potential unified literals
                potential_unified = analyze_literals(parent1.literals, parent2.literals)
                
                resolution_steps.append(ResolutionStep(
                    result_id=clause_id,
                    result_content=clause.content,
                    parent1_id=parent1_id,
                    parent1_content=parent1.content,
                    parent2_id=parent2_id,
                    parent2_content=parent2.content,
                    likely_unified_literals=potential_unified
                ))
    
    return clauses, resolution_steps

def print_resolution_steps(resolution_steps: List[ResolutionStep]):
    """Print the resolution steps in a readable format"""
    print("Resolution Steps Analysis:")
    print("==========================")
    
    for i, step in enumerate(resolution_steps, 1):
        print(f"Step {i}:")
        print(f"Result: {step.result_id} - {step.result_content}")
        print(f"Parent 1: {step.parent1_id} - {step.parent1_content}")
        print(f"Parent 2: {step.parent2_id} - {step.parent2_content}")
        
        if step.likely_unified_literals:
            print("Likely unified literals:")
            for lit1, lit2 in step.likely_unified_literals:
                print(f"  {lit1} ⟷ {lit2}")
        else:
            print("No clear unification candidates identified")
            
        print("-" * 80)

def analyze_vampire_output(input_file_path: str):
    """Analyze a Vampire output file to extract resolution information"""
    with open(input_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    clauses, resolution_steps = parse_vampire_output(content)
    
    print(f"Found {len(clauses)} clauses and {len(resolution_steps)} resolution steps.")
    
    if resolution_steps:
        print_resolution_steps(resolution_steps)
    else:
        print("No resolution steps identified. Try providing more complete Vampire output.")
    
    return clauses, resolution_steps

# Example for direct text input - can be useful for debugging
def analyze_vampire_text(text: str):
    clauses, resolution_steps = parse_vampire_output(text)
    
    print(f"Found {len(clauses)} clauses and {len(resolution_steps)} resolution steps.")
    
    if resolution_steps:
        print_resolution_steps(resolution_steps)
    else:
        print("No resolution steps identified in the provided text.")
    
    return clauses, resolution_steps

# Example text for direct testing
example_text = """
fof(f1071,plain,(
  ( ! [X2,X0,X1] : (~product(X2,b,X0) | ~product(X0,g,X1) | product(X2,d,X1)) )),
  inference(resolution,[],[f5,f22])).
fof(f19,axiom,(
  product(a,b,c)),
  file('/vampire/examples/CAT001-1.p',unknown)).
fof(f2306,plain,(
  ( ! [X0] : (~product(c,g,X0) | product(a,d,X0)) )),
  inference(resolution,[],[f1071,f19])).
"""

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        analyze_vampire_output(sys.argv[1])
    else:
        print("No file provided. Running example analysis...")
        analyze_vampire_text(example_text)

# python deriver_.py Output1/vampire_output.txt